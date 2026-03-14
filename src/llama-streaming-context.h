#pragma once

/**
 * llama-streaming-context.h
 *
 * Bridges StreamingWeightMapper into llama.cpp's model loader.
 *
 * Instead of loading all tensor data upfront (which OOMs on 400GB models),
 * this context:
 *   1. Registers tensors as "virtual" during load_all_data() — no data allocation
 *   2. Serves tensor data on demand via get_tensor_data() at inference time
 *   3. Manages an LRU cache so only a configurable window stays resident
 *
 * CUDA support (-ngl):
 *   Windows MapViewOfFile memory cannot be registered with cudaHostRegister.
 *   Instead, when CUDA is available, we allocate cudaMallocHost pinned buffers
 *   and memcpy from mmap into them. Pinned host memory is GPU-accessible via
 *   DMA so the CUDA scheduler can copy tensors to VRAM without illegal access.
 *
 * Include path: this file lives in llama.cpp/src/
 * StreamingWeightMapper lives in streaming-weight-mapper/
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "StreamingWeightMapper.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>  // memcpy

#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#define STREAMING_CUDA 1
#else
#define STREAMING_CUDA 0
#endif

// ── StreamingTensorInfo ───────────────────────────────────────────────────────

struct StreamingTensorInfo {
    std::string     name;
    uint64_t        file_offset  = 0;
    size_t          data_size    = 0;
    uint32_t        gguf_type    = 0;
    ggml_tensor   * tensor       = nullptr;
};

// ── LlamaStreamingContext ─────────────────────────────────────────────────────

class LlamaStreamingContext {
public:
    LlamaStreamingContext(const std::string & gguf_path,
                          size_t cache_size_bytes = 4ULL * 1024 * 1024 * 1024)
        : mapper_(gguf_path, cache_size_bytes) {

        // CPU routing buffer — 1 byte placeholder so the ggml scheduler
        // sees a valid host buffer and doesn't GGML_ABORT on routing.
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        cpu_routing_buf_ = ggml_backend_buft_alloc_buffer(cpu_buft, 1);
        if (cpu_routing_buf_) {
            ggml_backend_buffer_set_usage(cpu_routing_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        auto [biggest_name, biggest_bytes] = mapper_.largest_tensor();
        fprintf(stderr, "[STREAM-DBG] StreamingContext: %zu tensors, cache=%zu GiB, largest='%s' %.1f GiB\n",
            mapper_.tensor_count(),
            cache_size_bytes / (1024ULL*1024*1024),
            biggest_name.c_str(),
            biggest_bytes / (1024.0*1024*1024));
        if (biggest_bytes > cache_size_bytes) {
            fprintf(stderr, "[STREAM-DBG] WARNING: largest tensor (%.1f GiB) exceeds cache (%zu GiB)\n",
                biggest_bytes / (1024.0*1024*1024),
                cache_size_bytes / (1024ULL*1024*1024));
        }

#if STREAMING_CUDA
        fprintf(stderr, "[STREAM-DBG] CUDA mode: tensors will be copied into cudaMallocHost pinned buffers\n");
#endif
    }

    ~LlamaStreamingContext() {
        if (cpu_routing_buf_) {
            ggml_backend_buffer_free(cpu_routing_buf_);
        }
#if STREAMING_CUDA
        // Free all pinned buffers
        std::lock_guard<std::mutex> lock(pinned_mutex_);
        for (auto & kv : pinned_cache_) {
            if (kv.second.ptr) cudaFreeHost(kv.second.ptr);
        }
        pinned_cache_.clear();
#endif
    }

    // ── Registration ─────────────────────────────────────────────────────────

    void register_tensor(const char * name, ggml_tensor * tensor) {
        StreamingTensorInfo info;
        info.name   = name;
        info.tensor = tensor;
        registered_[name] = std::move(info);

        tensor->buffer = nullptr;
        tensor->data   = (void*)(uintptr_t)1; // sentinel
    }

    static void * STREAMING_SENTINEL() { return (void*)(uintptr_t)1; }

    bool is_streaming(const char * name) const {
        return mapper_.has_tensor(name);
    }

    // ── Data access ───────────────────────────────────────────────────────────

    void * get_tensor_data(const char * name) {
#if STREAMING_CUDA
        // On CUDA builds: return a cudaMallocHost pinned buffer.
        // Windows MapViewOfFile memory cannot be used with cudaHostRegister,
        // so we copy mmap data into pinned host memory that the GPU can DMA from.
        {
            std::lock_guard<std::mutex> lock(pinned_mutex_);
            auto it = pinned_cache_.find(name);
            if (it != pinned_cache_.end()) {
                return it->second.ptr;  // already pinned and ready
            }
        }

        // Get mmap pointer (blocks until pages are available)
        void * mmap_ptr = mapper_.get_tensor_ptr(name);
        if (!mmap_ptr) return nullptr;

        // Get tensor size directly from mapper descriptor (always correct)
        size_t sz = mapper_.tensor_size(name);
        if (sz == 0) {
            fprintf(stderr, "[STREAM-DBG] tensor_size=0 for '%s' — using mmap ptr directly\n", name);
            return mmap_ptr;
        }

        // Allocate pinned host memory and copy from mmap
        void * pinned = nullptr;
        cudaError_t err = cudaMallocHost(&pinned, sz);
        if (err != cudaSuccess) {
            fprintf(stderr, "[STREAM-DBG] cudaMallocHost failed for '%s' (%zu MiB): %s — falling back to mmap ptr\n",
                name, sz >> 20, cudaGetErrorString(err));
            cudaGetLastError();
            return mmap_ptr;  // fallback: may still crash if -ngl is set
        }

        memcpy(pinned, mmap_ptr, sz);

        {
            std::lock_guard<std::mutex> lock(pinned_mutex_);
            pinned_cache_[name] = {pinned, sz};
        }

        fprintf(stderr, "[STREAM-DBG] pinned '%s' (%.1f MiB)\n", name, sz / (1024.0*1024));
        return pinned;
#else
        return mapper_.get_tensor_ptr(name);
#endif
    }

    void evict_tensor(const char * name) {
#if STREAMING_CUDA
        std::lock_guard<std::mutex> lock(pinned_mutex_);
        auto it = pinned_cache_.find(name);
        if (it != pinned_cache_.end()) {
            cudaFreeHost(it->second.ptr);
            pinned_cache_.erase(it);
        }
#endif
        mapper_.evict(name);
    }

    void prefetch(const char * name) {
        mapper_.prefetch(name);
    }

    ggml_backend_buffer_t cpu_buffer() const { return cpu_routing_buf_; }

    size_t cache_bytes_used() const { return mapper_.cache_bytes_used(); }
    size_t registered_count() const { return registered_.size(); }

    std::vector<std::string> registered_names() const {
        std::vector<std::string> names;
        names.reserve(registered_.size());
        for (const auto & kv : registered_) names.push_back(kv.first);
        return names;
    }

private:
    StreamingWeightMapper mapper_;
    ggml_backend_buffer_t cpu_routing_buf_ = nullptr;
    std::unordered_map<std::string, StreamingTensorInfo> registered_;

#if STREAMING_CUDA
    // Pinned host buffers — GPU-accessible copies of mmap tensor data.
    // cudaMallocHost allocates page-locked memory that the GPU can DMA from,
    // unlike MapViewOfFile memory which cannot be registered with CUDA.
    struct PinnedBuffer { void * ptr; size_t size; };
    std::unordered_map<std::string, PinnedBuffer> pinned_cache_;
    std::mutex pinned_mutex_;
#endif
};
