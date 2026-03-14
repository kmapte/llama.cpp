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
        // Return mmap pointer directly. The ggml CUDA scheduler will issue a
        // cudaMemcpy from host to device automatically when it sees a CPU-buffered
        // weight tensor being used in a GPU op (triggered by -ngl).
        // cudaMallocHost / cudaHostRegister on MapViewOfFile memory is not
        // supported on Windows and causes OOM or illegal access errors.
        return mapper_.get_tensor_ptr(name);
    }

    void evict_tensor(const char * name) {
        mapper_.evict(name);
    }

    void prefetch(const char * name) {
        mapper_.prefetch(name);
    }

    ggml_backend_buffer_t cpu_buffer() const { return cpu_routing_buf_; }

    size_t cache_bytes_used()   const { return mapper_.cache_bytes_used(); }
    size_t registered_count()   const { return registered_.size(); }
    size_t total_tensor_count() const { return mapper_.tensor_count(); }

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

    // Note: cudaMallocHost / cudaHostRegister on Windows MapViewOfFile memory
    // is not supported. CUDA scheduler handles host->device copies automatically
    // via cudaMemcpy when it processes ops involving CPU-buffered weight tensors.
};
