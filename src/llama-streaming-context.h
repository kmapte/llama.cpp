#pragma once

/**
 * llama-streaming-context.h
 *
 * Bridges StreamingWeightMapper into llama.cpp's model loader.
 *
 * CUDA support (-ngl):
 *   Windows MapViewOfFile memory cannot be used directly by CUDA kernels.
 *   When CUDA is available, tensors destined for GPU ops are served from
 *   ggml_backend_cuda_host_buffer_type() pinned buffers (cudaMallocHost
 *   internally). These are GPU-accessible via DMA. We allocate them lazily
 *   per tensor, with an LRU cap so we don't exhaust pinned memory.
 *   Tensors that don't fit in the pinned budget fall back to mmap (CPU only).
 */

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "llama.h"
#include "StreamingWeightMapper.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <list>
#include <mutex>

// ── StreamingTensorInfo ───────────────────────────────────────────────────────

struct StreamingTensorInfo {
    std::string   name;
    ggml_tensor * tensor    = nullptr;
};

// ── LlamaStreamingContext ─────────────────────────────────────────────────────

class LlamaStreamingContext {
public:
    // pinned_budget_bytes: max bytes of cudaMallocHost pinned memory to use.
    // Tensors beyond this budget fall back to mmap (no -ngl benefit for those).
    // Default 2 GiB — safe for a 6GB GPU with other allocations present.
    LlamaStreamingContext(const std::string & gguf_path,
                          size_t cache_size_bytes    = 4ULL  * 1024 * 1024 * 1024,
                          size_t pinned_budget_bytes = 2ULL  * 1024 * 1024 * 1024)
        : mapper_(gguf_path, cache_size_bytes)
        , pinned_budget_(pinned_budget_bytes) {

        // 1-byte CPU routing buffer — gives ggml scheduler a valid host buffer
        // so it doesn't GGML_ABORT when checking buffer usage on our tensors.
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        cpu_routing_buf_ = ggml_backend_buft_alloc_buffer(cpu_buft, 1);
        if (cpu_routing_buf_) {
            ggml_backend_buffer_set_usage(cpu_routing_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        // Check if CUDA host buffer type is available
#if defined(GGML_USE_CUDA)
        cuda_host_buft_ = ggml_backend_cuda_host_buffer_type();
        fprintf(stderr, "[STREAM-DBG] CUDA host buffer available — pinned budget: %zu MiB\n",
            pinned_budget_ >> 20);
#endif

        auto [biggest_name, biggest_bytes] = mapper_.largest_tensor();
        fprintf(stderr, "[STREAM-DBG] StreamingContext: %zu tensors, cache=%zu GiB, largest='%s' %.1f GiB\n",
            mapper_.tensor_count(),
            cache_size_bytes / (1024ULL*1024*1024),
            biggest_name.c_str(),
            biggest_bytes / (1024.0*1024*1024));
    }

    ~LlamaStreamingContext() {
        if (cpu_routing_buf_) ggml_backend_buffer_free(cpu_routing_buf_);
        // Free all pinned buffers
        std::lock_guard<std::mutex> lk(pinned_mutex_);
        for (auto & kv : pinned_map_) {
            if (kv.second.buf) ggml_backend_buffer_free(kv.second.buf);
        }
    }

    // ── Registration ─────────────────────────────────────────────────────────

    void register_tensor(const char * name, ggml_tensor * tensor) {
        registered_[name] = { name, tensor };
        tensor->buffer = nullptr;
        tensor->data   = (void*)(uintptr_t)1; // sentinel
    }

    static void * STREAMING_SENTINEL() { return (void*)(uintptr_t)1; }

    bool is_streaming(const char * name) const {
        return mapper_.has_tensor(name);
    }

    // ── Data access ───────────────────────────────────────────────────────────

    // Returns a pointer suitable for the current backend:
    //   - If CUDA is available and tensor fits in pinned budget: returns ptr
    //     into a cudaMallocHost buffer (GPU-accessible, DMA-able).
    //   - Otherwise: returns raw mmap ptr (CPU only, -ngl will fail for this tensor).
    void * get_tensor_data(const char * name) {
#if defined(GGML_USE_CUDA)
        if (cuda_host_buft_) {
            // Check pinned cache first
            {
                std::lock_guard<std::mutex> lk(pinned_mutex_);
                auto it = pinned_map_.find(name);
                if (it != pinned_map_.end()) {
                    return it->second.ptr;
                }
            }

            // Get mmap pointer and tensor size
            void * mmap_ptr = mapper_.get_tensor_ptr(name);
            if (!mmap_ptr) return nullptr;

            size_t sz = mapper_.tensor_size(name);
            if (sz == 0) return mmap_ptr;

            // Check if we have budget
            std::lock_guard<std::mutex> lk(pinned_mutex_);
            if (pinned_used_ + sz <= pinned_budget_) {
                // Allocate pinned buffer via CUDA host buffer type
                ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(cuda_host_buft_, sz);
                if (buf) {
                    void * ptr = ggml_backend_buffer_get_base(buf);
                    // Mark as weights so CUDA scheduler copies to VRAM
                    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    // Copy from mmap into pinned memory
                    memcpy(ptr, mmap_ptr, sz);
                    pinned_map_[name] = { buf, ptr, sz };
                    pinned_used_ += sz;
                    return ptr;
                }
                // Allocation failed — fall through to mmap
                fprintf(stderr, "[STREAM-DBG] pinned alloc failed for '%s' (%.1f MiB) — using mmap\n",
                    name, sz / (1024.0*1024));
            }
            // Budget exhausted for this tensor — use mmap (CPU path only)
            return mmap_ptr;
        }
#endif
        return mapper_.get_tensor_ptr(name);
    }

    // Returns the appropriate buffer for a tensor pointer returned by get_tensor_data.
    // If the pointer is in our pinned cache, returns the pinned buffer.
    // Otherwise returns the plain CPU routing buffer.
    ggml_backend_buffer_t get_tensor_buffer(const char * name) {
#if defined(GGML_USE_CUDA)
        std::lock_guard<std::mutex> lk(pinned_mutex_);
        auto it = pinned_map_.find(name);
        if (it != pinned_map_.end()) {
            return it->second.buf;
        }
#endif
        return cpu_routing_buf_;
    }

    void evict_tensor(const char * name) {
        mapper_.evict(name);
    }

    // Call after each token's compute completes — evicts views over budget.
    // Never call during graph compute: live pointers would become invalid.
    void evict_over_budget() {
        mapper_.evict_over_budget();
    }

    void prefetch(const char * name) { mapper_.prefetch(name); }

    ggml_backend_buffer_t cpu_buffer()         const { return cpu_routing_buf_; }
    size_t cache_bytes_used()                  const { return mapper_.cache_bytes_used(); }
    size_t registered_count()                  const { return registered_.size(); }
    size_t total_tensor_count()                const { return mapper_.tensor_count(); }
    size_t pinned_bytes_used()                 const { return pinned_used_; }

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

    // Pinned buffer cache — cudaMallocHost memory, GPU-accessible
    struct PinnedEntry {
        ggml_backend_buffer_t buf  = nullptr;
        void *                ptr  = nullptr;
        size_t                size = 0;
    };
    std::unordered_map<std::string, PinnedEntry> pinned_map_;
    std::mutex  pinned_mutex_;
    size_t      pinned_used_   = 0;
    size_t      pinned_budget_ = 0;

#if defined(GGML_USE_CUDA)
    ggml_backend_buffer_type_t cuda_host_buft_ = nullptr;
#endif
};
