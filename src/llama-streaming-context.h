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
 * Integration points:
 *   - llama_model_loader::load_all_data()  → skip allocation, call register_tensor()
 *   - llama-graph.cpp (before tensor ops)  → call get_tensor_data(), set ggml_tensor::data
 *
 * Include path: this file lives in llama.cpp/src/
 * StreamingWeightMapper lives in streaming-weight-mapper/ (adjust include below)
 */

#include "ggml.h"           // ggml_tensor, ggml_type
#include "ggml-backend.h"   // ggml_backend_buffer_t, ggml_backend_buft_alloc_buffer
#include "llama.h"          // llama types

// StreamingWeightMapper.h is resolved via target_include_directories in src/CMakeLists.txt
// pointing to D:/home/dev/code/streaming-weight-mapper/
#include "StreamingWeightMapper.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

// CUDA host registration — makes mmap pages GPU-accessible via DMA.
// cudaHostRegister pins the virtual address range so the GPU can read it
// directly without a separate cudaMemcpy. This is what makes -ngl work
// with streaming tensors: the CUDA kernel can DMA from host mmap memory.
#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#define STREAMING_CUDA_REGISTER 1
#else
#define STREAMING_CUDA_REGISTER 0
#endif

// ── StreamingTensorInfo ───────────────────────────────────────────────────────

/**
 * Metadata for a single tensor managed by the streaming context.
 * Mirrors what llama_tensor_weight stores, but without a live data pointer —
 * the data is loaded from disk on demand.
 */
struct StreamingTensorInfo {
    std::string     name;
    uint64_t        file_offset  = 0;     // byte offset in the GGUF file
    size_t          data_size    = 0;     // total bytes for this tensor
    uint32_t        gguf_type    = 0;     // original GGUF quantization type
    ggml_tensor   * tensor       = nullptr; // back-pointer to the live ggml_tensor
};

// ── LlamaStreamingContext ─────────────────────────────────────────────────────

/**
 * Owns a StreamingWeightMapper and acts as the glue layer between
 * llama.cpp's model loader and the streaming IO engine.
 *
 * Thread safety: get_tensor_data() and evict_tensor() are thread-safe.
 * register_tensor() must be called from a single thread during model load.
 */
class LlamaStreamingContext {
public:
    /**
     * @param gguf_path         Path to the .gguf model file
     * @param cache_size_bytes  Max bytes to keep resident (default: 4 GiB)
     */
    LlamaStreamingContext(const std::string & gguf_path,
                          size_t cache_size_bytes = 4ULL * 1024 * 1024 * 1024)
        : mapper_(gguf_path, cache_size_bytes) {
        // Create a CPU buffer that streaming tensors will be assigned to.
        // This gives the ggml backend scheduler a valid buffer type for routing
        // operations to CPU without allocating any real memory.
        // We use the default CPU buffer type and allocate 1 byte as a placeholder.
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        cpu_routing_buf_ = ggml_backend_buft_alloc_buffer(cpu_buft, 1);
        // Mark as weights buffer so the CUDA scheduler copies it to GPU correctly
        // instead of treating it as an activation/compute buffer.
        if (cpu_routing_buf_) {
            ggml_backend_buffer_set_usage(cpu_routing_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        // Diagnostic: show tensor count, cache size, and largest single tensor
        {
            auto [biggest_name, biggest_bytes] = mapper_.largest_tensor();
            fprintf(stderr, "[STREAM-DBG] StreamingContext: %zu tensors, cache=%zu GiB, largest='%s' %.1f GiB\n",
                mapper_.tensor_count(),
                cache_size_bytes / (1024ULL*1024*1024),
                biggest_name.c_str(),
                biggest_bytes / (1024.0*1024*1024));
            if (biggest_bytes > cache_size_bytes) {
                fprintf(stderr, "[STREAM-DBG] WARNING: largest tensor (%.1f GiB) exceeds cache size (%zu GiB) — will OOM!\n",
                    biggest_bytes / (1024.0*1024*1024),
                    cache_size_bytes / (1024ULL*1024*1024));
            }
        }
    }

    ~LlamaStreamingContext() {
        if (cpu_routing_buf_) {
            ggml_backend_buffer_free(cpu_routing_buf_);
        }
    }

    // ── Registration (called during load_all_data) ────────────────────────────

    /**
     * Mark a tensor as streaming — its data will NOT be allocated upfront.
     * Called once per streaming tensor during model load.
     *
     * @param name    Tensor name (e.g. "blk.0.attn_q.weight")
     * @param tensor  Pointer to the already-created ggml_tensor (shape/type set, data=null)
     */
    void register_tensor(const char * name, ggml_tensor * tensor) {
        StreamingTensorInfo info;
        info.name   = name;
        info.tensor = tensor;

        // Pull descriptor from mapper (already parsed GGUF header)
        // We don't duplicate the offset/size here — mapper owns them.
        // We just track which tensors are ours.
        registered_[name] = std::move(info);

        // Do NOT set tensor->buffer — if buffer is set, ggml_backend_sched will
        // GGML_ABORT if it can't route an op to that specific buffer's backend.
        // Instead, leave buffer=nullptr and set data=sentinel so:
        //   1. ggml allocator sees data!=NULL and skips allocation (line 592 ggml-alloc.c)
        //   2. ggml scheduler sees buffer==NULL and routes ops normally to CPU
        //   3. Our fill code replaces data with the real pointer before compute
        tensor->buffer = nullptr;
        tensor->data   = (void*)(uintptr_t)1; // sentinel: real data filled at inference
    }

    // Sentinel value: marks a tensor as streaming-managed, not yet loaded.
    // Must be non-null but never dereferenced before fill_streaming_tensors() runs.
    static void * STREAMING_SENTINEL() { return (void*)(uintptr_t)1; }

    /**
     * Returns true if this tensor is managed by the streaming context.
     */
    bool is_streaming(const char * name) const {
        // Check mapper's descriptor map — O(1) lookup, populated at construction
        // from the GGUF header. registered_ can't be used here because it's only
        // populated AFTER register_tensor() is called (bootstrap problem).
        return mapper_.has_tensor(name);
    }

    // ── Data access (called at inference time) ────────────────────────────────

    /**
     * Returns a raw pointer to the tensor's data, blocking until available.
     *
     * The returned pointer is valid until evict_tensor() or the next call
     * that evicts this tensor from the LRU cache.
     *
     * @param name  Tensor name
     * @return      Raw pointer to tensor bytes (host memory, cache-pinned)
     */
    void * get_tensor_data(const char * name) {
        void * ptr = mapper_.get_tensor_ptr(name);
#if STREAMING_CUDA_REGISTER
        // Register the mmap pages with CUDA so the GPU can DMA from them.
        // Without this, any -ngl offloaded layer that reads this tensor
        // crashes with "illegal memory access" in the CUDA kernel.
        // cudaHostRegister is idempotent-safe: we track registered ptrs
        // to avoid double-registration.
        if (ptr) {
            std::lock_guard<std::mutex> lock(cuda_reg_mutex_);
            auto it = registered_tensors_.find(name);
            if (it == registered_tensors_.end()) {
                auto dit = registered_.find(name);
                size_t sz = (dit != registered_.end()) ? dit->second.data_size : 0;
                if (sz > 0) {
                    cudaError_t err = cudaHostRegister(ptr, sz,
                        cudaHostRegisterPortable | cudaHostRegisterReadOnly);
                    if (err == cudaSuccess) {
                        registered_tensors_[name] = {ptr, sz};
                    } else {
                        // Non-fatal: CUDA will fall back to a cudaMemcpy path
                        fprintf(stderr, "[STREAM-DBG] cudaHostRegister failed for '%s': %s\n",
                            name, cudaGetErrorString(err));
                        cudaGetLastError(); // clear error
                    }
                }
            }
        }
#endif
        return ptr;
    }

    /**
     * Evicts a tensor from the LRU tracking and unregisters CUDA pinning.
     */
    void evict_tensor(const char * name) {
#if STREAMING_CUDA_REGISTER
        {
            std::lock_guard<std::mutex> lock(cuda_reg_mutex_);
            auto it = registered_tensors_.find(name);
            if (it != registered_tensors_.end()) {
                cudaHostUnregister(it->second.ptr);
                registered_tensors_.erase(it);
            }
        }
#endif
        mapper_.evict(name);
    }

    /**
     * Async prefetch — enqueues a tensor for background IO.
     * Call this while the current layer is computing to overlap IO with compute.
     */
    void prefetch(const char * name) {
        mapper_.prefetch(name);
    }

    // ── Stats ─────────────────────────────────────────────────────────────────

    /** Total bytes currently resident in the streaming cache */
    /** Returns the CPU host buffer used for routing streaming tensors */
    ggml_backend_buffer_t cpu_buffer() const { return cpu_routing_buf_; }

    size_t cache_bytes_used() const {
        return mapper_.cache_bytes_used();
    }

    /** Number of registered streaming tensors */
    size_t registered_count() const {
        return registered_.size();
    }

    /** List all registered streaming tensor names */
    std::vector<std::string> registered_names() const {
        std::vector<std::string> names;
        names.reserve(registered_.size());
        for (const auto & kv : registered_) names.push_back(kv.first);
        return names;
    }

private:
    StreamingWeightMapper mapper_;

    // Shared CPU buffer used purely for backend routing — no real data stored here.
    // All streaming tensors are assigned this buffer so the scheduler routes
    // their ops to CPU. Real data is filled at inference via get_tensor_data().
    ggml_backend_buffer_t cpu_routing_buf_ = nullptr;

    // Tensors registered as streaming (name -> metadata)
    std::unordered_map<std::string, StreamingTensorInfo> registered_;

    // Data pointers are now owned by StreamingWeightMapper's mmap—no local buffers needed.

#if STREAMING_CUDA_REGISTER
    // Tracks which tensor mmap regions are currently registered with CUDA.
    // cudaHostRegister must be paired with cudaHostUnregister on eviction.
    struct CudaRegisteredPtr { void * ptr; size_t size; };
    std::unordered_map<std::string, CudaRegisteredPtr> registered_tensors_;
    std::mutex cuda_reg_mutex_;
#endif
};
