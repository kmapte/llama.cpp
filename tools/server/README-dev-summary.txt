llama-server Development Overview

This document summarizes key components and flows in llama-server to help
developers extend or integrate with the project.

================================================================================
ARCHITECTURE DIAGRAM (llama-server)
================================================================================

    API_User <--> server_http_context
    server_http_context <-- router mode --> server_models
    server_http_context <-- inference mode --> server_routes
    server_routes -- server_task --> server_queue
    subgraph server_context
        server_queue --> server_slot
        server_slot -- server_task_result --> server_response
        server_slot[multiple server_slot]
    end
    server_response --> server_routes

================================================================================
1. Backend Modes
   - Inference: single loaded GGUF model.
   - Router : multiple model instances with request routing.

2. Core Components
   * server_context   - holds main llama_context and active slots.
   * server_slot      - one sequence; handles parallel inference requests.
   * server_routes    - middleware between context and HTTP API.
   * server_http_context - HTTP server using cpp-httplib.
   * server_queue     - thread-safe queue from HTTP workers to context.
   * server_response  - queue for results back to HTTP workers.
   * server_response_reader - helper to manage both queues.
   * server_task      - unit of work enqueued by HTTP layer.
   * server_task_result - unit returned by context.
   * server_tokens    - unified token representation (text/multi-modal).
   * server_prompt_checkpoint - snapshot of KV cache for reuse (RWKV, SWA).
   * server_models    - manages multiple backend instances in router mode.

3. Batching
   * Single batch shared across slots; update_slots() fills from slots.
   * Slots batched only if configurations (e.g., LoRA) are compatible.
   * llama_decode is called when batch is full or all slots processed.
   * After decoding, embeddings or next-token sampling occurs.
   * Slots with pending prompt tokens yield until next iteration.

4. Threading
   * server_context runs on dedicated thread; keep post‑processing light.
   * HTTP threads handle parsing, templating, tokenization, JSON.
   * Avoid raw JSON in server_slot; convert early.

5. Request Trace Example
   - HTTP -> server_routes -> server_task -> server_queue
   - server_context picks a slot, update_slot() processes.
   - Partial/final responses via server_task_result -> server_response.
   - HTTP layer’s server_res_generator listens, converts to JSON.

6. Testing
   * Pytest suite that launches server, sends requests, checks responses.

7. Web UI
   * SvelteKit frontend in tools/server/webui.
   * Supports single-model & router modes, attachments, conversation
     management, etc.
   * Development uses Node/Vite; build artifacts embedded in server.

8. Notable PRs
   * Lists key pull requests adding major features.

This summary allows a quick understanding of server architecture and is
intended for developers extending llama-server or using it as a base.

================================================================================
llama.cpp EXECUTION FLOW DIAGRAM
================================================================================

    1. Load Model (llama.cpp)
        ↓
    llama_model_load_from_file()
        ↓
    llama_model_loader
        ├─ Parse GGUF file headers
        ├─ Load model weights
        └─ Initialize llama_model with hparams, vocab, tensors
    
    2. Create Context
        ↓
    llama_new_context_with_model()
        ↓
    llama_context constructor
        ├─ Allocate KV cache
        ├─ Initialize backend scheduler
        └─ Reserve computation graphs
    
    3. Prepare Batch (Optional)
        ↓
    llama_batch_init()
        └─ Create llama_batch struct with tokens/embeddings
    
    4. Inference Loop
        ↓
    llama_encode() OR llama_decode()
        ├─ Build computation graph
        │   ├─ Get architecture-specific model from models/
        │   ├─ Build tensor operations based on model type
        │   ├─ Assign tensors to backend devices (CPU/GPU)
        │   └─ Execute GGML operations
        ├─ Compute output
        ├─ Update KV cache
        └─ Return logits
    
    5. Generate Next Token
        ↓
    llama_sampler_*
        ├─ Apply sampling logic (Top-K, Top-P, Temperature)
        └─ Sample next token (add to batch)
    
    6. Convert Token to Text
        ↓
    llama_token_to_piece()
        ├─ Vocab lookup
        └─ Output text
    
    7. Cleanup
        ↓
    llama_free & llama_model_free

================================================================================
llama.cpp DATA FLOW PIPELINE
================================================================================

Model Loading Phase:
    GGUF File → llama_model_loader → llama_model struct
                                        ├─ hparams (vocab_size, n_layer)
                                        ├─ llama_vocab (token↔text mapping)
                                        ├─ ggml_context (tensors)
                                        └─ devices[] (GPU/CPU refs)

Context Initialization:
    llama_model → llama_context constructor
                  ├─ llama_memory_t (KV cache allocator)
                  ├─ ggml_backend_sched (Scheduler)
                  └─ ubatch (micro-batch buffer)

Inference Phase:
    Input Tokens → llama_batch struct
                   ├─ token/seq_id/pos/logits
                   ↓
    llama_decode (Process Batch)
                   ├─ Build Graph
                   │  ├─ Select Model Arch from models/
                   │  └─ Build Computation Graph
                   ├─ Backend Execution
                   │  ├─ Scheduler assigns ops to devices
                   │  ├─ GGML kernels (CPU/GPU compute)
                   │  └─ Update KV Cache for next step
                   └─ → logits (float[] shape n_vocab)

Output Phase:
    logits → llama_sampler
             ├─ Apply sampling logic
             ├─ Sample next token
             ↓
    token: int32_t → llama_token_to_piece
                     ├─ Vocab lookup
                     └─ text string (decoded output)

================================================================================
llama_decode() DEEP DIVE - EXECUTION STEPS
================================================================================

Step 1: VALIDATE INPUT
    Input Batch (tokens, seq_ids, positions)
        ↓ Validate batch
    ✓ Valid or X Return error

Step 2: ALLOCATE KV CACHE
    Get memory cells from allocator for attention key-value pairs
        ↓
    X No space → return error 1
    ✓ Cells allocated → continue

Step 3: CHECK GRAPH REBUILD NEEDED
    Compare current batch size with cached graph size
        ↓
    X Yes, rebuild needed → call arch-specific graph builder
    ✓ No, reuse cached graph → continue

Step 4: BUILD COMPUTATION DAG (if needed)
    For each transformer layer:
        ├─ Normalization (rmsnorm, layernorm)
        ├─ Attention layers (matmul, rope, softmax, KV cache)
        ├─ FFN layers (mlp, gelu, etc)
        └─ Output layer (to_logits)

Step 5: OPTIMIZE & ASSIGN TO DEVICES
    ggml_backend_sched assigns operations to:
        ├─ CPU cores (GGML_BACKEND_CPU)
        ├─ NVIDIA GPU (CUDA)
        ├─ Apple GPU (Metal)
        ├─ AMD GPU (HIP)
        ├─ Intel GPU (OneAPI)
        └─ RPC backend (remote execution)

Step 6: EXECUTE GRAPH
    ggml_graph_compute() runs all ops on assigned devices

Step 7: UPDATE KV CACHE
    Store current key/value for next decode step

Step 8: SYNCHRONIZE & RETURN
    Wait for all backends to complete
        ↓
    return 0 success (logits ready for sampling)

================================================================================
KEY DATA STRUCTURES
================================================================================

Structure               Purpose                    Location
─────────────────────────────────────────────────────────────────────
llama_model             Loaded weights, hparams    llama-model.h
llama_context           Runtime state, memory      llama-context.h
llama_batch             Input tokens + metadata    llama.h
llama_vocab             Token ↔ text mapping       llama-vocab.h
llama_memory_t          KV cache allocator         llama-memory.h
llama_kv_cache          Key-value pairs for attn   llama-kv-cache.h
ggml_graph              Computation DAG            GGML library
llama_sampler           Token selection strategy   llama-sampler.h

================================================================================
MEMORY MANAGEMENT: HANDLING MODELS LARGER THAN VRAM
================================================================================

The system handles large models through three mechanisms:

1. LAYER-LEVEL PARTITIONING
   - Each GPU gets N full transformer layers (n_gpu_layers)
   - Layers are computed on the device that holds them
   - Remaining layers run on CPU (slower but fits)

2. TENSOR OVERFLOW BUFFERS (tensor_buft_overrides)
   - When a layer doesn't fit entirely on GPU, split it:
     * Attention: stays on GPU
     * FFN up-proj: goes to CPU
     * FFN down-proj: goes to CPU
   - At runtime, scheduler moves only needed pieces to GPU

3. AUTOMATIC FIT ALGORITHM (llama_params_fit)
   - Query free VRAM on each GPU and host RAM
   - Simulate loading layers back-to-front on each device
   - Use false-position interpolation to maximize GPU utilization
   - Falls back to CPU if needed

Example: 70B model on 8GB GPU + 32GB RAM
   - ~20GB on GPU  (40 layers)
   - ~24GB on CPU  (remaining layers + context)
   - Context shrunk to fit remaining space

Limitation: Model must fit in total available memory
   - No streaming from disk
   - No on-demand weight loading
   - 400GB models require 400GB RAM + VRAM combined