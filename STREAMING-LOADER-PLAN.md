# llama.cpp Streaming Weight Loader — Project Brief
*Prepared for handoff. Start here tomorrow.*

---

## The Problem

`deepseek-v3` is 404GB (671B params, Q4_K_M GGUF). llama.cpp refuses to load it
because `llama_model_loader` requires **every byte mapped into addressable memory**
before inference can begin. Even `use_mmap=true` just maps the file into the process
address space — it still counts toward resident memory. With 6GB VRAM and typical
32-64GB RAM, the loader bails with "cannot meet free memory target".

**The existing overflow/tensor_split mechanism only moves whole layers between
GPU/CPU. It does NOT stream from disk. That is the gap we are filling.**

---

## The Goal

Build a **streaming weight loader** that:
1. Reads GGUF headers and builds a tensor descriptor table (name, shape, file offset)
   but **never allocates the full model data upfront**
2. Streams weight tensors from disk into a small VRAM window on demand, per layer
3. Evicts tensors from VRAM after they are consumed
4. (Phase 2) Splits compute across CPU + GPU + NPU simultaneously

This lets deepseek-v3 run on a 6GB GPU + normal RAM machine by treating VRAM
as a sliding window over 404GB of weights.

---

## What We Already Have

- `StreamingScheduler.h` — existing streaming weight mapper (location: `D:\home\dev\code\streaming-weight-mapper\`)
- Architecture analysis from GitHub Copilot covering all key llama.cpp files
- Clear integration points identified (see below)
- Known bug: **V tensor bug in `StreamingScheduler.h`** — fix this FIRST before integrating

---

## Key Files to Touch in llama.cpp

| File | What changes |
|------|-------------|
| `src/llama-model-loader.cpp` | Entry point — replace full allocation with lazy descriptor table |
| `src/llama-model-loader.h` | Add streaming context struct, IO queue, LRU cache |
| `src/llama-model.h` | Add `llama_streaming_context` alongside `llama_model` |
| `src/llama-context.cpp` | Hook streaming prefetch into context init |
| `ggml/src/ggml-backend.cpp` | Register new `GGML_BACKEND_STREAM` backend type |
| `ggml/include/ggml-backend.h` | Declare `GGML_BACKEND_DEVICE_TYPE_NPU` (Phase 2) |
| `src/llama-graph.cpp` | Insert `ggml_stream_load` nodes into graph build |
| `CMakeLists.txt` | Add streaming backend as build option `-DLLAMA_STREAMING=ON` |

---

## Architecture of the New Flow

```
CURRENT FLOW (broken for 400GB):
  llama_model_load_from_file()
    → allocate ALL tensors in RAM/VRAM  ← FAILS at 404GB
    → begin inference

NEW STREAMING FLOW:
  llama_model_load_from_file()
    → parse GGUF header only
    → build descriptor table: { name, shape, dtype, file_offset }
    → allocate small VRAM staging buffer (e.g. 4GB)
    → return — NO weight data loaded yet

  llama_decode() — per layer:
    → graph builder inserts ggml_stream_load node for each tensor
    → StreamingScheduler checks LRU cache
      HIT  → tensor already in staging buffer → use it
      MISS → async read from GGUF file at stored offset → load into staging
    → execute layer ops on GPU
    → evict tensor from staging buffer (LRU policy)
    → prefetch NEXT layer's tensors in background IO thread
```

---

## Implementation Phases

### Phase 1 — Core Streaming Loader (do this first)
1. Fix V tensor bug in `StreamingScheduler.h`
2. Fork llama.cpp: `git checkout -b feature/streaming-loader`
3. Modify `llama-model-loader.cpp`:
   - Skip tensor data allocation in `load_all_data()`
   - Build `TensorDescriptor` table instead
   - Store file descriptor + offset per tensor
4. Create `llama-streaming-context.h` / `.cpp`:
   - LRU cache (size = configurable, default 4GB)
   - IO queue with async reads (`io_uring` on Linux, `ReadFileEx` on Windows)
   - Prefetch thread that looks ahead N layers
5. Modify `llama-graph.cpp`:
   - Before each tensor op, check if tensor data is resident
   - If not, block until StreamingScheduler delivers it
6. Test with deepseek-v3 — verify it loads and runs (slowly but correctly)

### Phase 2 — NPU Backend
1. Register `GGML_BACKEND_DEVICE_TYPE_NPU` in `ggml-backend.h`
2. Implement NPU buffer type + basic kernels (matmul, attention)
3. Extend `llama_params_fit()` to query NPU memory alongside GPU/CPU
4. Stream weight tiles directly into NPU local memory (tiny — a few MB)
5. Vendor library integration (QNN for Qualcomm, OpenVINO for Intel, etc.)

### Phase 3 — Expose via Ollama
- Ollama vendors its own llama.cpp build
- Replace Ollama's bundled `llama.cpp` binary with our fork
- OR: set `OLLAMA_CUSTOM_CPU_LIBRARY` env var to point at our build
- No Ollama source changes needed — model format stays GGUF

---

## Critical Design Decisions

**LRU cache size:** Default to `min(available_VRAM * 0.8, 4GB)`. Configurable
via `--streaming-cache-size` CLI flag.

**IO strategy:** 
- Windows: `ReadFileEx` with overlapped IO (already on your machine)
- Linux: `io_uring` for zero-copy async reads
- Fallback: blocking `pread()` on a background thread pool

**Tensor granularity:** Stream whole tensors first (simpler). Later add
partial tensor loads (row slices) for very large weight matrices.

**GGUF compatibility:** No format changes needed for Phase 1. Tensor offsets
are already in the GGUF header — we just stop after reading them instead of
loading the data.

**Graph scheduler assumption:** Currently `ggml_backend_sched` assumes all
tensors are resident before compute. We need to add a
`ggml_tensor_is_virtual()` API and teach the scheduler to wait on IO.

---

## The V Tensor Bug (fix first)

Location: `D:\home\dev\code\streaming-weight-mapper\StreamingScheduler.h`

This is described as a "cosmetic one-liner" — likely an off-by-one or wrong
index when accessing the V (value) tensor in attention. Fix and commit before
starting the llama.cpp fork so you have a clean baseline.

---

## Local LLM Stack for This Work

| Task | Model |
|------|-------|
| C++ code generation | `qwen2.5-coder:14b` |
| Reasoning / design decisions | `llama3.1:8b` (Plan A) |
| Deep reasoning | `deepseek-r1:latest` with `/no_think` prefix (Plan B) |
| Code review of diffs | `qwen2.5-coder:14b` |

---

## Tomorrow's Attack Order

1. `cd D:\home\dev\code\streaming-weight-mapper`
   Fix V tensor bug in `StreamingScheduler.h`

2. `cd D:\home\dev\code\llama.cpp`
   `git checkout -b feature/streaming-loader`

3. Read `src/llama-model-loader.cpp` fully — understand `load_all_data()`

4. Start with a minimal proof of concept:
   - Make loader skip data allocation for one tensor type
   - Manually inject StreamingScheduler for that tensor
   - Verify inference still works (just slower)

5. Expand to full model once PoC works

---

*Written from Copilot analysis + session context. All file paths confirmed on your machine.*
