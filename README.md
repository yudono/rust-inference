# GGUF Rust Inference Engine

Pure Rust GGUF inference engine — **zero C/C++ ML library dependencies** (no llama.cpp, no GGML bindings).

Load and run GGUF models (Qwen2, Qwen2.5, LLaMA, Mistral) using **on-the-fly dequantization**: weights stay quantized in memory and are dequantized per block during mat-vec. Optional **GPU compute backend** via wgpu + WGSL shaders.

## Quick Start

### Build

```bash
cargo build --release
```

### Run

```bash
./target/release/gguf-infer --model model.gguf --prompt "Hello, how are you"
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | (required) | Path to GGUF model file |
| `--prompt` | `-p` | (required) | Input text prompt |
| `--max-tokens` | `-n` | 512 | Maximum tokens to generate |
| `--temperature` | `-t` | 0.8 | Sampling temperature (0 = greedy) |
| `--top-k` | `-k` | 40 | Top-K sampling |
| `--top-p` | | 0.95 | Nucleus sampling threshold |
| `--seed` | `-s` | 12345 | Random seed for reproducibility |
| `--max-seq-len` | | (from model) | Override max sequence length |
| `--system` | | (none) | System prompt for instruct models |
| `--cpu` | | auto | Force CPU backend |
| `--gpu` | | auto | Force GPU backend |

---

## Model Compatibility

| Model | Architecture | rope_base | Vocab Size | Status |
|-------|-------------|-----------|------------|--------|
| Llama 2 (7B/13B/70B) | llama | 10000.0 | 32000 | Supported |
| Llama 3 (8B/70B) | llama | 500000.0 | 128256 | Supported |
| CodeLlama (7B/13B/34B) | llama | 10000.0 | 32000 | Supported |
| Mistral (7B) | mistral | 10000.0 | 32000 | Supported |
| Qwen2 (0.5B/1.5B/7B/72B) | qwen2 | 1000000.0 | 151936 | Supported |
| Qwen2.5 (0.5B/1.5B/3B/7B/14B/32B/72B) | qwen2 | 1000000.0 | 151936 | Supported |

> Both Qwen2 and Qwen2.5 share `architecture = "qwen2"` in GGUF (llama.cpp convention).

---

## On-the-Fly Dequantization

Unlike traditional engines that expand all weights to f32 on load (consuming ~4x the GGUF file size), this engine stores weights **as raw quantized bytes** (`Vec<u8>`) and dequantizes them per block during mat-vec multiplication.

### Supported Quantization Types

| Type | Block Size | Bytes/Block | Storage Format |
|------|-----------|-------------|----------------|
| F32 | 1 | 4 | Raw f32 bytes |
| F16 | 1 | 2 | IEEE half-precision |
| Q8_0 | 32 | 34 | f16 scale + int8[32] |
| Q4_0 | 32 | 18 | f16 scale + uint4[32] |
| Q5_0 | 32 | 22 | f16 scale + uint4[32] + uint32 high-bits |
| Q4_K | 256 | 144 | f16 d + f16 dmin + uint8 scales[8] + uint4[256] |
| Q5_K | 256 | 176 | f16 d + f16 dmin + uint8 scales[8] + uint5[256] |
| Q6_K | 256 | 210 | uint8 scales[8] + f16 d + uint6[256] |

### Memory Impact

| Model | GGUF Size | Naive f32 Load | On-the-Fly Dequant |
|-------|-----------|----------------|-------------------|
| Qwen2-0.5B Q4_K_M | 0.4 GB | 1.6 GB | **~0.4 GB** |
| Qwen2-1.5B Q4_K_M | 1.0 GB | 4.0 GB | **~1.0 GB** |
| Qwen2.5-3B Q4_K_M | 2.0 GB | 8.0 GB | **~2.0 GB** |
| Qwen2-7B Q4_K_M | 4.4 GB | 16 GB | **~4.4 GB** |

---

## GPU Compute Backend

Optional GPU acceleration via **wgpu** (Vulkan/Metal/DX12) with hand-written WGSL compute shaders.

### Architecture

```
QuantizedMatrix::mat_vec_mul()
        |
        v
   GpuContext::mat_vec_mul()        fallback_cpu()
        |                                |
   wgpu dispatch                   dequant per block
   (1 workgroup per row,           (scalar, no SIMD)
    256 threads per workgroup,
    shared-memory reduction)
        |
   readback to staging buffer
```

### Supported Quant Types (GPU)

| Type | Shader |
|------|--------|
| F32 | Direct f32 bitcast |
| F16 | Manual f16 → f32 conversion |
| Q8_0 | f16 scale + int8 |
| Q4_0 | f16 scale + nibble |
| Q5_0 | f16 scale + nibble + high-bit |
| Q4_K | Sub-block scales + nibbles + d/dmin |
| Q5_K | Sub-block scales + nibble + high-bit |
| Q6_K | Sub-block scales + 6-bit values |

All shaders use **one workgroup per row, 256 threads per workgroup** with shared-memory reduction for output accumulation. Auto GPU/CPU detection; force with `--cpu` or `--gpu`. When GPU is unavailable, falls back to CPU.

> **Note**: Q6_K currently falls back to CPU due to a Metal/Naga GPU execution bug. Q4_K on GPU is verified correct (max_diff ≈ 1e-5 vs CPU).

---

## Accuracy

| Prompt | Logit Correlation | Top-5 Match | Model |
|--------|------------------|-------------|-------|
| "The" | **0.9956** | ✓ | Qwen2-0.5B Q4_K_M |
| Greedy output | Identical | — | Qwen2-0.5B Q4_K_M |

Logits verified against `llama_cpp` reference. Minor ~0.5% differences from FP precision in dequantization.

---

## Performance

| Model | Backend | Speed | Memory |
|-------|---------|-------|--------|
| Qwen2-0.5B Q4_K_M | CPU | ~25 ms/tok | ~400 MB |
| Qwen2-1.5B Q4_K_M | CPU | ~50 ms/tok | ~1 GB |
| Qwen2.5-3B Q4_K_M | CPU | ~5000 ms/tok | ~2 GB |

GPU performance pending optimization (current per-call staging buffers bottleneck).

---

## Chat Template (Instruct Models)

GGUF metadata typically includes a `tokenizer.chat_template` string (Jinja2 format). When present, the engine **automatically wraps prompts** in the correct chat format for that model:

- **ChatML (Qwen2/Qwen2.5)**: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
- **Llama 3**: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>`
- **Llama 2**: `[INST] ... [/INST]`

Without this formatting, instruct models produce irrelevant output (the raw prompt doesn't match their training format).

To add a system prompt:

```bash
./target/release/gguf-infer --model model.gguf --prompt "Hello" --system "Always respond in English."
```

---

## Project Structure

```
src/
+-- gguf.rs          # GGUF file format parser
+-- model.rs         # Model loading (raw bytes, no f32 expansion)
|   +-- raw_tensors HashMap<Vec<u8>>
|   +-- load_mat! macro for quantized weights
|   +-- f32 tensors only for small nroms/biases
|
+-- quant.rs         # QuantizedMatrix with on-the-fly dequant
|   +-- dequantize_row() for embedding lookup
|   +-- mat_vec_mul() for all weight projections
|   +-- dequantize() for full expansion (fallback only)
|
+-- gpu_backend.rs   # GPU compute (wgpu + WGSL)
|   +-- GpuContext: device, queue, cached pipelines
|   +-- WGSL shaders for all 8 quant types
|   +-- mat_vec_mul dispatch with shared-memory reduction
|   +-- Q6_K falls back to CPU (GPU execution bug)
|
+-- tokenizer.rs     # BPE tokenizer
+-- chat_template.rs # Chat template engine (ChatML, Llama3, Llama2)
+-- rope.rs          # Rotary Position Embedding
+-- rmsnorm.rs       # RMS Normalization
+-- attention.rs     # Multi-Head Self-Attention (GQA)
+-- mlp.rs           # SwiGLU Feed-Forward Network
+-- kv_cache.rs      # Key-Value Cache
+-- sampler.rs       # Token sampling
+-- math.rs          # Vector/matrix ops, softmax, SiLU, RMSNorm
+-- transformer.rs   # Transformer stack (pre-norm, residual)
+-- main.rs          # CLI interface (--cpu/--gpu flags)
```

---

## Dependencies

| Crate | Purpose | Optional? |
|-------|---------|-----------|
| `std` | I/O, collections, math | No (core) |
| `wgpu` | GPU compute (Vulkan/Metal/DX12) | **No** (compile-time) |
| `pollster` | Block on async wgpu init | No |
| `bytemuck` | Safe byte reinterpret casts | No |

> At runtime, GPU backend falls back to CPU if no adapter is found. All GPU crates are linked at compile time but only initialize on demand.

---

## Limitations

- No SIMD intrinsics for CPU path (compiler auto-vectorization only)
- GPU backend creates per-call staging buffers (optimization deferred)
- Q6_K GPU WGSL dequant has a Metal/Naga execution bug — falls back to CPU
- KV cache is f32 (not quantized), grows with sequence length
- Sliding window attention not implemented (Mistral)
- Qwen2 does not use BOS token (despite GGUF metadata)
- `output.weight` is optional; when missing, `token_embd.weight` is used as tied lm_head
- Chat template detection is heuristic (known formats only); full Jinja2 parser not implemented

## License

Public domain. Use however you like.
