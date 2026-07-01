# GGUF Rust Inference Engine

Pure Rust GGUF inference engine - **zero AI/ML library dependencies**.

Load and run GGUF models (LLaMA, Llama 3, Mistral, Qwen2) using only Rust standard library.

## Architecture

```
+------------------------------------------------------------------+
|                     CLI Interface                                  |
|  cargo run --release -- --model model.gguf --prompt "Hello"       |
+------------------------------+-----------------------------------+
                               |
+------------------------------v-----------------------------------+
|                      Model Loader                                 |
|  - Parse GGUF header (v3)                                        |
|  - Load metadata (arch, dims, vocab)                             |
|  - Dequantize tensor weights to f32                              |
+------------------------------+-----------------------------------+
                               |
+------------------------------v-----------------------------------+
|                   Tokenizer (BPE)                                 |
|  - Load vocab + merges from GGUF metadata                        |
|  - Pre-tokenize -> BPE merge -> Token IDs                        |
+------------------------------+-----------------------------------+
                               |
+------------------------------v-----------------------------------+
|                 Transformer Inference                              |
|                                                                   |
|  [Embedding]    [RoPE]    [RMSNorm]                              |
|      |             |          |                                   |
|  +---v-------------v----------v---+                              |
|  |      Multi-Head Attention      |                              |
|  |  Q=x*Wq  K=x*Wk  V=x*Wv      |                              |
|  |  scores = QK'/sqrt(d) + RoPE  |                              |
|  |  output = softmax(scores)*V*Wo |                              |
|  +---------------+---------------+                               |
|                  | + Residual                                     |
|  +---------------v---------------+                               |
|  |         SwiGLU FFN            |                              |
|  |  gate = SiLU(x*Wg)           |                              |
|  |  hidden = gate * (x*Wu)      |                              |
|  |  output = hidden * Wd        |                              |
|  +---------------+---------------+                               |
|                  | + Residual                                     |
|                  v                                                |
|           [Repeat x N layers]                                     |
|                  |                                                |
|  +---------------v---------------+                               |
|  |  Final RMSNorm -> LM Head     |                              |
|  |  -> Logits                    |                              |
|  +-------------------------------+                              |
+------------------------------+-----------------------------------+
                               |
+------------------------------v-----------------------------------+
|                      Sampler                                      |
|  Temperature -> Top-K -> Top-P -> Sample Token                   |
+------------------------------+-----------------------------------+
                               v
                      Generated Text Output
```

## Quick Start

### Build

```bash
cargo build --release
```

### Run

```bash
# Basic usage
./target/release/gguf-infer --model model.gguf --prompt "Hello, how are you"

# With options
./target/release/gguf-infer \
  --model model.gguf \
  --prompt "The meaning of life is" \
  --max-tokens 200 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --seed 42
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

---

## Model Compatibility

| Model | Architecture | rope_base | Vocab Size | Status |
|-------|-------------|-----------|------------|--------|
| Llama 2 (7B/13B/70B) | llama | 10000.0 | 32000 | Supported |
| Llama 3 (8B/70B) | llama | 500000.0 | 128256 | Supported |
| CodeLlama (7B/13B/34B) | llama | 10000.0 | 32000 | Supported |
| Mistral (7B) | mistral | 10000.0 | 32000 | Supported |
| Qwen2 (0.5B/1.5B/7B/72B) | qwen2 | 1000000.0 | 151936 | Supported |

### Architecture Detection

The engine automatically detects the model architecture from GGUF metadata:
- Reads `general.architecture` key
- Uses architecture-specific defaults for missing parameters
- Supports: `llama`, `qwen2`, `mistral`

---

## Download GGUF Models

Download quantized GGUF files from Hugging Face:

| Model | Recommended Quant | Size | Source |
|-------|-------------------|------|--------|
| Llama 2 7B | Q4_K_M | ~4.4 GB | `TheBloke/Llama-2-7B-GGUF` |
| Llama 3 8B | Q4_K_M | ~4.9 GB | `QuantFactory/Meta-Llama-3-8B-GGUF` |
| Mistral 7B | Q4_K_M | ~4.4 GB | `TheBloke/Mistral-7B-v0.1-GGUF` |
| Qwen2 7B | Q4_K_M | ~4.4 GB | `QuantFactory/Qwen2-7B-GGUF` |
| Qwen2 1.5B | Q4_K_M | ~1.0 GB | `QuantFactory/Qwen2-1.5B-GGUF` |
| Qwen2 0.5B | Q4_K_M | ~0.4 GB | `QuantFactory/Qwen2-0.5B-GGUF` |

Example download with `wget`:

```bash
# Llama 2 7B
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Llama 3 8B
wget https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF/resolve/main/Meta-Llama-3-8B.Q4_K_M.gguf

# Mistral 7B
wget https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf

# Qwen2 7B
wget https://huggingface.co/QuantFactory/Qwen2-7B-GGUF/resolve/main/Qwen2-7B.Q4_K_M.gguf

# Qwen2 1.5B (lightweight, good for testing)
wget https://huggingface.co/QuantFactory/Qwen2-1.5B-GGUF/resolve/main/Qwen2-1.5B.Q4_K_M.gguf
```

---

## Running Each Model

### Llama 2

```bash
cargo run --release -- \
  --model llama-2-7b.Q4_K_M.gguf \
  --prompt "The capital of France is" \
  --max-tokens 100 \
  --temperature 0.7

# Greedy decoding (deterministic output)
cargo run --release -- \
  --model llama-2-7b.Q4_K_M.gguf \
  --prompt "Once upon a time" \
  --temperature 0

# Creative writing with high temperature
cargo run --release -- \
  --model llama-2-7b.Q4_K_M.gguf \
  --prompt "Write a poem about the ocean:" \
  --max-tokens 300 \
  --temperature 1.0 \
  --top-k 50 \
  --top-p 0.95
```

**Llama 2 specs:**
- Embed dim: 4096
- Layers: 32
- Heads: 32
- KV heads: 32
- Hidden dim: 11008
- Rope base: 10000.0

### Llama 3

```bash
cargo run --release -- \
  --model Meta-Llama-3-8B.Q4_K_M.gguf \
  --prompt "Explain quantum computing in simple terms:" \
  --max-tokens 200 \
  --temperature 0.8

# Code generation
cargo run --release -- \
  --model Meta-Llama-3-8B.Q4_K_M.gguf \
  --prompt "def fibonacci(n):" \
  --max-tokens 200 \
  --temperature 0.3

# Conversational
cargo run --release -- \
  --model Meta-Llama-3-8B.Q4_K_M.gguf \
  --prompt "What are the three laws of robotics?" \
  --max-tokens 300 \
  --temperature 0.7 \
  --top-k 40
```

**Llama 3 specs:**
- Embed dim: 4096
- Layers: 32
- Heads: 32
- KV heads: 8 (GQA - Grouped Query Attention)
- Hidden dim: 14336
- Rope base: 500000.0
- Vocab size: 128256

### Mistral 7B

```bash
cargo run --release -- \
  --model mistral-7b-v0.1.Q4_K_M.gguf \
  --prompt "def fibonacci(n):" \
  --max-tokens 200 \
  --temperature 0.3

# Question answering
cargo run --release -- \
  --model mistral-7b-v0.1.Q4_K_M.gguf \
  --prompt "What is the speed of light?" \
  --max-tokens 150 \
  --temperature 0.5
```

**Mistral 7B specs:**
- Embed dim: 4096
- Layers: 32
- Heads: 32
- KV heads: 8 (GQA)
- Hidden dim: 14336
- Rope base: 10000.0

### Qwen2

```bash
# Qwen2 7B
cargo run --release -- \
  --model Qwen2-7B.Q4_K_M.gguf \
  --prompt "What is machine learning?" \
  --max-tokens 300 \
  --temperature 0.7

# Qwen2 1.5B (fast, lightweight)
cargo run --release -- \
  --model Qwen2-1.5B.Q4_K_M.gguf \
  --prompt "Write a haiku about programming:" \
  --max-tokens 100 \
  --temperature 0.8

# Qwen2 0.5B (ultra-lightweight)
cargo run --release -- \
  --model Qwen2-0.5B.Q4_K_M.gguf \
  --prompt "1 + 1 =" \
  --max-tokens 50 \
  --temperature 0.0
```

**Qwen2 specs:**

| Model | Embed Dim | Layers | Heads | KV Heads | Hidden Dim |
|-------|-----------|--------|-------|----------|------------|
| 0.5B | 896 | 24 | 14 | 2 | 4864 |
| 1.5B | 1536 | 28 | 12 | 2 | 8960 |
| 7B | 3584 | 28 | 28 | 4 | 18944 |

Common: RoPE base=`1000000.0`, Vocab size=`151936`, Norm eps=`1e-6`, Head dim=`64`

---

## Which Quantization to Choose?

| Quant | Size (7B) | Quality | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| F16 | ~14 GB | Best | Slowest | Maximum quality |
| Q8_0 | ~7.5 GB | Excellent | Slow | High quality |
| Q6_K | ~5.5 GB | Very Good | Medium | Good balance |
| Q5_K_M | ~5.0 GB | Good | Medium | Balanced |
| Q4_K_M | ~4.4 GB | Good | Fast | Recommended default |
| Q4_K_S | ~4.1 GB | Acceptable | Fast | Lower memory |
| Q4_0 | ~4.0 GB | Acceptable | Fastest | Fastest inference |

**Recommendation:** Start with `Q4_K_M` for best balance of quality and speed.

---

## Project Structure

```
src/
+-- gguf.rs          # GGUF file format parser
|   +-- Magic bytes (0x46554747)
|   +-- Version (u32, supports v3)
|   +-- Tensor count, Metadata count
|   +-- Key-value metadata pairs
|   +-- Tensor descriptors (name, dims, type, offset)
|   +-- Data alignment (512 bytes)
|
+-- tensor.rs        # Tensor data structure
|   +-- Shape handling (Vec<usize>)
|   +-- Flat f32 storage
|   +-- Row-major indexing
|   +-- 2D/3D views
|
+-- math.rs          # Mathematical operations
|   +-- Vector ops: add, mul, scale, dot product
|   +-- Matrix ops: matmul, matvec
|   +-- Softmax (numerically stable)
|   +-- SiLU activation
|   +-- RMSNorm
|   +-- Argmax, Top-K
|
+-- quant.rs         # Quantization dequantization
|   +-- FP16 -> FP32 conversion
|   +-- Q8_0 (34 bytes/block, 32 values)
|   +-- Q4_0 (18 bytes/block, 32 values)
|   +-- Q4_K (144 bytes/block, 256 values)
|   +-- Q5_K (176 bytes/block, 256 values)
|   +-- Q6_K (210 bytes/block, 256 values)
|
+-- tokenizer.rs     # BPE tokenizer
|   +-- Load vocab from GGUF metadata
|   +-- Load merge rules
|   +-- Pre-tokenization (whitespace, punctuation)
|   +-- BPE encoding
|   +-- Decode tokens to text
|
+-- rope.rs          # Rotary Position Embedding
|   +-- Precompute frequency table
|   +-- theta_i = 1 / (base^(2i/dim))
|   +-- Rotate query/key vectors
|
+-- rmsnorm.rs       # RMS Normalization
|   +-- y = x * w / sqrt(mean(x^2) + epsilon)
|
+-- attention.rs     # Multi-Head Self-Attention
|   +-- Q/K/V projections
|   +-- GQA (Grouped Query Attention)
|   +-- Causal masking
|   +-- Output projection
|
+-- mlp.rs           # Feed-Forward Network
|   +-- Gate projection
|   +-- Up projection
|   +-- SiLU activation
|   +-- Down projection
|
+-- kv_cache.rs      # Key-Value Cache
|   +-- Pre-allocated buffers
|   +-- Append new states
|   +-- Retrieve for attention
|
+-- sampler.rs       # Token sampling
|   +-- Greedy decoding
|   +-- Temperature scaling
|   +-- Top-K filtering
|   +-- Top-P (nucleus) filtering
|   +-- Repetition penalty
|
+-- transformer.rs   # Transformer stack
|   +-- Layer loop
|   +-- Pre-norm architecture
|   +-- Residual connections
|
+-- model.rs         # Model loading & inference
|   +-- Parse GGUF metadata for config
|   +-- Load & dequantize weights
|   +-- Build transformer layers
|   +-- generate() API
|
+-- main.rs          # CLI interface
    +-- Argument parsing
    +-- Model loading
    +-- Text generation loop
```

---

## GGUF Format Specification

### Header Layout

```
Offset  Size   Field
------  ----   ---------------------------
0x00    4      Magic: "GGUF" (0x46554747)
0x04    4      Version: 3
0x08    8      Tensor count (u64)
0x10    8      Metadata count (u64)
0x18    var    Metadata key-value pairs
  ...   var    Tensor descriptors
  ...   pad    Alignment to 512 bytes
  ...   var    Tensor data
```

### Metadata Value Types

| ID | Type | Size |
|----|------|------|
| 0 | UINT8 | 1 byte |
| 1 | INT8 | 1 byte |
| 2 | UINT16 | 2 bytes |
| 3 | INT16 | 2 bytes |
| 4 | UINT32 | 4 bytes |
| 5 | INT32 | 4 bytes |
| 6 | FLOAT32 | 4 bytes |
| 7 | BOOL | 1 byte |
| 8 | STRING | len(u64) + bytes |
| 9 | ARRAY | type + count + elements |
| 10 | UINT64 | 8 bytes |
| 11 | INT64 | 8 bytes |
| 12 | FLOAT64 | 8 bytes |

### Tensor Data Types

| ID | Type | Block Size | Bytes/Block |
|----|------|------------|-------------|
| 0 | F32 | 1 | 4 |
| 1 | F16 | 1 | 2 |
| 2 | Q4_0 | 32 | 18 |
| 8 | Q8_0 | 32 | 34 |
| 12 | Q4_K | 256 | 144 |
| 13 | Q5_K | 256 | 176 |
| 14 | Q6_K | 256 | 210 |

### LLaMA/Qwen2 Tensor Names

```
token_embd.weight                    # (vocab_size, embed_dim)
output_norm.weight                   # (embed_dim,)
output.weight                        # (vocab_size, embed_dim) [optional, tied]

blk.{i}.attn_norm.weight             # (embed_dim,)
blk.{i}.attn_q.weight                # (n_heads x head_dim, embed_dim)
blk.{i}.attn_k.weight                # (n_kv_heads x head_dim, embed_dim)
blk.{i}.attn_v.weight                # (n_kv_heads x head_dim, embed_dim)
blk.{i}.attn_output.weight           # (embed_dim, n_heads x head_dim)

blk.{i}.ffn_norm.weight              # (embed_dim,)
blk.{i}.ffn_gate.weight              # (hidden_dim, embed_dim)
blk.{i}.ffn_up.weight                # (hidden_dim, embed_dim)
blk.{i}.ffn_down.weight              # (embed_dim, hidden_dim)
```

---

## Quantization Details

### Q8_0 (8-bit quantization)

```
Block size: 32 values
Block bytes: 34

Layout: [f16 d | int8 qs[32]]

Dequantization:
  value[i] = d * qs[i]
  where d = fp16_to_fp32(block[0..2])
```

### Q4_0 (4-bit quantization)

```
Block size: 32 values
Block bytes: 18

Layout: [f16 d | uint8 qs[16]]

Dequantization:
  nibble = (qs[i/2] >> (4*(i%2))) & 0x0F
  value[i] = d * (nibble - 8)
```

### Q4_K (4-bit K-quant)

```
Block size: 256 values
Block bytes: 144

Layout: [f16 d | f16 dmin | uint8 scales[8] | uint8 qs[128]]

8 sub-blocks of 32 values each.
Each sub-block has its own scale nibble.
```

### Q5_K (5-bit K-quant)

```
Block size: 256 values
Block bytes: 176

Layout: [f16 d | f16 dmin | uint8 scales[8] | uint8 qh[32] | uint8 qs[128]]

5-bit values: 4 bits from nibble + 1 high bit from qh.
```

### Q6_K (6-bit K-quant)

```
Block size: 256 values
Block bytes: 210

Layout: [uint8 scales[8] | f16 d | uint8 ql[192] | uint8 qh[64]]

6-bit values: 4 bits from nibble + 2 high bits from qh.
```

---

## Tokenizer

The tokenizer is loaded from GGUF metadata keys:

| Key | Type | Description |
|-----|------|-------------|
| `tokenizer.ggml.tokens` | string[] | Vocabulary |
| `tokenizer.ggml.scores` | f32[] | Token scores |
| `tokenizer.ggml.token_type` | i32[] | Token types (1=normal, 3=control, 4=byte) |
| `tokenizer.ggml.merges` | string[] | BPE merge rules ("token_a token_b") |
| `tokenizer.ggml.pre` | string | Pre-tokenizer type |
| `tokenizer.ggml.bos_token_id` | i32 | Beginning of sequence token |
| `tokenizer.ggml.eos_token_id` | i32 | End of sequence token |

### Tokenization Process

```
Input: "Hello, world!"

1. Pre-tokenize:
   ["Hello", ",", " ", "world", "!"]

2. Byte-level tokens:
   ["H", "e", "l", "l", "o"]

3. Apply BPE merges (in priority order):
   "l" + "l" -> "ll"
   "e" + "ll" -> "ell"
   "H" + "ell" -> "Hell"
   "Hell" + "o" -> "Hello"

4. Map to token IDs:
   [15043, 29892, 278, ...]
```

---

## Inference Flow

### Prefill Phase

```
Prompt: "Hello"

1. Tokenize: "Hello" -> [15043]

2. For position 0:
   x = token_embd[15043]           # (embed_dim,)

   For each layer i:
     x_norm = RMSNorm(x)
     Q, K, V = project(x_norm)     # Q = x*Wq, K = x*Wk, V = x*Wv
     Q, K = apply_rope(Q, K, pos=0)
     cache[i].append(K, V)
     attn = attention(Q, cache[i])  # K,V from cache
     x = x + attn                   # Residual

     x_norm = RMSNorm(x)
     ffn = swiglu(x_norm)
     x = x + ffn                    # Residual

   x = final_rmsnorm(x)
   logits = x * W_lm_head          # (vocab_size,)

3. Sample token from logits
```

### Generation Phase

```
For each new token:

1. Forward pass (same as prefill, but only 1 position)
2. Sample next token from logits
3. Append to output
4. Repeat until EOS or max_tokens
```

### KV Cache

```
Layer 0 cache:
  keys:   [K_pos_0 | K_pos_1 | ... | K_pos_t]   # (seq_len, kv_dim)
  values: [V_pos_0 | V_pos_1 | ... | V_pos_t]

Attention uses all cached K,V for the current token.
```

---

## Sampling Strategies

### Greedy (temperature=0)

```
token = argmax(logits)
```

### Temperature

```
probs = softmax(logits / temperature)
token = sample(probs)
```

### Top-K

```
Keep only K highest probability tokens.
Zero out the rest, renormalize, sample.
```

### Top-P (Nucleus)

```
Sort tokens by probability (descending).
Keep smallest set with cumulative probability >= P.
Zero out the rest, renormalize, sample.
```

### Repetition Penalty

```
For each token in recent window:
  if logit > 0: logit /= penalty
  else: logit *= penalty
```

---

## Performance Tips

1. **Use release build:** Always use `--release` for meaningful performance
2. **Quantization matters:** Q4_K_M is 3x faster than F16 with good quality
3. **Prompt length:** Shorter prompts = faster prefill
4. **Max tokens:** Set reasonable `--max-tokens` to avoid long waits
5. **Temperature 0:** Fastest decoding (no sampling overhead)

### Memory Requirements

| Model Size | F16 | Q8_0 | Q4_K_M |
|------------|-----|------|--------|
| 0.5B | 1.0 GB | 0.6 GB | 0.4 GB |
| 1.5B | 3.0 GB | 1.7 GB | 1.0 GB |
| 7B | 14 GB | 7.5 GB | 4.4 GB |
| 13B | 26 GB | 14 GB | 8.0 GB |
| 70B | 140 GB | 75 GB | 40 GB |

---

## Performance Characteristics

- **Memory**: All weights dequantized to f32 on load (~4x quantized size)
- **Compute**: Scalar f32 operations (compiler may auto-vectorize)
- **Cache**: Pre-allocated KV cache, no allocation during generation
- **I/O**: Single file read, sequential tensor loading

### Verified Accuracy

| Prompt | Logit Correlation | Model |
|--------|------------------|-------|
| "The" | 0.9956 | Qwen2-0.5B Q4_K_M |

Logits compared against `llama_cpp` Python reference. Top-1 tokens match after Q5_0 dequant fix. Minor differences (~0.5%) from FP precision in dequantization.

### Quantization Support

| Type | Status | Note |
|------|--------|------|
| F32 | Supported | No dequant needed |
| F16 | Supported | Verified |
| Q8_0 | Supported | Verified |
| Q4_0 | Supported | Verified |
| Q5_0 | Supported | **Fixed** — non-interleaved nibble order matches GGML master |
| Q4_K | Supported | Verified |
| Q5_K | Supported | Verified |
| Q6_K | Supported | Verified |

## Limitations

- No SIMD intrinsics (relies on compiler auto-vectorization)
- No GPU acceleration
- No batch processing (single token at a time)
- No flash attention
- Quantized weights dequantized to f32 (higher memory usage)
- Sliding window attention not implemented (Mistral)
- Qwen2 does not use BOS token (despite GGUF metadata having it)
- `output.weight` is optional; when missing, `token_embd.weight` is used as tied lm_head

## Dependencies

**None.** Only Rust standard library (`std::io`, `std::fs`, `std::collections`).

## License

Public domain. Use however you like.
