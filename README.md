# bihaba-serve

**Zero-Dependency Native Apple Metal LLM Inference Engine & OpenAI-Compatible Server for Apple Silicon.**

> **Created by Boy Barley from Bihaba — Team Pasai Aceh**
> Powered by the custom **Taalas-Metal** compute runtime (Objective-C++ & Apple Metal).

`bihaba-serve` runs quantized **Qwen3.5 Dense & 256-Expert MoE** models directly from standard `.safetensors` directories via zero-copy `mmap()` — with **0 GB disk duplication**, **~270 ms cold start**, **100+ tok/s greedy**, **250+ tok/s lossless speculative decoding**, and **<100 ms multi-user TTFT** via baked GPU state snapshots, all without PyTorch, MLX runtime, FastAPI, or Conda bloat.

---

## Key Features

- **Native Apple Metal Compute (`engines/qwen35/kernels.metal`):** Hand-tuned Metal shaders with `BM=8, BN=32, BK=64` compact-half Hardware MMA (`simdgroup_half8x8`), register-only 32-lane SIMD Top-K routing, 3D contiguous-expert MoE kernels, and native **BF16 / FP16** affine dequantization (`2-bit`, `3-bit`, `4-bit`, `8-bit`).
- **Direct Zero-Copy `.safetensors` Multi-Shard `mmap()`:** Point directly to any `~/models/<model>` directory — zero format conversion and **0 GB extra SSD space**. Includes `bihaba-serve align` for in-place 64-byte header alignment (~270 ms cold start).
- **Zero-Copy Speculative Chunked DeltaNet (`M = 1..8`, `k = 0..7`):** Reads the FP32 DeltaNet recurrent state once from DRAM, steps all `M` draft tokens in GPU registers, and commits only accepted tokens — **100% bit-exact (`128/128`)** with zero state-backup memory.
- **2-Level Baked Prefix State (`.tstate`):** Automatically snapshots shared system prompts (`messages[0]`) in GPU RAM and supports persistent disk `.tstate` baking (`bihaba-serve bake` / `POST /v1/state/bake`), cutting multi-user TTFT by **>20×**.
- **Exact GPU Top-64 Sampling + Lossless Rejection Sampling:** Supports `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, and `stop` sequences without breaking speculative exactness.
- **Zero Web Dependencies:** OpenAI-compatible HTTP server (`/v1/chat/completions` with SSE streaming, `/v1/models`, `/v1/state/bake`, `/health`, full CORS) built purely on Python's standard library + `tokenizers`.
- **Multi-Model Hot-Swap (`--models-dir`):** Serve your entire local Qwen collection and switch models on-the-fly via the OpenAI `model` field (~270 ms swap).

---

## Verified Benchmarks (Apple M1 Max, 64 GB Unified Memory)

All benchmarks measured directly from `.safetensors` directories (**0 GB disk duplication**, full exact vocabulary, **100% bit-exact** speculative verification across both `qwen35` and `transformer` engines):

| Model (`~/models`) | Architecture | Size | Cold Start | Greedy `M=1` | Speculative `k=7` | Multi-User Baked TTFT |
|---|---|---|---|---|---|---|
| **Bihaba-bagus-pakai** (`--top-k 4`) | 40L 35B-A3B MoE (4b) | 20.40 GB | **273 ms** | **100.3 tok/s** | **249.2 tok/s** | **62.7 ms** *(vs 1,311 ms cold)* |
| **Qwen3.6-35B-A3B-Hyperion-2Bit-Top4** | 40L 35B-A3B MoE (2b/4b) | 12.26 GB | **433 ms** | **101.0 tok/s** | **252.2 tok/s** | **61.7 ms** *(vs 1,304 ms cold)* |
| **bihaba-v16-hyperion-q8-3bit-top4** | 40L 35B-A3B MoE (3b/8b) | 18.12 GB | **396 ms** | **92.0 tok/s** | **195.3 tok/s** | **77.7 ms** *(vs 1,687 ms cold)* |
| **Qwythos-9B-Claude** | 32L 9B Dense (4b) | 5.04 GB | **45 ms** | **50.1 tok/s** | **153.9 tok/s** | — |
| **qwen3.8-27b-hyperion-60L-2bit** | 60L 27B Dense (2b) | 10.53 GB | **290 ms** | **16.5 tok/s** | **50.4 tok/s** | — |
| **Bihaba-AI-1.7B-Final** | 28L Qwen3 1.7B Dense (6b) | 1.40 GB | **367 ms** | **131.5 tok/s** | **222.4 tok/s** | — |
| **MNN-Pocket-0.3B-MLX-4bit** | 28L Qwen3 0.3B Dense (4b) | 0.17 GB | **337 ms** | **337.6 tok/s** | **1,205.4 tok/s** | — |

> **How to read the `Speculative k=7` column:** these numbers come from `bench`, whose prompt is a *synthetic repeating pattern* (the best case for n-gram drafting). On real chat/code the speedup depends on how repetitive the output is: typically 1.2–2× for code/JSON/RAG and ≈1× for free-form prose. The speculation is lossless (the output is identical to greedy), so it never makes quality worse.

### v1.0 → v1.2 on the same machine (M1 Max 64 GB, same models, same session)

| Model | Metric | v1.0 | v1.2 | Gain |
|---|---|---|---|---|
| **Bihaba-AI-1.7B** (6-bit) | Prefill 512 tok | 275 tok/s | **765 tok/s** | 2.8× |
| | Speculative k=7 (synthetic) | 232.7 tok/s | **457.5 tok/s** | 1.97× |
| | Verify step M=8 | 32.0 ms | **16.8 ms** | 1.9× |
| **Qwythos-9B** (4-bit, DeltaNet) | Prefill 512 tok | 104 tok/s | **241 tok/s** | 2.3× |
| | Speculative k=7 (synthetic) | 141.3 tok/s | 141.3 tok/s | = |
| **35B-A3B MoE** (`--top-k 4`) | Prefill 512 tok | 217 tok/s | **343 tok/s** | 1.58× |
| | Speculative k=7 (synthetic) | 246.6 tok/s | 246.8 tok/s | = |
| All | Greedy decode M=1 | | unchanged | = |

Every run reports `=greedy 128/128` (speculative output identical to greedy) and `Prefill check 32/32` (the GEMM prefill produces exactly the same greedy continuation as the v1.0 8-token path).

---

## Supported Models

| Engine Plugin | Architectures | Example Models | Quantization Format |
|---|---|---|---|
| `engines/qwen35/` *(Active)* | `qwen3_5`, `qwen3_5_moe` (DeltaNet + FullAttn `hd=256` + Partial RoPE / YaRN) | Qwen3.5 0.8B, 9B, 27B, 35B-A3B MoE, Qwen3.6/3.8 MLX fine-tunes | Direct `.safetensors` affine MLX (`2b`/`3b`/`4b`/`8b`, `gs=32/64`, BF16/FP16) |
| `engines/transformer/` *(Active)* | `qwen2`, `qwen3`, `llama`, `mistral` | Qwen2.5 0.5B/1.5B, Qwen3 1.7B, Llama-3.2 1B, MNN-Pocket 0.3B | Direct `.safetensors` affine MLX (`2b`/`3b`/`4b`/`6b`/`8b`, `gs=32/64`, BF16/FP16) |

> **Not supported yet:** GGUF containers and multimodal vision towers.

---

## System Requirements

- **Hardware:** Apple Silicon Mac (`M1`, `M2`, `M3`, or `M4`)
- **OS:** macOS 14+ (Sonoma / Sequoia)
- **Compiler:** Xcode Command Line Tools (`xcode-select --install`)
- **Python:** Python 3.9+ and **1 package only**: `tokenizers` (installed automatically by `./install.sh`)

---

## Installation

```zsh
git clone https://github.com/boybarley/bihaba-serve.git
cd bihaba-serve
./install.sh
```

If `~/.local/bin` is not yet in your `PATH`:
```zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

---

## Usage

### 1. Align `.safetensors` Headers Once (Optional, 0 GB Extra Disk)
Pads trailing whitespace inside `.safetensors` JSON headers to 64-byte boundaries (100% spec-compliant with MLX & PyTorch) so Metal maps weights with zero copies in ~270 ms:
```zsh
bihaba-serve align ~/models/Bihaba-bagus-pakai
```

### 2. List Compatible Models
```zsh
bihaba-serve list ~/models
```

### 3. Start OpenAI-Compatible API Server
```zsh
# Single model (binds 127.0.0.1 by default)
bihaba-serve serve --model ~/models/Bihaba-bagus-pakai --top-k 4 --temp 0.6 --port 8090

# Exposed on the LAN: always with an API key; allow HTTP baking only into one folder
bihaba-serve serve --model ~/models/Bihaba-bagus-pakai --top-k 4 --host 0.0.0.0 \
  --api-key "$(openssl rand -hex 24)" --state-dir ~/.bihaba/states --cors '*'

# Multi-model hot-swap from your ~/models directory
bihaba-serve serve --models-dir ~/models --model ~/models/Bihaba-bagus-pakai --top-k 4 --port 8090
```

### 4. Interactive Terminal Chat
```zsh
bihaba-serve chat --model ~/models/Bihaba-bagus-pakai --top-k 4 --temp 0.6
```

### 5. Bake a System Prompt / RAG Context to Disk (`.tstate`)
```zsh
bihaba-serve bake --model ~/models/Bihaba-bagus-pakai --top-k 4   --system-file deploy/system_prompt.example.txt --out bihaba_sys.tstate

bihaba-serve serve --model ~/models/Bihaba-bagus-pakai --top-k 4   --load-state bihaba_sys.tstate --port 8090
```

### 6. Run Native Metal GPU Benchmark & X-Ray
```zsh
bihaba-serve bench --model ~/models/Bihaba-bagus-pakai --top-k 4 --profile --tokens 128
```

---

## What's new in v1.1 / v1.2

**Faster**
- **Prefill uses a dedicated GEMM kernel** (`*_gemm`, 32 tokens × 64 weight rows per threadgroup, 8 accumulators per simdgroup). It covers every bit-width (2/3/4/6/8). Each weight tile is dequantized once and reused for 32 tokens. Verification steps (M ≤ 8) now also use MMA for 3/6/8-bit models. DeltaNet prefill now runs a dedicated kernel that keeps the state in registers for the whole chunk. The LM head is computed only for the last prompt token. Control it with `--prefill-chunk N` (1..32; 8 = v1.0 behaviour).
- **Speculative decoding:** a per-n-gram-order acceptance estimator, α = accepted / (accepted + rejections), replaces the old EMA, which penalized k too hard after a single rejection.

**Correct**
- **Top-64 sampling is now exact.** Each thread tracks the largest candidate it evicted. If any evicted candidate beats the 64th result, that row is recomputed on the CPU (rare). v1.0 silently dropped candidates on a large fraction of steps.
- **Repetition penalty per verification row:** row *m* sees history plus drafts 1..m, so the speculative output equals the non-speculative output. The default is now `1.0` (off).
- **Bounds checks:**
  - `max_seq` is checked on every request;
  - token ids and `prefix_len` are validated;
  - unsupported architectures and quantization are rejected at load instead of producing garbage.
- **`.tstate` v2:** the header carries a model fingerprint and every size is checked before loading, and saves are atomic. v1 files are rejected, so re-bake them.
- **Stream fixes:**
  - when a client disconnects mid-stream, the engine output is drained, so it no longer leaks into the next request;
  - engine errors become HTTP 400;
  - a model hot-swap no longer closes a daemon that another request is still using.
- **`align` writes through HuggingFace-cache symlinks.** The cache stays intact and the disk is not doubled. It also checks free disk space first.

**Secure by default**
- The server binds `127.0.0.1`.
- `--api-key` (or env `BIHABA_API_KEY`) enforces Bearer authentication.
- CORS is off unless `--cors` is given.
- Request bodies are limited to 16 MB.
- `/v1/state/bake` can write only `<name>.tstate` inside `--state-dir`. In v1.0 the HTTP `path` field could write anywhere on disk.

### Testing v1.1 on your Mac
```zsh
make                                                        # build both engines (Metal compiles at first run)
python3 -m unittest discover -s tests -v                    # offline tests: no GPU or model needed
bihaba-serve bench --model ~/models/Bihaba-bagus-pakai --top-k 4 --tokens 128                     # decode + prefill A/B
bihaba-serve bench --model ~/models/Bihaba-bagus-pakai --top-k 4 --tokens 128 --prefill-chunk 8   # v1.0 prefill path
```
`bench` prints a prefill A/B (chunk 8 vs active chunk, 512 tokens) and the greedy-vs-speculative match (`=greedy`).

---

## Architecture & Modularity

```text
bihaba-serve/
├── engines/
│   ├── qwen35/          # Isolated Qwen3.5 / MoE Objective-C++ & Metal engine
│   └── transformer/     # Isolated Qwen2.5 / Qwen3 / Llama / MNN-Pocket Metal engine (0.3B–8B)
├── python/bihaba_serve/ # Zero-dependency control plane (CLI, detector, templates, OpenAI API)
├── deploy/              # launchd 24/7 plist template & example system prompt
├── Makefile             # Partial builds: `make qwen35`, `make transformer`, `make all`
├── install.sh           # 1-command installer
└── uninstall.sh         # Clean uninstaller (never touches ~/models)
```

All engines communicate with the Python control plane through a single **NDJSON stdin/stdout contract**, allowing compute kernels to be optimized—or rewritten in Zig/Rust—without modifying the API server or CLI.

---

## Uninstall

```zsh
cd bihaba-serve
./uninstall.sh
```
Removes `~/.local/bin/bihaba-serve`, `~/.local/share/bihaba-serve`, `build/`, and any installed `launchd` service. Your `~/models` directory is **never** modified.

---

## Credits

Created by **Boy Barley** from **Bihaba — Team Pasai Aceh**.
Released under the **MIT License**.
