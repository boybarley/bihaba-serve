# Standard GQA Transformer Engine (`engines/transformer/`)

Isolated native Apple Metal engine for standard GQA Transformer architectures:
- **Qwen2.5** (`0.5B`, `1.5B`, `3B`, `7B`) — with automatic QKV bias handling
- **Qwen3** (`0.6B`, `1.7B`, `4B`, `8B` — e.g. `Bihaba-AI-1.7B-Final`) — with automatic per-head QK-Norm
- **Llama-3 / 3.2** (`1B`, `3B`, `8B`) — with Llama-3 frequency-scaled RoPE
- **MNN-Pocket** (`0.3B`)

Supports direct zero-copy `.safetensors` loading (**2b / 3b / 4b / 6b / 8b**, Native BF16/FP16), `head_dim = 64 / 128 / 256`, `BM=8` Hardware MMA, Adaptive Suffix Speculative Decoding (`k=0..7`), GPU Top-64 Sampling, and Baked KV-Cache `.tstate` Snapshots.

Build independently without touching `qwen35`:
```zsh
make transformer
```
