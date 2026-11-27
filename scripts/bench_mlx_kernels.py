#!/usr/bin/env python3
"""Microbench mx.quantized_matmul (M=1) pada tensor ASLI model, untuk dibandingkan dengan
`BIHABA_BENCH_KERNELS=1 bihaba-serve bench` (baris "Kernel microbench bihaba").

Jalankan di lingkungan MLX terbaru (Python ≥3.10), mis.:
  source ~/mlx-env/bin/activate && python scripts/bench_mlx_kernels.py ~/models/Qwythos-9B-Claude

Setiap matmul dibuat bergantung pada hasil sebelumnya (seperti decode yang serial), sehingga MLX tidak
bisa menjalankannya bersamaan. Biaya rantai dependensi (2 kernel kecil) ikut terhitung → angka MLX
sedikit pesimis, bukan optimis.
"""
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

R = 100


def find(index: dict, pattern: str):
    rx = re.compile(pattern)
    for k in sorted(index, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]):
        if rx.search(k):
            return k
    return None


def main() -> int:
    mdir = Path(sys.argv[1]).expanduser()
    cfg = json.loads((mdir / "config.json").read_text())
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    idx_path = mdir / "model.safetensors.index.json"
    index = json.loads(idx_path.read_text())["weight_map"] if idx_path.exists() else \
        {k: "model.safetensors" for k in mx.load(str(mdir / "model.safetensors"))}
    cache: dict = {}

    def tensor(name):
        f = index[name]
        if f not in cache:
            cache[f] = mx.load(str(mdir / f))
        return cache[f][name]

    def qparams(base):
        o = q.get(base.rsplit(".weight", 1)[0], q) if isinstance(q, dict) else {}
        o = o if isinstance(o, dict) else {}
        return int(o.get("group_size", q.get("group_size", 64))), int(o.get("bits", q.get("bits", 4)))

    targets = [
        ("ffn gate", r"layers\.\d+\.mlp\.gate_proj\.weight$"),
        ("ffn up", r"layers\.\d+\.mlp\.up_proj\.weight$"),
        ("ffn down", r"layers\.\d+\.mlp\.down_proj\.weight$"),
        ("dn in_proj_qkv", r"layers\.\d+\.linear_attn\.in_proj_qkv\.weight$"),
        ("dn in_proj_z", r"layers\.\d+\.linear_attn\.in_proj_z\.weight$"),
        ("dn in_proj_a", r"layers\.\d+\.linear_attn\.in_proj_a\.weight$"),
        ("dn in_proj_b", r"layers\.\d+\.linear_attn\.in_proj_b\.weight$"),
        ("attn q_proj", r"layers\.\d+\.self_attn\.q_proj\.weight$"),
        ("attn k_proj", r"layers\.\d+\.self_attn\.k_proj\.weight$"),
        ("attn v_proj", r"layers\.\d+\.self_attn\.v_proj\.weight$"),
        ("attn o_proj", r"layers\.\d+\.self_attn\.o_proj\.weight$"),
        ("deltanet out_proj", r"layers\.\d+\.linear_attn\.out_proj\.weight$"),
        ("lm_head", r"(^|\.)lm_head\.weight$"),
    ]
    print(f"▸ Kernel microbench MLX {mx.__version__} (M=1, R={R}, serial, quantized_matmul):")
    total_ffn = 0.0
    sums: dict = {}
    for nm, pat in targets:
        key = find(index, pat)
        if key is None:
            print(f"   {nm:20s} (tidak ada)")
            continue
        base = key[: -len(".weight")]
        if base + ".scales" not in index:
            print(f"   {nm:20s} (tidak terkuantisasi, dilewati)")
            continue
        w, s, b = tensor(key), tensor(base + ".scales"), tensor(base + ".biases")
        gs, bits = qparams(key)
        N, K = w.shape[0], w.shape[1] * 32 // bits
        x = mx.random.normal((1, K)).astype(s.dtype)
        mx.eval(w, s, b, x)

        def run(x0):
            y = None
            xi = x0
            for _ in range(R):
                y = mx.quantized_matmul(xi, w, s, b, transpose=True, group_size=gs, bits=bits)
                xi = xi + y[:, :1] * 0          # dependensi serial
            return y

        for _ in range(3):
            mx.eval(run(x))
        best = 1e30
        for _ in range(3):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(run(x))
            mx.synchronize()
            best = min(best, (time.perf_counter() - t0) / R)
        us = best * 1e6
        nbytes = N * K * bits / 8 + 2 * N * (K // gs) * s.itemsize
        if nm in ("ffn gate", "ffn up"):
            total_ffn += us
        if nm.startswith("dn in_proj"):
            sums["dn"] = sums.get("dn", 0.0) + us
        if nm in ("attn q_proj", "attn k_proj", "attn v_proj"):
            sums["qkv"] = sums.get("qkv", 0.0) + us
        print(f"   {nm:20s} N={N:<6d} K={K:<6d} {bits}b : {us:8.1f} us  {nbytes / (us * 1e3):5.0f} GB/s")
    if total_ffn:
        print(f"   {'ffn gate+up (jumlah)':20s} {'':22s}: {total_ffn:8.1f} us")
    if "dn" in sums:
        print(f"   {'dn in_proj (jumlah 4)':20s} {'':22s}: {sums['dn']:8.1f} us")
    if "qkv" in sums:
        print(f"   {'attn q+k+v (jumlah)':20s} {'':22s}: {sums['qkv']:8.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
