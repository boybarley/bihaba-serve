#!/usr/bin/env python3
"""
Uji regresi end-to-end di Mac (engine + tokenizer + daemon asli, tanpa HTTP).

Untuk setiap model: percakapan 3 giliran, greedy (temperature 0), dua kali:
  A = cache sesi MATI (riwayat dirender ulang tiap giliran, perilaku sebelum v1.5)
  B = cache sesi HIDUP (v1.5)
Lalu dicetak jawaban keduanya + reuse/TTFT, supaya kerusakan kualitas langsung terlihat dan bisa
dilacak ke penyebabnya (cache sesi vs engine).

Pakai (versi stabil):     python3 scripts/regress_chat.py ~/models/MNN-Pocket-0.3B-MLX-4bit ~/models/Bihaba-AI-1.7B-Final
Pakai (folder eksperimen): cd ~/bihaba-dev && python3 scripts/regress_chat.py ...
Opsi: --top-k 4 (MoE)  --max-seq 4096
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from bihaba_serve.daemon import EngineDaemon  # noqa: E402
from bihaba_serve.detector import detect_model  # noqa: E402

TURNS = [
    "Siapa kamu? Jawab singkat dalam satu kalimat.",
    "Sebutkan 3 buah tropis, satu per baris.",
    "Dari daftar itu, mana yang paling manis? Jawab satu kalimat.",
]


def run(info, session: bool, top_k: int, max_seq: int):
    d = EngineDaemon(info, REPO, session_cache=session, top_k_moe=top_k, max_seq=max_seq, default_temp=0.0)
    msgs = [{"role": "system", "content": "Kamu asisten yang ringkas dan akurat."}]
    out = []
    try:
        for t in TURNS:
            msgs.append({"role": "user", "content": t})
            text, st = [], {}
            for ev in d.stream_tokens(msgs, max_tokens=96, temperature=0.0):
                if "delta" in ev:
                    text.append(ev["delta"])
                elif "error" in ev:
                    text.append(f"[ERROR] {ev['error']}")
                elif ev.get("done"):
                    st = ev["stats"]
            answer = "".join(text)
            msgs.append({"role": "assistant", "content": answer})
            out.append((answer, st))
    finally:
        d.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=4096)
    a = ap.parse_args()
    for mp in a.models:
        info = detect_model(mp)
        print(f"\n{'=' * 78}\n■ {info.name}  (engine={info.engine}, template={info.template})")
        res = {s: run(info, s, a.top_k, a.max_seq) for s in (False, True)}
        for i, t in enumerate(TURNS):
            (ta, sa), (tb, sb) = res[False][i], res[True][i]
            same = ta.strip() == tb.strip()
            print(f"\n  Giliran {i + 1}: {t}")
            print(f"    A (tanpa sesi) reuse={sa.get('reused', '?'):>5}  prefill={sa.get('prefill_ms', 0):7.1f} ms | {ta.strip()[:300]!r}")
            print(f"    B (cache sesi) reuse={sb.get('reused', '?'):>5}  prefill={sb.get('prefill_ms', 0):7.1f} ms | {tb.strip()[:300]!r}")
            print(f"    → {'IDENTIK' if same else 'BERBEDA (cek apakah B rusak/ngawur dibanding A)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
