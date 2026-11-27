#!/usr/bin/env python3
"""
Uji regresi end-to-end di Mac (engine + tokenizer + daemon asli, tanpa HTTP).

Setiap model: percakapan 3 giliran, greedy (temperature 0), jalur kanonik (cache sesi mati = default).
Model yang namanya mengandung "MNN" memakai pertanyaan bahasa Inggris (model itu tidak bisa bahasa Indonesia).

Alur aman untuk setiap build baru:
  1) di versi STABIL :  cd ~/bihaba-serve && python3 scripts/regress_chat.py --save /tmp/ref.json  <model...>
  2) di build BARU   :  cd ~/bihaba-dev   && python3 scripts/regress_chat.py --compare /tmp/ref.json <model...>
     → setiap jawaban harus IDENTIK dengan versi stabil. Satu saja BERBEDA = build ditolak.
Opsi: --with-session (juga jalankan jalur cache sesi untuk dibandingkan), --top-k 4 (MoE), --max-seq 4096
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from bihaba_serve.daemon import EngineDaemon  # noqa: E402
from bihaba_serve.detector import detect_model  # noqa: E402

TURNS_ID = [
    "Siapa kamu? Jawab singkat dalam satu kalimat.",
    "Sebutkan 3 buah tropis, satu per baris.",
    "Dari daftar itu, mana yang paling manis? Jawab satu kalimat.",
]
TURNS_EN = [
    "Who are you? Answer in one short sentence.",
    "Name 3 tropical fruits, one per line.",
    "From that list, which one is the sweetest? Answer in one sentence.",
]


def run(info, session: bool, top_k: int, max_seq: int):
    turns = TURNS_EN if "mnn" in info.name.lower() else TURNS_ID
    d = EngineDaemon(info, REPO, session_cache=session, top_k_moe=top_k, max_seq=max_seq, default_temp=0.0)
    msgs = [{"role": "system", "content": "You are a concise, accurate assistant." if turns is TURNS_EN
             else "Kamu asisten yang ringkas dan akurat."}]
    out = []
    try:
        for t in turns:
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
            out.append({"q": t, "a": answer.strip(), "reused": st.get("reused"), "prefill_ms": st.get("prefill_ms"),
                        "tps": st.get("tps")})
    finally:
        d.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--with-session", action="store_true")
    ap.add_argument("--save", help="simpan jawaban sebagai referensi (jalankan di versi stabil)")
    ap.add_argument("--compare", help="bandingkan dengan referensi dari versi stabil")
    a = ap.parse_args()

    ref = json.loads(Path(a.compare).read_text()) if a.compare else {}
    saved, fails = {}, 0
    for mp in a.models:
        info = detect_model(mp)
        print(f"\n{'=' * 78}\n■ {info.name}  (engine={info.engine}, template={info.template})")
        res = run(info, False, a.top_k, a.max_seq)
        saved[info.name] = res
        sess = run(info, True, a.top_k, a.max_seq) if a.with_session else None
        for i, r in enumerate(res):
            print(f"\n  Giliran {i + 1}: {r['q']}")
            print(f"    kanonik  reuse={r['reused']!s:>5} prefill={r['prefill_ms'] or 0:7.1f} ms {r['tps'] or 0:6.1f} tok/s | {r['a'][:240]!r}")
            if info.name in ref and i < len(ref[info.name]):
                same = ref[info.name][i]["a"] == r["a"]
                fails += 0 if same else 1
                print(f"    vs stabil: {'IDENTIK ✓' if same else 'BERBEDA ✗  stabil: ' + repr(ref[info.name][i]['a'][:240])}")
            if sess:
                s = sess[i]
                print(f"    sesi     reuse={s['reused']!s:>5} prefill={s['prefill_ms'] or 0:7.1f} ms | {s['a'][:240]!r}"
                      f"  → {'IDENTIK' if s['a'] == r['a'] else 'BERBEDA'}")
    if a.save:
        Path(a.save).write_text(json.dumps(saved, ensure_ascii=False, indent=1))
        print(f"\n✓ Referensi disimpan ke {a.save}")
    if a.compare:
        print(f"\n{'✓ LULUS: semua jawaban identik dengan versi stabil' if fails == 0 else f'✗ GAGAL: {fails} jawaban berbeda dari versi stabil'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
