#!/usr/bin/env python3
"""
SATU perintah untuk menguji SEMUA kelas model — pengganti "test sana sini".

  python3 scripts/uji_semua.py --simpan-acuan   # sekali, di versi yang sudah dinilai baik (mis. v1.6.1)
  python3 scripts/uji_semua.py                  # setiap build baru: bandingkan dengan acuan

Yang diuji per kelas model (transformer kecil, transformer menengah, qwen35 dense, qwen35 MoE, qwen35 dense besar):
  1. tes offline                       2. bench: Prefill check & =greedy (kebenaran kernel)
  3. jawaban chat 3 giliran vs acuan    4. angka kecepatan vs acuan (turun > 5% = GAGAL)
Hasil akhir: satu tabel + LULUS/GAGAL. Folder repo = folder tempat skrip ini berada (stabil atau eksperimen).
Ubah daftar MODEL di bawah bila nama folder model berbeda.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()
ACUAN = HOME / "bihaba-acuan"

MODEL = [  # (kelas, folder model, argumen tambahan)
    ("transformer kecil",    HOME / "models/MNN-Pocket-0.3B-MLX-4bit", []),
    ("transformer menengah", HOME / "models/Bihaba-AI-1.7B-Final",     []),
    ("qwen35 dense",         HOME / "models/Qwythos-9B-Claude",        []),
    ("qwen35 MoE",           HOME / "models/Bihaba-bagus-pakai",       ["--top-k", "4"]),
    ("qwen35 dense besar",   HOME / "models/qwen3.8-27b",              []),   # Qwen3.8-27B (sejak 1.7.0.dev18)
]

ENV = dict(os.environ, BIHABA_SERVE_HOME=str(REPO), BIHABA_SHADER_DIR=str(REPO / "build"),
           PYTHONPATH=str(REPO / "python"))

ANGKA = {   # nama metrik → regex dari output bench (semakin besar semakin baik)
    "decode":   r"Live Decode Bench \(Greedy M=1\):.*=> ([\d.]+) tok/s",
    # chunk aktif = baris Prefill Bench terakhir (128 / 512 per kelas, 1.7.0.dev15)
    "prefill":  r"Prefill Bench \(512 tok, chunk *(?:128|256|512)\):.*=> ([\d.]+) tok/s",
    "spek":     r"Live Speculative Bench.*=> ([\d.]+) tok/s",
}


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV, **kw)


def bench(model: Path, extra: list[str]) -> tuple[dict, str]:
    r = sh([sys.executable, "-m", "bihaba_serve.cli", "bench", "--model", str(model), "--tokens", "128", *extra])
    out = r.stdout + r.stderr
    res = {"error": bool(re.search(r"error|rror:", out)) and "Ready" not in out}
    m = re.search(r"Prefill check.*?(\d+)/(\d+)", out)
    res["prefill_ok"] = bool(m and m.group(1) == m.group(2))
    m = re.search(r"=greedy (\d+)/(\d+)", out)
    res["greedy_ok"] = bool(m and m.group(1) == m.group(2))
    for k, rx in ANGKA.items():
        ms = re.findall(rx, out)
        res[k] = float(ms[-1]) if ms else None
    return res, out


def chat(model: Path, extra: list[str], mode: str, ref: Path) -> bool | None:
    flag = ["--save", str(ref)] if mode == "simpan" else ["--compare", str(ref)]
    if mode == "banding" and not ref.exists():
        return None
    r = sh([sys.executable, str(REPO / "scripts/regress_chat.py"), *flag, *extra, str(model)])
    return True if mode == "simpan" else ("LULUS" in r.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simpan-acuan", action="store_true")
    ap.add_argument("--hanya", default="", help="uji/simpan hanya kelas yang namanya memuat teks ini, mis. 'besar'")
    ap.add_argument("--repo", default="", help="uji build repo lain (mis. ~/bihaba-serve = versi stabil) dengan skrip ini")
    a = ap.parse_args()
    global REPO, ENV
    if a.repo:
        REPO = Path(a.repo).expanduser().resolve()
        ENV = dict(os.environ, BIHABA_SERVE_HOME=str(REPO), BIHABA_SHADER_DIR=str(REPO / "build"),
                   PYTHONPATH=str(REPO / "python"))
    mode = "simpan" if a.simpan_acuan else "banding"
    ACUAN.mkdir(exist_ok=True)

    print(f"Repo: {REPO}  |  mode: {'SIMPAN ACUAN' if mode == 'simpan' else 'BANDINGKAN DENGAN ACUAN'}")
    r = sh([sys.executable, "-m", "unittest", "discover", "-s", str(REPO / "tests")])
    unit_ok = r.returncode == 0
    print(f"Tes offline: {'OK' if unit_ok else 'GAGAL'}")

    angka_acuan = json.loads((ACUAN / "angka.json").read_text()) if (ACUAN / "angka.json").exists() else {}
    angka_baru, baris, gagal = {}, [], 0 if unit_ok else 1
    for kelas, model, extra in MODEL:
        if a.hanya and a.hanya not in kelas:
            continue
        if not model.exists():
            baris.append((kelas, "-", "-", "-", f"folder tidak ada: {model}", "GAGAL"))
            gagal += 1
            continue
        print(f"… {kelas}: {model.name}", flush=True)
        b, raw = bench(model, extra)
        (ACUAN / f"log_{model.name}.txt").write_text(raw)
        angka_baru[model.name] = {k: b[k] for k in ANGKA}
        c = chat(model, extra, mode, ACUAN / f"jawaban_{model.name}.json")
        turun = []
        for k in ANGKA:
            lama, baru = angka_acuan.get(model.name, {}).get(k), b[k]
            if lama and baru and baru < lama * 0.95:
                turun.append(f"{k} {lama:.0f}→{baru:.0f}")
            elif lama and not baru:
                turun.append(f"{k} hilang dari output")   # angka tak terbaca = GAGAL, bukan lolos diam-diam
        ok = b["prefill_ok"] and b["greedy_ok"] and not b["error"] and c is not False and not turun
        gagal += 0 if ok else 1
        baris.append((kelas, "✓" if b["prefill_ok"] else "✗", "✓" if b["greedy_ok"] else "✗",
                      {True: "identik ✓", False: "BERBEDA ✗", None: "(belum ada acuan)"}[c] if mode == "banding" else "disimpan",
                      ", ".join(turun) or " / ".join(f"{b[k]:.0f}" if b[k] else "-" for k in ANGKA),
                      "LULUS" if ok else "GAGAL"))

    if mode == "simpan":   # gabung: menyimpan satu kelas tidak menghapus acuan kelas lain
        (ACUAN / "angka.json").write_text(json.dumps({**angka_acuan, **angka_baru}, indent=1))
    print("\n{:<22} {:>9} {:>8} {:>18}  {:<34} {}".format("Kelas", "Prefill", "Greedy", "Jawaban", "decode/prefill/spek (tok/s)", "Hasil"))
    for row in baris:
        print("{:<22} {:>9} {:>8} {:>18}  {:<34} {}".format(*row))
    print(f"\nLog lengkap: {ACUAN}/log_*.txt")
    if mode == "simpan":
        print("✓ Acuan disimpan. Build berikutnya cukup: python3 scripts/uji_semua.py")
        return 0
    print("✓ LULUS SEMUA — build boleh naik ke stabil" if gagal == 0 else f"✗ GAGAL ({gagal} baris) — build DITOLAK, versi stabil tidak diubah")
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
