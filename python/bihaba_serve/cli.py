#!/usr/bin/env python3
"""
bihaba-serve CLI — Unified Entrypoint for serve | chat | bench | align | bake | list
Created by Boy Barley from Bihaba (Team Pasai Aceh)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .align import align_model_dir
from .daemon import EngineDaemon, ModelPool, default_prefill_chunk, find_engine_binary
from .detector import detect_model, scan_models_dir
from .server import run_server


def _repo_root() -> Path:
    env = os.environ.get("BIHABA_SERVE_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _run_chat(daemon: EngineDaemon, max_tokens: int, system_text: str | None) -> None:
    print(
        f"══ bihaba-serve v{__version__} Chat [{daemon.model_name} | topK={daemon.active_top_k} | "
        f"temp={daemon.default_temp} | spec=k{daemon.max_spec}] ('/clear' or 'exit') ══"
    )
    transcript_ids: list[int] = []
    prefix_len = 0
    if system_text:
        sys_ids = daemon.tok.encode_raw(f"<|im_start|>system\n{system_text}<|im_end|>\n")
        transcript_ids.extend(sys_ids)
        prefix_len = len(sys_ids)
        print(f"  [system prompt: {prefix_len} toks — auto-cached in GPU snapshot]")

    im_end_nl = daemon.tok.encode_raw("<|im_end|>\n")
    asst_hdr = daemon.tok.encode_raw(
        "<|im_start|>assistant\n<think>\n" if daemon.enable_thinking
        else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    while True:
        try:
            user_msg = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_msg or user_msg.lower() in ("exit", "quit"):
            break
        if user_msg.lower() == "/clear":
            transcript_ids = transcript_ids[:prefix_len]
            print("  [conversation cleared — system prompt snapshot retained]")
            continue

        transcript_ids.extend(daemon.tok.encode_raw(f"<|im_start|>user\n{user_msg}<|im_end|>\n"))
        transcript_ids.extend(asst_hdr)

        print("AI > ", end="", flush=True)
        stats, raw_ids, err = {}, [], None
        turn_start = len(transcript_ids) - len(asst_hdr) - len(daemon.tok.encode_raw(f"<|im_start|>user\n{user_msg}<|im_end|>\n"))
        for ev in daemon.stream_prompt_ids(transcript_ids, prefix_len=prefix_len, max_tokens=max_tokens):
            if "delta" in ev:
                print(ev["delta"], end="", flush=True)
            elif "error" in ev:
                err = ev["error"]
            elif ev.get("done"):
                stats, raw_ids = ev["stats"], ev["raw_ids"]
        if err:
            transcript_ids = transcript_ids[:max(prefix_len, turn_start)]
            print(f"\n   ✗ {err} — ketik /clear untuk memulai percakapan baru.")
            continue

        transcript_ids.extend([t for t in raw_ids if t not in daemon.tok.eos_ids])
        transcript_ids.extend(im_end_nl)
        print(
            f"\n   ⚡ {stats.get('tps', 0):.1f} tok/s ({stats.get('tokens', 0)} toks in {stats.get('ms', 0):.0f} ms | "
            f"{stats.get('tok_per_step', 1.0):.2f} tok/step | prefill {stats.get('prefill_ms', 0):.0f} ms, reused={stats.get('reused', 0)})"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="bihaba-serve",
        description="bihaba-serve — Zero-Dependency Native Apple Metal LLM Engine (Boy Barley — Bihaba / Team Pasai Aceh)",
    )
    ap.add_argument("--version", action="version", version=f"bihaba-serve {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_srv = sub.add_parser("serve", help="Start OpenAI-compatible HTTP server (0 web dependencies)")
    p_srv.add_argument("--model", default=None, help="Path to local model folder (~/models/<dir>)")
    p_srv.add_argument("--models-dir", default=None, help="Scan directory for multi-model hot-swap (/v1/models)")
    p_srv.add_argument("--top-k", type=int, default=0, help="Override MoE active experts (e.g. 4)")
    p_srv.add_argument("--spec", type=int, default=7, help="Max speculative draft tokens (0..7)")
    p_srv.add_argument("--max-seq", type=int, default=16384)
    p_srv.add_argument("--temp", type=float, default=0.6)
    p_srv.add_argument("--rep-penalty", type=float, default=1.0,
                       help="1.0 = mati (default). >1 merusak kode/JSON & menurunkan penerimaan spekulasi")
    p_srv.add_argument("--prefill-chunk", type=int, default=None, help="Token per langkah prefill (qwen35 1..512, transformer 1..128; 8 = perilaku lama)")
    p_srv.add_argument("--thinking", action="store_true")
    p_srv.add_argument("--system-file", default=None, help="Pre-bake system prompt into GPU RAM on startup")
    p_srv.add_argument("--load-state", default=None, help="Preload baked .tstate snapshot on startup (0ms TTFT)")
    p_srv.add_argument("--host", default="127.0.0.1",
                       help="Default hanya lokal. Pakai 0.0.0.0 untuk membuka ke jaringan (wajib --api-key)")
    p_srv.add_argument("--port", type=int, default=8090)
    p_srv.add_argument("--api-key", default=os.environ.get("BIHABA_API_KEY"),
                       help="Wajibkan 'Authorization: Bearer <key>' (atau env BIHABA_API_KEY)")
    p_srv.add_argument("--cors", default=None, help="Aktifkan CORS untuk origin ini, mis. '*' atau 'http://localhost:3000'")
    p_srv.add_argument("--state-dir", default=None, help="Folder tempat POST /v1/state/bake boleh menyimpan .tstate")
    p_srv.add_argument("--session-cache", action="store_true",
                       help="Paksa cache sesi AKTIF untuk semua model (default: aktif untuk Qwen3.5/MoE, mati untuk model transformer)")
    p_srv.add_argument("--no-session-cache", action="store_true",
                       help="Paksa cache sesi MATI untuk semua model (riwayat selalu dirender sesuai template kanonik)")

    p_chat = sub.add_parser("chat", help="Interactive terminal chat")
    p_chat.add_argument("--model", required=True)
    p_chat.add_argument("--top-k", type=int, default=0)
    p_chat.add_argument("--spec", type=int, default=7)
    p_chat.add_argument("--max-seq", type=int, default=16384)
    p_chat.add_argument("--temp", type=float, default=0.6)
    p_chat.add_argument("--rep-penalty", type=float, default=1.0)
    p_chat.add_argument("--prefill-chunk", type=int, default=None)
    p_chat.add_argument("--thinking", action="store_true")
    p_chat.add_argument("--max-tokens", type=int, default=512)
    p_chat.add_argument("--system-file", default=None)
    p_chat.add_argument("--load-state", default=None)

    p_b = sub.add_parser("bench", help="Run native Metal GPU X-Ray & speculative benchmark")
    p_b.add_argument("--model", required=True)
    p_b.add_argument("--top-k", type=int, default=0)
    p_b.add_argument("--spec", type=int, default=7)
    p_b.add_argument("--tokens", type=int, default=128)
    p_b.add_argument("--profile", action="store_true")
    p_b.add_argument("--prefill-chunk", type=int, default=None)

    p_bk = sub.add_parser("bake", help="Bake a system prompt / RAG prefix into a persistent .tstate file")
    p_bk.add_argument("--model", required=True)
    p_bk.add_argument("--system-file", required=True)
    p_bk.add_argument("--out", required=True)
    p_bk.add_argument("--top-k", type=int, default=0)

    p_al = sub.add_parser("align", help="64-byte align .safetensors headers in-place for ~250ms zero-copy boot")
    p_al.add_argument("dirs", nargs="+")

    p_ls = sub.add_parser("list", help="List compatible models in a directory")
    p_ls.add_argument("dir", nargs="?", default=str(Path.home() / "models"))

    args = ap.parse_args()
    repo = _repo_root()

    if args.cmd == "align":
        for d in args.dirs:
            align_model_dir(d)
        return

    if args.cmd == "list":
        models = scan_models_dir(args.dir)
        if not models:
            print(f"No compatible models found in {args.dir}")
            return
        print(f"Compatible models in {Path(args.dir).expanduser().resolve()}:")
        for m in models.values():
            moe_tag = "MoE" if m.is_moe else "Dense"
            print(f"  • {m.name:<42} [{m.engine:<11} | {m.model_type} | {m.num_layers}L dim={m.hidden_size} {moe_tag}]")
        return

    if args.cmd == "bench":
        info = detect_model(args.model)
        bin_path = find_engine_binary(info.engine, repo)
        cmd = [str(bin_path), str(info.path), "--bench", str(args.tokens), "--spec", str(args.spec),
               "--prefill-chunk", str(args.prefill_chunk if args.prefill_chunk is not None else default_prefill_chunk(info))]
        if args.profile:
            cmd.append("--profile")
        if args.top_k > 0:
            cmd += ["--top-k", str(args.top_k)]
        sys.exit(subprocess.call(cmd))

    if args.cmd == "bake":
        info = detect_model(args.model)
        sys_text = Path(args.system_file).expanduser().read_text()
        daemon = EngineDaemon(info, repo, top_k_moe=args.top_k)
        try:
            res = daemon.bake_system_prompt(sys_text, args.out)
            if not res.get("baked"):
                sys.exit(f"✗ Bake gagal: {res.get('error', res)}")
            print(f"✓ Baked state saved to {args.out}: {res}")
        finally:
            daemon.close()
        return

    if args.cmd == "chat":
        info = detect_model(args.model)
        sys_text = Path(args.system_file).expanduser().read_text() if args.system_file else None
        daemon = EngineDaemon(
            info, repo, rep_penalty=args.rep_penalty, enable_thinking=args.thinking,
            top_k_moe=args.top_k, max_spec=args.spec, max_seq=args.max_seq,
            load_state=args.load_state, default_temp=args.temp, prefill_chunk=args.prefill_chunk,
        )
        try:
            _run_chat(daemon, args.max_tokens, sys_text)
        finally:
            daemon.close()
        return

    if args.cmd == "serve":
        sys_text = Path(args.system_file).expanduser().read_text() if args.system_file else None
        pool = ModelPool(
            repo, default_model=args.model, models_dir=args.models_dir,
            rep_penalty=args.rep_penalty, enable_thinking=args.thinking,
            top_k_moe=args.top_k, max_spec=args.spec, max_seq=args.max_seq,
            load_state=args.load_state, default_temp=args.temp, prefill_chunk=args.prefill_chunk,
            session_cache=(False if args.no_session_cache else (True if args.session_cache else None)),
        )
        if sys_text and not args.load_state and pool.active:
            res = pool.active.bake_system_prompt(sys_text, None)
            print(f"✓ Pre-baked --system-file into GPU RAM snapshot: {res}")
        try:
            run_server(pool, args.host, args.port, api_key=args.api_key, cors_origin=args.cors, state_dir=args.state_dir)
        except KeyboardInterrupt:
            print("\nShutting down bihaba-serve...")
        finally:
            if pool.active:
                pool.active.close()


if __name__ == "__main__":
    main()
