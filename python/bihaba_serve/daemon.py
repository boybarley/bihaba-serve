"""
Subprocess NDJSON Engine Daemon + Multi-Model Hot-Swap Pool (~270ms swap on M1 Max).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .detector import ModelInfo, detect_model, scan_models_dir
from .tokenizer import ResilientTokenizer


def find_engine_binary(engine_name: str, repo_root: Path) -> Path:
    candidates = [
        repo_root / "build" / f"bihaba-{engine_name}",
        Path.home() / ".local" / "share" / "bihaba-serve" / f"bihaba-{engine_name}",
        Path("/usr/local/share/bihaba-serve") / f"bihaba-{engine_name}",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        f"Engine binary `bihaba-{engine_name}` not found. Run `./install.sh` or `make {engine_name}` first."
    )


class EngineDaemon:
    def __init__(
        self,
        model_info: ModelInfo,
        repo_root: Path,
        rep_penalty: float = 1.0,
        enable_thinking: bool = False,
        top_k_moe: int = 0,
        max_spec: int = 7,
        max_seq: int = 4096,
        load_state: str | None = None,
        default_temp: float = 0.0,
        prefill_chunk: int = 128,
    ):
        self.info = model_info
        self.enable_thinking = enable_thinking
        self.default_temp = default_temp
        self.default_rep_pen = rep_penalty
        self.lock = threading.Lock()

        bin_path = find_engine_binary(model_info.engine, repo_root)
        cmd = [
            str(bin_path),
            str(model_info.path),
            "--daemon",
            "--rep-penalty", str(rep_penalty),
            "--spec", str(max_spec),
            "--max-seq", str(max_seq),
            "--prefill-chunk", str(prefill_chunk),
        ]
        if top_k_moe > 0:
            cmd += ["--top-k", str(top_k_moe)]
        if load_state:
            cmd += ["--state", str(Path(load_state).expanduser().resolve())]

        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready_line = self.proc.stdout.readline()
        if not ready_line or '"ready":true' not in ready_line:
            self.close()
            raise RuntimeError(f"Engine `{bin_path.name}` failed to start: {ready_line!r}")

        meta = json.loads(ready_line)
        tdir = meta.get("model_dir") or str(model_info.path)
        self.tok = ResilientTokenizer(tdir, template=model_info.template)
        self.model_name = model_info.name
        self.arch = meta.get("arch", model_info.model_type)
        self.active_top_k = meta.get("top_k", top_k_moe)
        self.max_spec = meta.get("max_spec", max_spec)
        self.baked_toks = meta.get("baked_toks", 0)
        self.max_seq = int(meta.get("max_seq", max_seq))

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()

    def bake_system_prompt(self, system_text: str, out_path: str | None = None) -> dict:
        prompt_ids = self.tok.encode_raw(f"<|im_start|>system\n{system_text}<|im_end|>\n")
        payload: dict = {"cmd": "bake", "prompt_ids": prompt_ids}
        if out_path:
            payload["path"] = str(Path(out_path).expanduser().resolve())
        with self.lock:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                return {"baked": False, "error": "engine berhenti"}
            res = json.loads(line)
            if res.get("baked"):
                self.baked_toks = int(res.get("tokens", 0))
            return res

    def stream_prompt_ids(
        self,
        prompt_ids: list[int],
        prefix_len: int = 0,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        rep_penalty: float | None = None,
        seed: int | None = None,
        stop: list[str] | None = None,
    ):
        temp = self.default_temp if temperature is None else float(temperature)
        rp = self.default_rep_pen if rep_penalty is None else float(rep_penalty)
        payload: dict = {
            "prompt_ids": prompt_ids,
            "prefix_len": prefix_len,
            "max_tokens": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "rep_penalty": rp,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        stop_seqs = [s for s in (stop or []) if s]

        with self.lock:
            if self.proc.poll() is not None:
                yield {"error": "engine tidak berjalan"}
                return
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            acc_ids: list[int] = []
            raw_ids: list[int] = []
            emitted_text = ""
            stopped_early = False
            finished = False
            try:
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        finished = True
                        yield {"error": "engine berhenti di tengah generasi"}
                        break
                    obj = json.loads(line)
                    if "tok" in obj:
                        tid = int(obj["tok"])
                        raw_ids.append(tid)
                        if stopped_early or tid in self.tok.eos_ids:
                            continue
                        acc_ids.append(tid)
                        full = self.tok.decode(acc_ids)
                        if stop_seqs:
                            for sseq in stop_seqs:
                                idx = full.find(sseq)
                                if idx != -1:
                                    delta = full[len(emitted_text):idx]
                                    emitted_text = full[:idx]
                                    if delta:
                                        yield {"delta": delta}
                                    stopped_early = True
                                    break
                            if stopped_early:
                                continue
                        delta = full[len(emitted_text):]
                        emitted_text = full
                        if delta:
                            yield {"delta": delta}
                    elif obj.get("done"):
                        finished = True
                        if obj.get("error"):
                            yield {"error": obj["error"]}
                            break
                        self.baked_toks = int(obj.get("baked_toks", self.baked_toks))
                        obj["prompt_tokens"] = len(prompt_ids)
                        obj["finish_reason"] = "stop" if stopped_early or (raw_ids and raw_ids[-1] in self.tok.eos_ids) \
                            else ("length" if obj.get("tokens", 0) + 1 >= max_tokens else "stop")
                        yield {"done": True, "stats": obj, "raw_ids": raw_ids}
                        break
            finally:
                # Klien putus / generator ditutup sebelum engine selesai: buang sisa output sampai baris "done".
                # Tanpa ini, token sisa terbaca oleh request BERIKUTNYA (respons tercampur).
                while not finished:
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    try:
                        if json.loads(line).get("done"):
                            break
                    except ValueError:
                        break

    def stream_tokens(self, messages: list[dict], **kwargs):
        prompt_ids, prefix_len = self.tok.format_chat_with_prefix(messages, enable_thinking=self.enable_thinking)
        yield from self.stream_prompt_ids(prompt_ids, prefix_len=prefix_len, **kwargs)


class ModelPool:
    """Manages active EngineDaemon + supports instant hot-swap when --models-dir is enabled."""

    def __init__(
        self,
        repo_root: Path,
        default_model: str | None = None,
        models_dir: str | None = None,
        **daemon_kwargs,
    ):
        self.repo_root = repo_root
        self.daemon_kwargs = daemon_kwargs
        self.pool_lock = threading.Lock()
        # Satu engine = satu GPU stream. Lock ini dipegang selama swap DAN selama satu request penuh, sehingga
        # hot-swap tidak bisa menutup daemon yang sedang dipakai request lain (race di versi lama).
        self.gen_lock = threading.RLock()
        self.catalog: dict[str, ModelInfo] = {}
        self.active: EngineDaemon | None = None
        self.total_requests = 0
        self.started_at = time.time()

        if models_dir:
            self.catalog.update(scan_models_dir(models_dir))
        if default_model:
            info = detect_model(default_model)
            self.catalog[info.name] = info
            self.active = EngineDaemon(info, repo_root, **daemon_kwargs)
        elif self.catalog:
            first_info = next(iter(self.catalog.values()))
            self.active = EngineDaemon(first_info, repo_root, **daemon_kwargs)
        else:
            raise ValueError("Provide either --model <path> or --models-dir <dir> with compatible models.")

    @contextmanager
    def session(self, requested_model: str | None = None):
        """Pakai: `with pool.session(model) as daemon: ...` — aman terhadap hot-swap bersamaan."""
        with self.gen_lock:
            yield self.get_daemon(requested_model)

    def get_daemon(self, requested_model: str | None = None) -> EngineDaemon:
        with self.gen_lock, self.pool_lock:
            self.total_requests += 1
            if not requested_model or (self.active and requested_model == self.active.model_name):
                return self.active
            if requested_model in self.catalog:
                print(f"▸ Hot-swapping model -> {requested_model} ...")
                if self.active:
                    self.active.close()
                kw = dict(self.daemon_kwargs)
                kw["load_state"] = None
                self.active = EngineDaemon(self.catalog[requested_model], self.repo_root, **kw)
                return self.active
            return self.active
