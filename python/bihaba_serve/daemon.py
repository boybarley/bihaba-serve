"""
Subprocess NDJSON Engine Daemon + Multi-Model Hot-Swap Pool (~270ms swap on M1 Max).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .detector import ModelInfo, detect_model, scan_models_dir
from .templates import _flatten_content, format_messages
from .tokenizer import ResilientTokenizer

_THINK_RE = re.compile(r"^.*?</think>", re.S)


def _norm_assistant(text: str) -> str:
    """Bentuk pembanding jawaban: klien biasanya membuang blok <think> dan spasi tepi."""
    return _THINK_RE.sub("", text, count=1).strip()


def _msg_key(messages: list[dict]) -> list[tuple[str, str]]:
    out = []
    for m in messages:
        role = m.get("role", "user")
        text = _flatten_content(m.get("content", ""))
        out.append((role, _norm_assistant(text) if role == "assistant" else text))
    return out


class StreamDecoder:
    """Detokenizer inkremental: O(panjang token baru), bukan decode ulang seluruh jawaban tiap token.
    Menahan potongan UTF-8 yang belum lengkap (karakter pengganti di ujung)."""

    def __init__(self, tok):
        self.tok, self.ids, self.prefix, self.read = tok, [], 0, 0

    def push(self, tid: int) -> str:
        self.ids.append(tid)
        prev = self.tok.decode(self.ids[self.prefix:self.read])
        cur = self.tok.decode(self.ids[self.prefix:])
        if len(cur) > len(prev) and not cur.endswith("\ufffd"):
            self.prefix, self.read = self.read, len(self.ids)
            return cur[len(prev):]
        return ""


class StopFilter:
    """Menahan ekor teks yang bisa menjadi awal stop sequence, agar potongan stop tidak terlanjur terkirim."""

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.pending = ""

    def feed(self, delta: str) -> tuple[str, bool]:
        """→ (teks aman untuk dikirim, berhenti?)"""
        if not self.stops:
            return delta, False
        self.pending += delta
        hits = [i for i in (self.pending.find(s) for s in self.stops) if i != -1]
        if hits:
            out, self.pending = self.pending[:min(hits)], ""
            return out, True
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.pending)), 0, -1):
                if self.pending.endswith(s[:k]):
                    hold = max(hold, k)
                    break
        out = self.pending[:len(self.pending) - hold]
        self.pending = self.pending[len(self.pending) - hold:]
        return out, False

    def flush(self) -> str:
        out, self.pending = self.pending, ""
        return out


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
        max_seq: int = 16384,
        load_state: str | None = None,
        default_temp: float = 0.0,
        prefill_chunk: int = 128,
        session_cache: bool = True,
    ):
        self.info = model_info
        self.session_cache = session_cache
        self._sess: dict | None = None
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
        self.can_cancel = bool(meta.get("cancel", False))   # engine lama tanpa handler SIGUSR1 → jangan kirim sinyal
        self._im_end = self.tok.token_id("<|im_end|>")

    def _cancel(self) -> None:
        """Hentikan generasi engine sekarang (stop sequence cocok / klien putus), bukan menunggu max_tokens."""
        if self.can_cancel and self.proc.poll() is None:
            try:
                os.kill(self.proc.pid, signal.SIGUSR1)
            except OSError:
                pass

    def close(self) -> None:
        self._sess = None
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

    def clear_state(self) -> dict:
        """Hapus snapshot system prompt di RAM GPU (hasil bake/auto-bake) + cache percakapan."""
        with self.lock:
            self.proc.stdin.write(json.dumps({"cmd": "clear_state"}) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                return {"cleared": False, "error": "engine berhenti"}
            res = json.loads(line)
            if res.get("cleared"):
                self.baked_toks = 0
                self._sess = None
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
        # Sesuaikan max_tokens dengan sisa konteks (seperti API OpenAI): percakapan panjang tidak gagal hanya
        # karena max_tokens default terlalu besar. Gagal hanya bila prompt sendiri tidak muat.
        room = int(getattr(self, "max_seq", 1 << 30)) - len(prompt_ids) - 9
        if room < 1:
            yield {"error": f"Percakapan ({len(prompt_ids)} token) melebihi konteks maksimal engine ({self.max_seq} token). "
                            "Mulai percakapan baru, atau jalankan server dengan --max-seq lebih besar (mis. 32768)."}
            return
        max_tokens = max(1, min(int(max_tokens), room))
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
            raw_ids: list[int] = []
            dec = StreamDecoder(self.tok)
            stopper = StopFilter(stop_seqs)
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
                        out, hit = stopper.feed(dec.push(tid))
                        if out:
                            yield {"delta": out}
                        if hit:
                            stopped_early = True
                            self._cancel()
                    elif obj.get("done"):
                        finished = True
                        if obj.get("error"):
                            yield {"error": obj["error"]}
                            break
                        if not stopped_early:
                            tail = stopper.flush()
                            if tail:
                                yield {"delta": tail}
                        self.baked_toks = int(obj.get("baked_toks", self.baked_toks))
                        obj["prompt_tokens"] = len(prompt_ids)
                        obj["finish_reason"] = "stop" if stopped_early or (raw_ids and raw_ids[-1] in self.tok.eos_ids) \
                            else ("length" if obj.get("tokens", 0) + 1 >= max_tokens else "stop")
                        yield {"done": True, "stats": obj, "raw_ids": raw_ids, "stopped_early": stopped_early}
                        break
            finally:
                # Klien putus / generator ditutup sebelum engine selesai: buang sisa output sampai baris "done".
                # Tanpa ini, token sisa terbaca oleh request BERIKUTNYA (respons tercampur).
                if not finished:
                    self._cancel()
                while not finished:
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    try:
                        if json.loads(line).get("done"):
                            break
                    except ValueError:
                        break

    def _session_prompt(self, messages: list[dict], key: list[tuple[str, str]]):
        """Giliran lanjutan dari percakapan yang sama → id token PERSIS giliran lalu + ekor baru.

        Kenapa: state engine berisi token yang benar-benar diproses (termasuk blok <think></think> kosong
        dari template qwen35 dan tokenisasi jawaban apa adanya). Merender ulang riwayat sebagai teks
        menghasilkan token berbeda di titik itu → cache live engine selalu gagal → seluruh riwayat
        di-prefill ulang tiap giliran. Dengan id persis, engine hanya memproses pesan baru."""
        s = self._sess
        if not (s and len(key) > len(s["key"]) and key[:len(s["key"])] == s["key"]):
            return None
        _, tail = format_messages(messages[len(s["key"]):], template=self.tok.template,
                                  enable_thinking=self.enable_thinking)
        return s["ids"] + self.tok.encode_raw("\n" + tail), s["prefix_len"]

    def stream_tokens(self, messages: list[dict], **kwargs):
        chatml = self.tok.template != "llama3" and self._im_end is not None
        use_sess = self.session_cache and chatml and not self.enable_thinking
        key = _msg_key(messages) if use_sess else []
        hit = self._session_prompt(messages, key) if use_sess else None
        if hit:
            prompt_ids, prefix_len = hit
        else:
            prompt_ids, prefix_len = self.tok.format_chat_with_prefix(messages, enable_thinking=self.enable_thinking)
        self._sess = None
        text: list[str] = []
        for ev in self.stream_prompt_ids(prompt_ids, prefix_len=prefix_len, **kwargs):
            if "delta" in ev:
                text.append(ev["delta"])
            elif ev.get("done") and use_sess and not ev.get("stopped_early"):
                raw = list(ev.get("raw_ids") or [])
                if raw and raw[-1] in self.tok.eos_ids:
                    raw = raw[:-1]
                # Engine menyimpan prompt + raw[:-1] (token terakhir belum jadi input). Giliran berikut diawali
                # <|im_end|> → id sesi = prompt + raw + <|im_end|>, selalu memuat state engine sebagai prefix.
                self._sess = {"key": key + [("assistant", _norm_assistant("".join(text)))],
                              "ids": prompt_ids + raw + [self._im_end], "prefix_len": prefix_len}
            if "stats" in ev:
                ev["stats"]["session_hit"] = bool(hit)
            yield ev


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
