"""
Zero-Dependency OpenAI-Compatible HTTP Server (Python Stdlib Only).
Endpoints: /v1/chat/completions | /v1/models | /v1/state/bake | /health

Keamanan (default aman untuk laptop di jaringan publik):
  • bind 127.0.0.1 kecuali --host diubah; --api-key wajib bila dibuka ke jaringan;
  • CORS mati kecuali --cors '<origin>' (mis. '*' untuk web app lokal);
  • /v1/state/bake hanya menulis file di dalam --state-dir, dengan nama file yang disanitasi.
"""
from __future__ import annotations

import hmac
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .daemon import ModelPool

MAX_BODY_BYTES = 16 * 1024 * 1024
WEB_UI = Path(__file__).resolve().parent / "web" / "index.html"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _state_file(state_dir: str | None, name: str | None) -> tuple[str | None, str | None]:
    """→ (path, error). Tanpa nama: bake ke RAM saja. Dengan nama: hanya di dalam state_dir."""
    if not name:
        return None, None
    if not state_dir:
        return None, "menyimpan .tstate lewat HTTP dinonaktifkan (jalankan server dengan --state-dir <folder>)"
    base = Path(str(name)).name
    if not base.endswith(".tstate"):
        base += ".tstate"
    if not _SAFE_NAME.match(base) or base.startswith("."):
        return None, "nama file hanya boleh huruf, angka, '.', '_' dan '-'"
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return str(root / base), None


def run_server(pool: ModelPool, host: str, port: int, api_key: str | None = None,
               cors_origin: str | None = None, state_dir: str | None = None) -> None:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send_cors(self):
            if cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")

        def _send_json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not api_key:
                return True
            auth = self.headers.get("Authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("X-API-Key", "")
            if token and hmac.compare_digest(token.encode(), api_key.encode()):
                return True
            self._send_json(401, {"error": {"message": "API key tidak valid", "type": "invalid_request_error"}})
            return False

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path in ("/", "/index.html", "/ui"):
                # UI dilayani dari origin yang sama → tidak butuh CORS. Halaman statis, tanpa data rahasia.
                try:
                    body = WEB_UI.read_bytes()
                except OSError:
                    self._send_json(404, {"error": {"message": "web UI tidak ditemukan"}})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/health":
                act = pool.active
                self._send_json(200, {
                    "status": "ok",
                    "engine": "bihaba-serve (Taalas-Metal)",
                    "version": __version__,
                    "author": "Boy Barley — Bihaba (Team Pasai Aceh)",
                    "active_model": act.model_name if act else None,
                    "arch": act.arch if act else None,
                    "top_k": act.active_top_k if act else 0,
                    "max_seq": act.max_seq if act else 0,
                    "baked_prefix_tokens": act.baked_toks if act else 0,
                    "total_requests": pool.total_requests,
                    "uptime_sec": round(time.time() - pool.started_at, 1),
                })
                return
            if not self._authorized():
                return
            if self.path == "/v1/models":
                data = [
                    {"id": name, "object": "model", "owned_by": "bihaba-serve",
                     "engine": info.engine, "arch": info.model_type}
                    for name, info in pool.catalog.items()
                ]
                self._send_json(200, {"object": "list", "data": data})
            else:
                self._send_json(404, {"error": {"message": "not found"}})

        def _read_json(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._send_json(413 if length > 0 else 400, {"error": {"message": "ukuran body tidak valid"}})
                return None
            try:
                req = json.loads(self.rfile.read(length)) if length > 0 else {}
            except ValueError:
                self._send_json(400, {"error": {"message": "body bukan JSON yang valid"}})
                return None
            if not isinstance(req, dict):
                self._send_json(400, {"error": {"message": "body harus objek JSON"}})
                return None
            return req

        def do_POST(self):
            if not self._authorized():
                return
            req = self._read_json()
            if req is None:
                return

            if self.path == "/v1/state/bake":
                path, err = _state_file(state_dir, req.get("path") or req.get("name"))
                if err:
                    self._send_json(403, {"error": {"message": err}})
                    return
                with pool.session(req.get("model")) as daemon:
                    res = daemon.bake_system_prompt(str(req.get("system", "")), path)
                self._send_json(200 if res.get("baked") else 400, res)
                return

            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "not found"}})
                return

            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_json(400, {"error": {"message": "'messages' wajib berupa list tidak kosong"}})
                return
            try:
                max_tokens = max(1, int(req.get("max_tokens") or req.get("max_completion_tokens") or 512))
                stop = req.get("stop")
                if isinstance(stop, str):
                    stop = [stop]
                gen_kw = dict(
                    max_tokens=max_tokens,
                    temperature=req.get("temperature"),
                    top_p=float(req.get("top_p", 0.95)),
                    top_k=int(req.get("top_k", 40)),
                    min_p=float(req.get("min_p", 0.05)),
                    rep_penalty=req.get("repetition_penalty"),
                    seed=req.get("seed"),
                    stop=stop,
                )
            except (TypeError, ValueError):
                self._send_json(400, {"error": {"message": "parameter sampling tidak valid"}})
                return
            stream = bool(req.get("stream", False))
            cid = f"chatcmpl-bihaba-{int(time.time() * 1000)}"
            created = int(time.time())

            with pool.session(req.get("model")) as daemon:
                if stream:
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    gen = daemon.stream_tokens(messages, **gen_kw)
                    try:
                        for ev in gen:
                            if "delta" in ev:
                                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                         "model": daemon.model_name,
                                         "choices": [{"index": 0, "delta": {"content": ev["delta"]}, "finish_reason": None}]}
                            elif "error" in ev:
                                chunk = {"error": {"message": ev["error"], "type": "invalid_request_error"}}
                            else:
                                st = ev["stats"]
                                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                                         "model": daemon.model_name,
                                         "choices": [{"index": 0, "delta": {}, "finish_reason": st.get("finish_reason", "stop")}],
                                         "usage": {"prompt_tokens": st.get("prompt_tokens", 0),
                                                   "completion_tokens": st.get("tokens", 0),
                                                   "total_tokens": st.get("prompt_tokens", 0) + st.get("tokens", 0)},
                                         "bihaba_stats": st}
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass          # klien putus: generator ditutup → daemon membuang sisa output engine
                    finally:
                        gen.close()
                else:
                    parts, st, err = [], {}, None
                    for ev in daemon.stream_tokens(messages, **gen_kw):
                        if "delta" in ev:
                            parts.append(ev["delta"])
                        elif "error" in ev:
                            err = ev["error"]
                        else:
                            st = ev["stats"]
                    if err:
                        self._send_json(400, {"error": {"message": err, "type": "invalid_request_error"}})
                        return
                    self._send_json(200, {
                        "id": cid, "object": "chat.completion", "created": created, "model": daemon.model_name,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                                     "finish_reason": st.get("finish_reason", "stop")}],
                        "usage": {"prompt_tokens": st.get("prompt_tokens", 0),
                                  "completion_tokens": st.get("tokens", 0),
                                  "total_tokens": st.get("prompt_tokens", 0) + st.get("tokens", 0)},
                        "bihaba_stats": st,
                    })

    act = pool.active
    exposed = host not in ("127.0.0.1", "localhost", "::1")
    print(f"⚡ bihaba-serve v{__version__} live on http://{host}:{port} (Zero Web Dependencies)")
    print(f"   Active : {act.model_name} (arch={act.arch}, topK={act.active_top_k}, spec=k{act.max_spec}, baked={act.baked_toks} toks)")
    if len(pool.catalog) > 1:
        print(f"   Catalog: {', '.join(pool.catalog.keys())} (hot-swap enabled)")
    print(f"   UI     : http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/  ← buka di browser")
    print(f"   Routes : /v1/chat/completions | /v1/models | /v1/state/bake | /health")
    print(f"   Auth   : {'API key aktif' if api_key else 'TANPA API key'} | CORS: {cors_origin or 'mati'} | "
          f"state-dir: {state_dir or '(bake ke file via HTTP mati)'}")
    if exposed and not api_key:
        print("   ⚠ Server terbuka ke jaringan TANPA API key. Siapa pun di jaringan ini bisa memakainya. "
              "Tambahkan --api-key atau pakai --host 127.0.0.1.")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
