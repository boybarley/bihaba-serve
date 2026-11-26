"""
Zero-Dependency OpenAI-Compatible HTTP Server (Python Stdlib Only).
Endpoints: /v1/chat/completions | /v1/models | /v1/state/bake | /health
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .daemon import ModelPool


def run_server(pool: ModelPool, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

        def _send_json(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._send_cors()
            self.end_headers()

        def do_GET(self):
            if self.path == "/v1/models":
                data = [
                    {
                        "id": name,
                        "object": "model",
                        "owned_by": "bihaba-serve",
                        "engine": info.engine,
                        "arch": info.model_type,
                    }
                    for name, info in pool.catalog.items()
                ]
                self._send_json(200, {"object": "list", "data": data})
            elif self.path == "/health":
                act = pool.active
                self._send_json(200, {
                    "status": "ok",
                    "engine": "bihaba-serve (Taalas-Metal)",
                    "version": __version__,
                    "author": "Boy Barley — Bihaba (Team Pasai Aceh)",
                    "active_model": act.model_name if act else None,
                    "arch": act.arch if act else None,
                    "top_k": act.active_top_k if act else 0,
                    "baked_prefix_tokens": act.baked_toks if act else 0,
                    "available_models": list(pool.catalog.keys()),
                    "total_requests": pool.total_requests,
                    "uptime_sec": round(time.time() - pool.started_at, 1),
                })
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length)) if length > 0 else {}

            if self.path == "/v1/state/bake":
                daemon = pool.get_daemon(req.get("model"))
                res = daemon.bake_system_prompt(req.get("system", ""), req.get("path"))
                self._send_json(200, res)
                return

            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": "not found"})
                return

            daemon = pool.get_daemon(req.get("model"))
            messages = req.get("messages", [])
            max_tokens = int(req.get("max_tokens", 512))
            stream = bool(req.get("stream", False))
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
            cid = f"chatcmpl-bihaba-{int(time.time() * 1000)}"
            created = int(time.time())

            if stream:
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for ev in daemon.stream_tokens(messages, **gen_kw):
                    if "delta" in ev:
                        chunk = {
                            "id": cid, "object": "chat.completion.chunk", "created": created,
                            "model": daemon.model_name,
                            "choices": [{"index": 0, "delta": {"content": ev["delta"]}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                    elif ev.get("done"):
                        st = ev["stats"]
                        usage = {
                            "prompt_tokens": st.get("prompt_tokens", 0),
                            "completion_tokens": st.get("tokens", 0),
                            "total_tokens": st.get("prompt_tokens", 0) + st.get("tokens", 0),
                        }
                        chunk = {
                            "id": cid, "object": "chat.completion.chunk", "created": created,
                            "model": daemon.model_name,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": usage, "bihaba_stats": st,
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
            else:
                parts, st = [], {}
                for ev in daemon.stream_tokens(messages, **gen_kw):
                    if "delta" in ev:
                        parts.append(ev["delta"])
                    elif ev.get("done"):
                        st = ev["stats"]
                usage = {
                    "prompt_tokens": st.get("prompt_tokens", 0),
                    "completion_tokens": st.get("tokens", 0),
                    "total_tokens": st.get("prompt_tokens", 0) + st.get("tokens", 0),
                }
                self._send_json(200, {
                    "id": cid, "object": "chat.completion", "created": created, "model": daemon.model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": "stop"}],
                    "usage": usage, "bihaba_stats": st,
                })

    act = pool.active
    print(f"⚡ bihaba-serve v{__version__} live on http://{host}:{port} (Zero Web Dependencies)")
    print(f"   Active : {act.model_name} (arch={act.arch}, topK={act.active_top_k}, spec=k{act.max_spec}, baked={act.baked_toks} toks)")
    if len(pool.catalog) > 1:
        print(f"   Catalog: {', '.join(pool.catalog.keys())} (hot-swap enabled!)")
    print(f"   Routes : /v1/chat/completions | /v1/models | /v1/state/bake | /health")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
