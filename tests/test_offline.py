"""
Tes offline (tanpa Mac / Metal / model). Jalankan:  python3 -m unittest discover -s tests -v
Mencakup: emulasi algoritma kernel (top-64 eksak + flag, penalty per baris, estimator α),
keamanan server (path bake, API key, body limit) dan daemon (buang sisa output saat klien putus).
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

NEG = -1e30


# ─── Port 1:1 dari topk_logits_64 (kernels.metal) ──────────────────────────────────────────────
def topk64_emulated(row):
    V = len(row)
    th_v = [[NEG] * 8 for _ in range(256)]
    th_i = [[0] * 8 for _ in range(256)]
    th_ev = [NEG] * 256
    for t in range(256):
        bv, bi, ev = th_v[t], th_i[t], NEG
        for i in range(t, V, 256):
            v = float(row[i])
            if v > bv[7]:
                ev = max(ev, bv[7])
                j = 7
                while j > 0 and bv[j - 1] < v:
                    bv[j], bi[j] = bv[j - 1], bi[j - 1]
                    j -= 1
                bv[j], bi[j] = v, i
            else:
                ev = max(ev, v)
        th_ev[t] = ev

    def sg_select(cands):                      # cands: list of (v, i) per lane-register
        cands = list(cands)
        out = []
        for _ in range(64):
            k = max(range(len(cands)), key=lambda q: (cands[q][0], -q))
            out.append(cands[k])
            cands[k] = (NEG, cands[k][1])
        return out

    s = []
    for sg in range(8):
        lanes = range(sg * 32, sg * 32 + 32)
        s += sg_select([(th_v[t][j], th_i[t][j]) for t in lanes for j in range(8)])
    final = sg_select(s)
    ev_all = max(th_ev)
    flag = ev_all > final[63][0]
    return [v for v, _ in final], [i for _, i in final], flag


def topk64_with_fallback(row):
    vals, ids, flag = topk64_emulated(row)
    if flag:                                   # = cpu_top64 di main.mm
        ids = list(np.argsort(-row.astype(np.float32), kind="stable")[:64])
        vals = [float(row[i]) for i in ids]
    return vals, ids, flag


@unittest.skipIf(np is None, "numpy tidak tersedia")
class TestTop64(unittest.TestCase):
    def check(self, row):
        vals, ids, flag = topk64_with_fallback(row)
        ref = np.sort(row.astype(np.float32))[::-1][:64]
        np.testing.assert_array_equal(np.array(vals, dtype=np.float32), ref)
        self.assertEqual(len(set(ids)), 64)
        for v, i in zip(vals, ids):
            self.assertEqual(np.float32(row[i]), np.float32(v))
        return flag

    def test_random_rows_exact(self):
        rng = np.random.default_rng(0)
        flags = 0
        for _ in range(6):
            flags += self.check(rng.standard_normal(4096 * 3).astype(np.float16))
        self.assertLess(flags, 6)              # fallback jarang pada logit acak

    def test_clustered_forces_flag(self):
        # 64 nilai tertinggi semua jatuh di thread yang sama (stride 256) → hanya 8 per thread bisa lolos.
        row = np.full(256 * 70, -5.0, dtype=np.float16)
        row[np.arange(64) * 256 + 7] = np.linspace(10, 20, 64).astype(np.float16)
        self.assertTrue(self.check(row))

    def test_old_kernel_would_lose_tokens(self):
        # Kasus yang di versi lama (tanpa flag) membuang token benar: tetap eksak sekarang.
        row = np.zeros(256 * 16, dtype=np.float16)
        row[[3, 259, 515, 771, 1027, 1283, 1539, 1795, 2051, 2307]] = 9.0
        self.check(row)


# ─── Port dari build_rep_rows & OrderAlpha (main.mm) ───────────────────────────────────────────
REP_WIN = 64


def build_rep_rows(hist, batch, M, vocab, pen):
    if not (pen > 0 and (pen > 1.0001 or pen < 0.9999)):
        return None
    rows = []
    for m in range(M):
        total = len(hist) + m
        row = []
        for i in range(max(0, total - REP_WIN), total):
            t = hist[i] if i < len(hist) else batch[1 + i - len(hist)]
            if 0 <= t < vocab and t not in row:
                row.append(t)
        rows.append(row)
    return rows


class OrderAlpha:
    def __init__(self):
        self.A = [0.0] * 5
        self.R = [0.0] * 5

    @staticmethod
    def prior(o):
        return 0.4 if o <= 1 else (0.6 if o == 2 else 0.85)

    def alpha(self, o):
        a = (self.A[o] + 4 * self.prior(o)) / (self.A[o] + self.R[o] + 4)
        return min(0.995, max(0.05, a))

    def update(self, o, acc, rejected):
        self.A[o] = 0.9 * self.A[o] + acc
        self.R[o] = 0.9 * self.R[o] + (1.0 if rejected else 0.0)


class TestSpecHelpers(unittest.TestCase):
    def test_rep_rows_include_own_drafts(self):
        hist = [5, 6, 7]
        batch = [7, 100, 101, 102]              # batch[0] = token terakhir (sudah di hist), lalu draft
        rows = build_rep_rows(hist, batch, 4, 1000, 1.1)
        self.assertEqual(rows[0], [5, 6, 7])
        self.assertEqual(rows[1], [5, 6, 7, 100])          # baris 1 melihat draft 1
        self.assertEqual(rows[3], [5, 6, 7, 100, 101, 102])
        self.assertIsNone(build_rep_rows(hist, batch, 4, 1000, 1.0))
        self.assertIsNone(build_rep_rows(hist, batch, 4, 1000, 0.0))   # struct default {} = mati

    def test_rep_rows_window_and_vocab(self):
        hist = list(range(200))
        rows = build_rep_rows(hist + [999999], [0], 1, 1000, 1.2)
        self.assertEqual(len(rows[0]), 63)                 # 64 terakhir minus 1 token di luar vocab
        self.assertNotIn(999999, rows[0])

    def test_alpha_is_geometric(self):
        a = OrderAlpha()
        for _ in range(30):
            a.update(3, 5, True)                           # 5 diterima lalu 1 ditolak
        self.assertAlmostEqual(a.alpha(3), 5 / 6, delta=0.02)
        b = OrderAlpha()
        for _ in range(30):
            b.update(2, 7, False)                          # tak pernah ditolak → mendekati batas atas
        self.assertGreater(b.alpha(2), 0.97)
        self.assertEqual(OrderAlpha().alpha(1), 0.4)       # prior


# ─── Server: path bake, auth, body ─────────────────────────────────────────────────────────────
from bihaba_serve import server as srv  # noqa: E402


class TestStatePath(unittest.TestCase):
    def test_disabled_without_state_dir(self):
        self.assertEqual(srv._state_file(None, None), (None, None))
        p, err = srv._state_file(None, "/etc/passwd")
        self.assertIsNone(p)
        self.assertIn("--state-dir", err)

    def test_traversal_is_neutralized(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in ["../../x", "/etc/cron.d/evil", "..\\..\\x", "a/b/c.tstate"]:
                p, err = srv._state_file(d, bad)
                if p:
                    self.assertEqual(Path(p).parent, Path(d).resolve())
                    self.assertTrue(p.endswith(".tstate"))
            self.assertIsNotNone(srv._state_file(d, "spa ce;rm")[1])
            self.assertIsNotNone(srv._state_file(d, ".hidden")[1])
            p, err = srv._state_file(d, "chat_sys")
            self.assertEqual(Path(p).name, "chat_sys.tstate")


class _FakeDaemon:
    model_name, arch, active_top_k, max_spec, baked_toks, max_seq = "fake", "qwen3_5", 8, 7, 0, 4096

    def __init__(self):
        self.baked_paths = []

    def bake_system_prompt(self, text, path):
        self.baked_paths.append(path)
        return {"baked": True, "tokens": 3}

    def clear_state(self):
        return {"cleared": True, "mb": 70.9}

    def stream_tokens(self, messages, **kw):
        if messages[-1].get("content") == "boom":
            yield {"error": "prompt terlalu panjang"}
            return
        yield {"delta": "halo"}
        yield {"done": True, "stats": {"tokens": 1, "prompt_tokens": 2, "finish_reason": "stop"}, "raw_ids": [1]}


class _FakePool:
    def __init__(self):
        import contextlib, time
        self.active = _FakeDaemon()
        self.catalog = {}
        self.total_requests = 0
        self.started_at = time.time()
        self._cm = contextlib.contextmanager

    def session(self, model=None):
        import contextlib

        @contextlib.contextmanager
        def cm():
            yield self.active
        return cm()


class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); cls.port = s.getsockname()[1]; s.close()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pool = _FakePool()
        cls._stdout = sys.stdout
        sys.stdout = io.StringIO()
        threading.Thread(target=srv.run_server, args=(cls.pool, "127.0.0.1", cls.port),
                         kwargs=dict(api_key="rahasia", state_dir=cls.tmp.name), daemon=True).start()
        import time
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.05)
        sys.stdout = cls._stdout

    def req(self, path, body=None, key="rahasia", raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data)
        if key:
            r.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(r, timeout=5) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_web_ui_served_same_origin(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5) as r:
            body = r.read().decode()
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers["Content-Type"])
        self.assertIn("/v1/chat/completions", body)

    def test_health_is_public(self):
        code, j = self.req("/health", key=None)
        self.assertEqual(code, 200)
        self.assertEqual(j["max_seq"], 4096)

    def test_auth_required(self):
        self.assertEqual(self.req("/v1/models", key=None)[0], 401)
        self.assertEqual(self.req("/v1/models", key="salah")[0], 401)
        self.assertEqual(self.req("/v1/models")[0], 200)

    def test_chat_ok_and_error_propagates(self):
        code, j = self.req("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200)
        self.assertEqual(j["choices"][0]["message"]["content"], "halo")
        code, j = self.req("/v1/chat/completions", {"messages": [{"role": "user", "content": "boom"}]})
        self.assertEqual(code, 400)
        self.assertIn("terlalu panjang", j["error"]["message"])

    def test_bad_bodies(self):
        self.assertEqual(self.req("/v1/chat/completions", raw=b"{nope")[0], 400)
        self.assertEqual(self.req("/v1/chat/completions", raw=b"[1,2]")[0], 400)
        self.assertEqual(self.req("/v1/chat/completions", {"messages": []})[0], 400)
        self.assertEqual(self.req("/v1/chat/completions", {"messages": [{"content": "x"}], "max_tokens": "abc"})[0], 400)

    def test_clear_state(self):
        self.assertEqual(self.req("/v1/state/clear", {}, key=None)[0], 401)
        code, j = self.req("/v1/state/clear", {})
        self.assertEqual(code, 200)
        self.assertTrue(j["cleared"])

    def test_bake_confined_to_state_dir(self):
        code, _ = self.req("/v1/state/bake", {"system": "x", "path": "../../../tmp/pwn"})
        self.assertEqual(code, 200)
        written = self.pool.active.baked_paths[-1]
        self.assertEqual(Path(written).parent, Path(self.tmp.name).resolve())
        self.assertEqual(Path(written).name, "pwn.tstate")


# ─── Daemon: sisa output engine dibuang saat generator ditutup lebih awal ─────────────────────
FAKE_ENGINE = r'''
import sys, json
for line in sys.stdin:
    req = json.loads(line)
    for t in req["prompt_ids"]:
        print(json.dumps({"tok": t}), flush=True)
    print(json.dumps({"done": True, "tokens": len(req["prompt_ids"])}), flush=True)
'''


class _Tok:
    eos_ids = {0}

    def decode(self, ids):
        return "".join(chr(96 + i) for i in ids)


@unittest.skipIf(os.name == "nt", "butuh pipe POSIX")
class TestDaemonDrain(unittest.TestCase):
    def test_disconnect_does_not_leak_into_next_request(self):
        import subprocess
        from bihaba_serve.daemon import EngineDaemon
        d = object.__new__(EngineDaemon)
        d.proc = subprocess.Popen([sys.executable, "-c", FAKE_ENGINE], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        d.lock = threading.Lock()
        d.tok = _Tok()
        d.default_temp, d.default_rep_pen, d.baked_toks = 0.0, 1.0, 0
        d.can_cancel = False
        try:
            g = d.stream_prompt_ids([1, 2, 3, 4, 5], max_tokens=10)
            self.assertEqual(next(g), {"delta": "a"})
            g.close()                                      # klien putus setelah 1 token
            evs = list(d.stream_prompt_ids([3, 3], max_tokens=10))
            text = "".join(e.get("delta", "") for e in evs)
            self.assertEqual(text, "cc")                   # bukan "bcdecc" seperti di versi lama
            self.assertTrue(evs[-1]["done"])
        finally:
            d.proc.kill(); d.proc.wait(); d.proc.stdin.close(); d.proc.stdout.close()


# ─── Port dari dequant_gen16_to_tg (MMA 3/6/8-bit) vs unpack referensi ─────────────────────────
def pack_mlx(q, bits):
    """Bit-stream LSB-first, sama dengan mlx.core.quantize."""
    acc, n, words = 0, 0, []
    for v in q:
        acc |= int(v) << n
        n += bits
        while n >= 32:
            words.append(acc & 0xFFFFFFFF); acc >>= 32; n -= 32
    if n:
        words.append(acc & 0xFFFFFFFF)
    return words


def dequant_gen16(wrow, k, bits):
    mask = (1 << bits) - 1
    nw = (16 * bits + 31) // 32 + 1
    bit0 = k * bits
    base, sh = bit0 >> 5, bit0 & 31
    a = [(wrow[base + i] if i * 32 < sh + 16 * bits else 0) for i in range(nw)]
    out = []
    for j in range(16):
        pos = sh + j * bits
        wi, o = pos >> 5, pos & 31
        q = a[wi] >> o
        if o + bits > 32:
            q |= (a[wi + 1] << (32 - o)) & 0xFFFFFFFF
        out.append(q & mask)
    return out


@unittest.skipIf(np is None, "numpy tidak tersedia")
class TestGenericDequant(unittest.TestCase):
    def test_bits_3_6_8(self):
        rng = np.random.default_rng(1)
        for bits in (3, 6, 8):
            K = 256
            q = rng.integers(0, 1 << bits, size=K)
            w = pack_mlx(q, bits)
            self.assertEqual(len(w), K * bits // 32)            # = row_words di kernel
            for k in range(0, K, 16):
                self.assertEqual(dequant_gen16(w, k, bits), list(q[k:k + 16]), (bits, k))


# ─── Streaming: detokenizer inkremental, filter stop, cache sesi token-eksak ──────────────────
from bihaba_serve.daemon import EngineDaemon, StopFilter, StreamDecoder  # noqa: E402
from bihaba_serve.templates import format_messages  # noqa: E402

_SPECIAL = {"<|im_start|>": 1000, "<|im_end|>": 1001, "<think>": 1002, "</think>": 1003}
_SPECIAL_INV = {v: k for k, v in _SPECIAL.items()}


class _CharTok:
    """Tokenizer mainan: token khusus utuh, karakter lain = ord(c). Cukup untuk logika prefix/LCP."""
    template = "qwen35"
    eos_ids = {1001}

    def encode_raw(self, text):
        out, i = [], 0
        while i < len(text):
            for sp, tid in _SPECIAL.items():
                if text.startswith(sp, i):
                    out.append(tid); i += len(sp); break
            else:
                out.append(ord(text[i])); i += 1
        return out

    def decode(self, ids):
        return "".join(chr(t) for t in ids if t not in _SPECIAL_INV)

    def token_id(self, text):
        return _SPECIAL.get(text)

    def format_chat_with_prefix(self, messages, enable_thinking=False):
        sys_s, full = format_messages(messages, template=self.template, enable_thinking=enable_thinking)
        return self.encode_raw(full), 0


# Engine palsu dengan semantik cache yang sama dengan engine asli: state = prompt + token keluaran kecuali
# yang terakhir; cache live dipakai hanya bila state lama adalah prefix utuh prompt baru.
FAKE_CACHE_ENGINE = r"""
import sys, json
cached = []
for line in sys.stdin:
    req = json.loads(line)
    pids = req["prompt_ids"]
    reused = len(cached) if cached and pids[:len(cached)] == cached else 0
    out = [ord(c) for c in "Halo!"] + [1001]
    for t in out:
        print(json.dumps({"tok": t}), flush=True)
    cached = pids + out[:-1]
    print(json.dumps({"done": True, "tokens": len(out), "reused": reused}), flush=True)
"""


class TestStreaming(unittest.TestCase):
    def test_stop_filter_holds_partial_stop(self):
        f = StopFilter(["\nUser:"])
        sent = ""
        for piece in ["Jawab", "an.\nUs", "er: lagi"]:
            out, hit = f.feed(piece)
            sent += out
            if hit:
                break
        self.assertTrue(hit)
        self.assertEqual(sent, "Jawaban.")          # potongan "\nUs" tidak pernah terkirim
        g = StopFilter(["###"])
        out, hit = g.feed("a#b#")
        self.assertEqual((out, hit), ("a#b", False))
        self.assertEqual(g.flush(), "#")

    def test_context_overflow_is_clear_error(self):
        d = object.__new__(EngineDaemon)
        d.max_seq, d.default_temp, d.default_rep_pen = 32, 0.0, 1.0
        evs = list(d.stream_prompt_ids(list(range(30)), max_tokens=1024))
        self.assertEqual(len(evs), 1)
        self.assertIn("--max-seq", evs[0]["error"])

    def test_stream_decoder_matches_full_decode(self):
        tok = _CharTok()
        ids = tok.encode_raw("Halo dunia, apa kabar?")
        d = StreamDecoder(tok)
        self.assertEqual("".join(d.push(t) for t in ids), tok.decode(ids))

    @unittest.skipIf(os.name == "nt", "butuh pipe POSIX")
    def test_session_cache_hits_on_second_turn(self):
        import subprocess
        d = object.__new__(EngineDaemon)
        d.proc = subprocess.Popen([sys.executable, "-c", FAKE_CACHE_ENGINE], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        d.lock = threading.Lock()
        d.tok = _CharTok()
        d.default_temp, d.default_rep_pen, d.baked_toks = 0.0, 1.0, 0
        d.enable_thinking, d.session_cache, d._sess, d.can_cancel = False, True, None, False
        d._im_end = 1001
        try:
            conv = [{"role": "system", "content": "Kamu asisten."}, {"role": "user", "content": "Hai"}]
            ev1 = list(d.stream_tokens(conv, max_tokens=50))
            self.assertEqual(ev1[-1]["stats"]["reused"], 0)
            # Klien mengirim balik jawaban (tanpa blok think, spasi dipangkas) + pesan baru
            conv2 = conv + [{"role": "assistant", "content": " Halo! "}, {"role": "user", "content": "Apa kabar?"}]
            # Bukti bug lama: render ulang kanonik TIDAK memuat state engine sebagai prefix
            canon, _ = d.tok.format_chat_with_prefix(conv2)
            first, _ = d.tok.format_chat_with_prefix(conv)
            state = first + d.tok.encode_raw("Halo!")
            self.assertNotEqual(canon[:len(state)], state)
            ev2 = list(d.stream_tokens(conv2, max_tokens=50))
            st = ev2[-1]["stats"]
            self.assertTrue(st["session_hit"])
            self.assertEqual(st["reused"], len(state))     # seluruh riwayat dipakai ulang, hanya ekor diproses
            # Percakapan lain (riwayat berbeda) → tidak memakai sesi
            ev3 = list(d.stream_tokens([{"role": "user", "content": "Topik baru"}], max_tokens=50))
            self.assertFalse(ev3[-1]["stats"]["session_hit"])
        finally:
            d.proc.kill(); d.proc.wait(); d.proc.stdin.close(); d.proc.stdout.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
