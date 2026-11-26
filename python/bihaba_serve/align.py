"""
64-Byte Header Alignment Utility for Instant Zero-Copy Metal mmap() (0 GB disk duplication).
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path


def align_model_dir(model_dir: str | Path) -> None:
    p = Path(model_dir).expanduser().resolve()
    shards = sorted(p.glob("*.safetensors"))
    if not shards:
        print(f"No .safetensors files in {p}")
        return
    print(f"▸ Aligning {p.name} ({len(shards)} shards)...")
    for fp in shards:
        with fp.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            data_off = 8 + hlen
            rem = data_off % 64
            if rem == 0:
                print(f"  ✓ {fp.name}: already 64B-aligned (data_off={data_off})")
                continue
            pad = 64 - rem
            hdr = f.read(hlen)
            json.loads(hdr)
            tmp = fp.with_suffix(".safetensors.align_tmp")
            with tmp.open("wb") as out:
                out.write(struct.pack("<Q", hlen + pad))
                out.write(hdr)
                out.write(b" " * pad)
                while True:
                    buf = f.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
        os.replace(tmp, fp)
        print(f"  ✓ {fp.name}: padded +{pad}B -> data_off={data_off + pad} (64B-aligned)")
