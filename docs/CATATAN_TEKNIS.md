# Catatan Teknis bihaba-serve: temuan, perubahan, dan peta jalan

Dokumen kerja untuk siapa pun (manusia atau AI) yang melanjutkan pengembangan.
Semua angka diukur di **Apple M1 Max 64 GB**, dengan model yang sama dan dalam sesi yang sama.
Ini kecuali disebut lain.

---

## 1. Hasil terukur: v1.0 → v1.4

| Model | Metrik | v1.0 | v1.4 | Perubahan |
|---|---|---|---|---|
| Bihaba-AI-1.7B (Qwen3, 6-bit) | Prefill 512 tok | 275 tok/s | **1153 tok/s** | 4.2× |
| | Spekulatif k=7 (pola sintetis) | 232.7 tok/s | **455 tok/s** | 1.96× |
| | Langkah verifikasi M=8 | 32.0 ms | **16.1 ms** | 2.0× |
| | Decode greedy M=1 | 132.6 tok/s | 133.7 tok/s | = |
| Qwythos-9B (Qwen3.5 DeltaNet, 4-bit) | Prefill 512 tok | 104 tok/s | **279 tok/s** | 2.7× |
| | Spekulatif / decode | 141 / 46.3 | 135–141 / 46.3 | = |
| 35B-A3B MoE (`--top-k 4`) | Prefill 512 tok | 217 tok/s | **434 tok/s** | 2.0× |
| | Spekulatif / decode | 246.6 / 97 | 246 / 97 | = |

Di setiap run, `=greedy 128/128` (output spekulatif identik dengan greedy) dan `Prefill check 32/32` (prefill baru identik dengan jalur v1.0).

**Pembanding mlx-lm 0.29.1** (model 1.7B yang sama, prompt sekitar 550 token):

| | mlx-lm | bihaba v1.4 | Selisih |
|---|---|---|---|
| Prefill | 912 tok/s | 1153 tok/s | +26% |
| Decode | 118.6 tok/s | 133.7 tok/s | +13% |

mlx-lm 0.29.1 menolak `qwen3_5` (Qwythos/MoE). Versi itu tertahan karena Python 3.9 dari Command Line Tools. Belum diuji dengan mlx-lm terbaru.

---

## 2. Bug yang ditemukan dan diperbaiki

| # | Masalah (v1.0) | Dampak | Perbaikan |
|---|---|---|---|
| 1 | Top-64 di GPU tidak eksak (tiap thread hanya menyimpan 8 kandidat) | Sampling kehilangan kandidat pada sebagian besar langkah | Thread mencatat nilai tertinggi yang dibuang; bila melebihi kandidat ke-64, baris itu dihitung ulang di CPU (jarang) |
| 2 | Load `.tstate` tanpa validasi ukuran | Buffer overflow dengan file yang salah atau dari model lain | Format v2: fingerprint model (FNV-1a config.json + top_k), semua ukuran dicek, simpan atomik |
| 3 | `/v1/state/bake` menerima path bebas, server di 0.0.0.0, CORS `*`, tanpa auth | Siapa pun di jaringan, atau situs web mana pun, bisa menulis file di disk | Bind 127.0.0.1, `--api-key`, CORS opt-in, bake hanya ke `--state-dir`, batas body 16 MB |
| 4 | `cached_ids` dicatat di dalam loop emisi yang bisa berhenti dini | Cache prefix tidak sinkron dengan state DeltaNet → giliran chat berikutnya salah tanpa error | Catat semua token yang sudah masuk state sebelum loop emisi |
| 5 | Tidak ada batas `max_seq` per request | Tulis di luar KV cache | Cek di setiap `run_step` dan request |
| 6 | Repetition penalty satu baris untuk semua baris verifikasi | Output spekulatif ≠ non-spekulatif | Penalty per baris: baris *m* melihat history + draft 1..m |
| 7 | Estimator α memakai EMA dari acc/(M−1) | Satu penolakan memangkas k terlalu keras | α per orde n-gram = accepted / (accepted + rejections), dengan prior |
| 8 | Klien putus di tengah stream | Sisa token bocor ke request berikutnya | Daemon membuang output engine sampai baris `done` |
| 9 | Hot-swap model menutup daemon yang sedang dipakai request lain | Crash atau respons rusak | `gen_lock` dipegang selama swap dan selama satu request penuh |
| 10 | Error dari engine tidak diteruskan | Request menggantung atau 200 kosong | Diteruskan sebagai HTTP 400 / event `error` |
| 11 | `align` mengganti symlink cache HuggingFace | Disk 2× dan cache rusak | Menulis ke target symlink, cek ruang disk |
| 12 | README menampilkan angka spekulatif sintetis tanpa label | Ekspektasi salah | Diberi label; tabel v1.0 vs v1.x terukur |

---

## 3. Temuan performa (pelajaran untuk kernel Metal)

1. **Jangan simpan akumulator `simdgroup_matrix` dalam array.**
   - Versi `c[MT]` ditambah loop dengan `break` dinamis membuat compiler menaruh akumulator di memori.
   - Akibatnya **1.5–1.7× lebih lambat** di semua bagian pada M=8 (FFN 32 → 50 ms).
   - Pakai variabel bernama (`c0..c3`) dan cabang berdasarkan konstanta template.
   - Gejala yang terlihat: M=1 normal, tetapi M≥4 lambat secara merata.
2. **Rasio load:MMA menentukan kecepatan GEMM di Apple GPU.**
   - Kernel tile 8 token: sekitar 1.25 load per MMA, sehingga dibatasi memori threadgroup.
   - Tile 32×64 dengan 8 akumulator per simdgroup (0.75 load per MMA): prefill **+50%**.
3. **Okupansi GPU penting.**
   - Chunk 32 hanya memberi N/64 threadgroup (misalnya 32 untuk N=2048), yaitu satu per core.
   - Grid 2D (N/64 × M/32) dengan chunk 128: prefill **+50%** lagi.
4. **Model 3/6/8-bit tadinya tidak punya jalur MMA** dan selalu jatuh ke `qmv` multi-token (biaya ∝ M).
   - Dekuantisasi generik bit-stream LSB-first (format MLX) memperbaikinya.
   - Dekuantisasi generik ini lebih mahal, sehingga MMA baru menang mulai **M=6** (M=4: 23 ms MMA vs 18.6 ms qmv). Lihat `mma_min_m()`.
5. **Decode M=1 sudah dibatasi bandwidth memori** (9B: sekitar 275 GB/s). Peluang di sana kecil; keuntungan besar ada di prefill, verifikasi, dan batching.
6. **Prefill MoE dibatasi kernel expert**, bukan proyeksi dense (434 tok/s). Expert dijalankan per token × top-k tanpa pemakaian ulang bobot antar-token.
7. **Hasil negatif yang tercatat** (jangan diulang tanpa ide baru): SwiGLU v2 1.8× lebih lambat, dan head "bound" tidak berguna (lihat hardcore-on-mac).

### Metode kerja yang terbukti efektif
- **Selalu A/B dengan biner v1.0** (`git checkout 8b8962e && make`) di mesin dan model yang sama, dalam sesi yang sama. Angka README lama tidak bisa dijadikan pembanding.
- **`--profile` per kernel** langsung menunjukkan bagian yang melambat.
- **Cek kebenaran dibangun ke dalam `bench`:**
  - `=greedy` menguji spekulasi;
  - `Prefill check` menguji prefill.
  Benchmark tanpa cek kebenaran tidak dihitung.
- Shader Metal tidak bisa dikompilasi di luar Mac, jadi setiap perubahan kernel butuh satu putaran run di Mac.
- **Aturan Metal yang sudah memakan korban:**
  - variabel `threadgroup` hanya boleh di scope kernel;
  - hindari `INFINITY` dengan fast-math (pakai `-1e30f`);
  - waspadai vexing parse C++ (`std::vector<int> v(static_cast<size_t>(n))`).

---

## 4. Ringkasan perubahan per versi

| Versi | Isi |
|---|---|
| v1.1 | Kernel MMA template (8 & 32 token), kernel prefill DeltaNet, LM head hanya token terakhir, top-64 eksak, penalty per baris, OrderAlpha, `.tstate` v2, server aman, tes offline |
| v1.1.1 | MMA untuk 3/6/8-bit (dekuantisasi generik) |
| v1.1.2 | Ambang MMA per bit-width (`mma_min_m`) |
| v1.1.3 | Akumulator bernama: regresi 1.5× hilang |
| v1.2 | Kernel GEMM prefill 32×64, cek kebenaran prefill di bench |
| v1.3 | Chunk prefill 128 + grid GEMM 2D |
| v1.4 | Web Console bawaan di `/` (same-origin, tanpa CORS) |
| v1.4.2 | `POST /v1/state/clear` + tombol "Hapus bake": buang snapshot system prompt dari RAM GPU |

---

## 5. Peta jalan (urut dampak / usaha)

### Metal / engine
1. **Batching multi-user di dimensi M** (dampak terbesar untuk server).
   - Kernel verifikasi sudah memproses 8 baris per baca bobot; baris itu bisa berasal dari *pengguna berbeda*.
   - 4 pengguna decode bersamaan ≈ biaya 1.5× satu pengguna. Ini prinsip Taalas di level server.
   - Butuh KV/DeltaNet state per slot dan scheduler.
2. **Prefill MoE: grouped GEMM per expert.** Urutkan token per expert, lalu jalankan GEMM 32×64 per kelompok. Target: prefill MoE mendekati 2× dari sekarang.
3. **Profil prefill per kernel** (`--profile` untuk M=128). Ini untuk mengetahui apakah DeltaNet, attention, atau GEMM yang dominan di Qwythos (279 tok/s).
4. **DeltaNet prefill bentuk chunkwise** (WY / chunked delta rule berbasis MMA), menggantikan loop per token, bila profil menunjukkan DeltaNet dominan.
5. **Attention prefill ber-tile (flash-attention MMA).** Kernel sekarang per (head, token) dengan dot skalar, yaitu O(n²) skalar. Perlu benchmark prompt 2k/8k/32k dulu.
6. **Dekuantisasi 3/6/8-bit yang lebih cepat** (trik bit per kata seperti 4-bit) agar MMA menang di M=4.
7. **Tuning GEMM:** double-buffer tile, BK 32 vs 64, tile 64 token untuk model besar.
8. **Benchmark spekulasi di prompt nyata** (chat, kode, JSON). Angka sintetis adalah batas atas.

### Server / Python
1. **Scheduler untuk batching** (pasangan poin Metal #1). Saat ini `gen_lock` menserialkan semua request.
2. **Cache prefix banyak sesi** (LRU snapshot per percakapan), bukan hanya `cached_ids` tunggal + satu `ram_snap`.
3. **Kompatibilitas OpenAI lebih lengkap:** `response_format` JSON, `tools`, `logprobs`, `usage` di stream (sebagian sudah ada).
4. **`/metrics`** (Prometheus) untuk tok/s, TTFT, panjang antrean.
5. **Banner versi engine** masih "v1.0" / ready JSON "v15", jadi perlu disinkronkan dengan `__version__`.
6. **Skrip regresi GPU** (`scripts/regress.sh`): jalankan bench untuk 3 model, gagal bila `=greedy` atau `Prefill check` tidak penuh.
7. **Distribusi:**
   - upload lewat web GitHub menghapus bit eksekusi, jadi pakai `git push` atau `bash install.sh`;
   - Python 3.9 dari CLT tetap didukung (semua file `from __future__ import annotations`).

---

## 6. Cara uji cepat setelah perubahan

```zsh
cd ~/bihaba-serve && git pull && bash install.sh 2>&1 | tail -1
python3 -m unittest discover -s tests -v
bihaba-serve bench --model ~/models/Bihaba-AI-1.7B-Final --tokens 128          # transformer, 6-bit
bihaba-serve bench --model ~/models/Qwythos-9B-Claude --tokens 128             # qwen35 dense
bihaba-serve bench --model ~/models/Bihaba-bagus-pakai --top-k 4 --tokens 128  # qwen35 MoE
# profil per kernel:
B=~/.local/share/bihaba-serve; BIHABA_SHADER_DIR=$B $B/bihaba-qwen35 ~/models/Qwythos-9B-Claude --profile --bench 128
```

Syarat lulus:
- `=greedy 128/128`;
- `Prefill check 32/32`;
- tidak ada angka yang turun dibanding tabel di bagian 1.
