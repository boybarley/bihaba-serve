# Catatan Teknis bihaba-serve: temuan, perubahan, dan peta jalan

## 0. ATURAN KERJA (wajib untuk siapa pun: manusia atau AI)
Aturan ini lahir dari kesalahan nyata pada 26–27 Nov 2026 (lihat §8–§9). Aturan ditulis di repo, bukan di ingatan siapa pun.

1. **Empat kelas model, empat keputusan.** Setiap perubahan dinilai terpisah untuk:
   - transformer kecil (MNN 0.3B);
   - transformer menengah (Bihaba-AI-1.7B);
   - qwen35 dense (Qwythos-9B);
   - qwen35 MoE (Bihaba-bagus-pakai).

   Setelan tidak pernah disamaratakan. Kebijakan per kelas ditulis eksplisit di kode (contoh: `SESSION_CACHE_BY_ENGINE`).
2. **Satu perintah uji, sebelum push apa pun:** `python3 scripts/uji_semua.py`. Build hanya boleh naik ke stabil bila hasilnya **LULUS SEMUA**:
   - kebenaran kernel;
   - jawaban chat identik dengan acuan;
   - tidak ada angka kecepatan yang turun lebih dari 5%.
3. **Acuan = versi yang sudah dinilai baik oleh pemilik proyek**, disimpan dengan `uji_semua.py --simpan-acuan` di folder `~/bihaba-acuan`. Acuan hanya diganti setelah versi baru LULUS dan dipakai.
4. **Yang sudah bagus tidak boleh turun.** Perbaikan yang menurunkan kelas lain ditolak, bukan kelas lain yang dikorbankan.
5. **Fitur yang mengubah input model** (template, riwayat, cache) mati secara default, kecuali di kelas yang sudah terbukti identik jawabannya.
6. **Dugaan ditulis sebagai dugaan.** Yang dicatat sebagai fakta hanya angka yang diukur di Mac pemilik proyek.
7. **Eksperimen di `~/bihaba-dev`, stabil di `~/bihaba-serve` + GitHub `main`.** Eksperimen tidak pernah di-push.
8. **Aturan Metal yang sudah memakan korban** dicek otomatis oleh `TestMetalLint`:
   - variabel `threadgroup` hanya boleh di scope kernel;
   - tidak boleh memakai `INFINITY` (gunakan `-1e30f`);
   - akumulator simdgroup tidak boleh berupa array.
9. **Batas fisik disebut di awal, bukan di akhir.** Lihat roofline §7: decode M=1 ≤ bandwidth / byte per token.


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
| v1.5-dev | Revisi kolaborator (qwen35) + port ke transformer: tile DN/QKV MMA per tensor (perbaiki batas segmen yang bukan kelipatan 32), race `sm_ss` antar-token di `deltanet_recurrent_spec`, emisi dipotong `n_keep` sebelum commit (cache prefix percakapan berikutnya tetap cocok), tolak bit-width tanpa kernel (qwen35 M=1: 2/3/4/8; expert MoE: 2/3/4), QoS USER_INTERACTIVE, command buffer unretained + autoreleasepool, GEMM lewati MMA untuk setengah tile kosong. Roofline + `qmv_4b_r4` A/B. |
| v1.5-dev (2) | Review AI kolaborator: **cache sesi token-eksak** (multi-giliran tadinya selalu prefill ulang seluruh riwayat karena blok `<think></think>` kosong di template qwen35 tidak ikut dirender ulang → prefix pecah); **cancel SIGUSR1** saat stop sequence / klien putus (engine tadinya jalan sampai max_tokens sambil memegang lock); detokenizer inkremental (O(n) bukan O(n²)); stop filter menahan potongan stop; `disable_nagle_algorithm`. Flag `--no-session-cache`. |
| **v1.6.2** | `scripts/uji_semua.py` (satu perintah untuk 4 kelas, LULUS/GAGAL terhadap acuan) + §0 Aturan Kerja. Tanpa perubahan runtime. |
| **v1.6.1** | Cache sesi **per kelas model**: aktif untuk qwen35 (Qwythos, MoE; perilaku cepat v1.5.0), mati untuk transformer (model kecil). Kesalahan v1.5.1–v1.6.0: satu setelan global untuk semua kelas sehingga Qwythos kembali lambat. |
| **v1.6.0** | Attention decode split-sequence (flash-decoding 2 pass) untuk konteks ≥1024, **aktif di engine transformer saja**. Konteks dibatasi ke `max_position_embeddings`. Tes lint Metal. Lihat §9. |
| **v1.5.1 (hotfix)** | Cache sesi dimatikan secara default (opt-in `--session-cache`). Regresi terbukti dengan `scripts/regress_chat.py`, lihat §8. |
| **v1.5 (stabil)** | Semua di atas. `qmv_4b_r4` dihapus setelah A/B (hasil negatif, lihat bagian 7). |
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

---

## 7. Roofline decode (v1.5-dev, M1 Max, bandwidth terukur ≈ 322–327 GB/s)

| Model | MB/token | Decode | % bandwidth | Waktu/token | GPU | CPU+sinkron | Ideal |
|---|---|---|---|---|---|---|---|
| MNN-Pocket 0.3B (4-bit) | 171 | 331 tok/s | **18%** | 3.02 ms | 2.56 | 0.47 (15%) | 0.53 ms |
| Bihaba-AI 1.7B (6-bit) | 1405 | 132 tok/s | 58% | 7.56 ms | 7.08 | 0.48 (6%) | 4.36 ms |
| Qwythos-9B (4-bit) | 4567 | 46.3 tok/s | 65% | 21.58 ms | 21.03 | 0.55 (3%) | 14.11 ms |
| MoE 35B-A3B top-k 4 | 1511 | 98.4 tok/s | 46% | 10.16 ms | 9.51 | 0.65 (6%) | 4.63 ms |

Kesimpulan:
- **CPU/sinkron bukan masalah utama** (3–15%). Sumber kerugian terbesar adalah waktu GPU, yaitu 1.5–4.8× di atas waktu ideal.
- **Ada overhead tetap per layer sekitar 70–120 µs:**
  - 10 dispatch dan 8 barrier per layer transformer;
  - fase kecil yang tidak mengisi GPU (RMSNorm satu threadgroup, rope, attention per head);
  - ekor (tail) tiap kernel.

  Overhead ini mendominasi model kecil (0.3B: 18%).
- **Kernel matriks besar pun tidak penuh.** Di Qwythos, FFN M=1 butuh 12.7 ms vs ideal sekitar 8.4 ms (≈66%), sedangkan head 248K efisien (≈89%).
  - Hipotesis: qmv v1 hanya 2 baris × 8 byte per lane, sehingga terlalu sedikit load yang berjalan bersamaan.
  - Diuji dengan `qmv_4b_r4` (A/B otomatis di bench).
- **Rencana berikutnya, berdasarkan data:**
  1. qmv lebih lebar (r4) untuk semua bit-width;
  2. fusi RMSNorm ke proyeksi berikutnya dan fusi QKV satu dispatch (8 → 5 barrier per layer);
  3. decode loop di GPU (token tidak kembali ke CPU) untuk greedy/sampling GPU.

### Hasil A/B `qmv_4b_r4` (hasil negatif, kernel dihapus di v1.5)
Kernel ini menangani 4 baris per simdgroup dengan 16 byte per lane. Hasilnya: Qwythos 46.3 → 46.2 tok/s, 1.7B tidak terpengaruh (6-bit), 0.3B −3%, MoE −6%. Greedy identik di semua model.
- **Kesimpulan:** hipotesis "load kurang paralel" salah. Kerugian di qmv M=1 bukan soal jumlah byte yang dibaca bersamaan.
- **Penjelasan yang tersisa:** overhead per dispatch/barrier dan ekor (tail) kernel per fase.
- **Langkah berikutnya:** fusi kernel (RMSNorm ke proyeksi, satu dispatch QKV, combine+residual+norm), bukan kernel matvec yang lebih lebar.

### Cache sesi di engine asli (Qwythos-9B, Web Console, Greedy)
| Riwayat dipakai ulang | TTFT | prefill ekor | Decode | token/langkah |
|---|---|---|---|---|
| 4.265 tok | 351 ms | 335 ms | 81.3 tok/s | 4.65 |
| 5.307 tok | 325 ms | 303 ms | 90.1 tok/s | 5.48 |
| 5.885 tok | 311 ms | 287 ms | 97.3 tok/s | 6.34 |

- **Tanpa cache:** sekitar 21 detik per giliran (5.9k token dengan laju 279 tok/s).
- **Catatan:** prefill ekor sekitar 300 ms untuk puluhan token masih mahal. Penyebabnya attention prefill naif yang membaca seluruh riwayat per token. Ini kandidat berikutnya: attention ber-tile (flash-attention MMA).

---

## 8. Regresi v1.5: cache sesi mengubah jawaban model (dibuktikan `scripts/regress_chat.py`)

Percakapan 3 giliran, greedy. A = riwayat dirender ulang secara kanonik, B = cache sesi.
- **Bihaba-AI-1.7B:** giliran 1 identik (tanpa riwayat). Di giliran 2–3, jawaban B **berubah**: bergeser ke kosakata Aceh ("areueng lembab", "hana buah"), sementara A tetap berbahasa Indonesia.
- **Penyebab:** di B, riwayat jawaban lama tetap membawa blok `<think>\n\n</think>\n\n` kosong, persis seperti saat model menghasilkannya. Template kanonik Qwen3 merender jawaban lama TANPA blok itu. Model yang di-fine-tune dengan template kanonik melihat format di luar distribusinya, sehingga perilakunya bergeser.
- **Pelajaran:** cek `=greedy`/`Prefill check` hanya menguji konsistensi engine. Fitur yang mengubah **input** model (template, riwayat) harus diuji dengan membandingkan jawaban terhadap jalur kanonik, di **semua** model.
- **Arah perbaikan yang aman:**
  - engine transformer (attention murni) bisa memakai ulang prefix SEBAGIAN: KV di belakang LCP cukup ditimpa, sehingga riwayat kanonik tetap dipakai dan TTFT tetap cepat;
  - engine qwen35 (DeltaNet) tidak bisa mundur, jadi butuh checkpoint state di awal setiap jawaban.
- **MNN-Pocket 0.3B:** jawaban sudah ngawur di giliran 1 (tanpa riwayat), baik di A maupun B. Ini bukan akibat cache sesi. Perlu pembanding mlx-lm untuk memastikan apakah ini batas kualitas model atau masalah template/engine.

---

## 9. v1.6: attention decode di konteks panjang (M1 Max, konteks 3072, 64 token greedy)

| Model | Attention lama | Split | Greedy | Default |
|---|---|---|---|---|
| MNN-Pocket 0.3B | 129.7 tok/s | **250.5 tok/s (+93%)** | identik 64/64 | aktif |
| Bihaba-AI-1.7B | 81.2 tok/s | **109.3 tok/s (+35%)** | identik 64/64 | aktif |
| Qwythos-9B | 43.4 tok/s | 44.8 tok/s (+3%) | identik 64/64 | mati |
| MoE 35B-A3B | 83.1 tok/s | 91.0 tok/s (+10%) | **menyimpang di token 32** | mati |

- **Penyebab kernel lama lambat:** di konteks panjang, satu threadgroup per head memproses ribuan token secara berurutan.
- **Kenapa hanya transformer yang diaktifkan:** di engine qwen35 hanya ¼ layer yang memakai attention, sehingga untungnya kecil. Di MoE, greedy menyimpang di token 32. Kemungkinan itu seri fp16 yang diperbesar routing expert, tapi belum dibuktikan. Bench v1.6 kini mencetak selisih logit top-1/top-2 di titik beda untuk membuktikannya. Uji di qwen35: `BIHABA_ATTN_SPLIT=1024`.
- **Regresi chat** (`regress_chat.py --compare`, 4 model): **LULUS**, semua jawaban identik dengan versi stabil.
- **Kesalahan yang sempat lolos:** variabel `threadgroup` di fungsi pembantu (aturan Metal yang sudah tercatat). Sekarang dicegah oleh `TestMetalLint` di tes offline.
