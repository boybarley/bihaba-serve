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

## 10. bihaba-next Fase 0: instrumen overhead dispatch/barrier (1.7.0.dev1)

Tujuan: mengukur langsung, bukan menghitung, berapa biaya dispatch + barrier per token decode, sebelum
memutuskan Fase 1 (pipelining CPU/GPU atau fusi kernel). **Tidak ada kernel produksi yang berubah.**

- `set_pso()` menggantikan `setComputePipelineState` di jalur encode; `bar()` menghitung barrier.
  Dengan `ovh_mode = 0` (selalu, kecuali di dalam `bench_overhead`) perilakunya identik dengan v1.6.2.
- `null_kernel` (kernel kosong) dipakai dengan grid, buffer, dan barrier yang SAMA dengan kernel asli,
  agar biaya peluncuran threadgroup tetap terhitung.
- Opt-in: `BIHABA_BENCH_OVERHEAD=1 bihaba-serve bench --model … --tokens 128`. Tanpa env ini output bench sama.

| Ukuran | Cara | Arti |
|---|---|---|
| a | step normal (`GPUStartTime`/`GPUEndTime`) | waktu GPU total |
| b | semua kernel kosong, grid + barrier sama | batas atas overhead dispatch + barrier |
| c | seperti b, barrier dilewati | overhead dispatch saja (b − c = barrier) |
| d | jam CPU dari awal `run_step_m` sampai tepat sebelum `commit` | encode CPU yang bisa disembunyikan pipelining |

Kelas dibaca dan diputuskan TERPISAH — engine boleh sama (qwen35), arsitekturnya tidak:
- transformer kecil/menengah: semua layer attention, ~10 dispatch/layer; overhead dominan di model kecil.
- qwen35 dense: 3/4 layer DeltaNet + 1/4 attention, FFN dense; dekat batas bandwidth → overhead porsi kecil.
- qwen35 MoE: DeltaNet/attention + router + top-k expert + shared expert + combine (~13 dispatch/layer);
  byte per token kecil tetapi dispatch terbanyak → overhead porsi terbesar. Angka MoE selalu dicatat beserta topK.
Grid kernel expert tidak bergantung pada routing (ditentukan CPU sebelum encode), jadi mode kernel kosong tetap sah untuk MoE.
Setiap baris output `Overhead/token` mencantumkan kelasnya; hasil satu kelas tidak dipakai untuk kelas lain.

Keputusan per kelas: d besar → Fase 1a (pipelining N=1, greedy bit-identik); b − c besar → kurangi barrier;
b besar → fusi (Fase 1b, hanya yang bit-identik).

### 10.1 Hasil Fase 0 (M1 Max, decode M=1, 64 token)

| Kelas | dispatch / barrier | (a) GPU | (b) kosong | (c) tanpa barrier | (d) encode | celah total−GPU |
|---|---|---|---|---|---|---|
| transformer 0.3B | 284 / 227 | 2.397 ms | 0.440 (18%) | 0.087 | 0.119 ms | 0.402 ms (14%) |
| transformer 1.7B | 284 / 227 | 6.929 ms | 0.518 (7%) | 0.160 | 0.140 ms | 0.510 ms (7%) |
| qwen35 dense 9B | 260 / 259 | 21.082 ms | 0.616 (3%) | 0.260 | 0.148 ms | 0.544 ms (2.5%) |
| qwen35 MoE topK=4 | 524 / 443 | 9.629 ms | 0.909 (9%) | 0.256 | 0.250 ms | 0.648 ms (6%) |

Kesimpulan: biaya tetap dispatch+barrier 1.5–2.4 µs (bukan 7 µs perkiraan); encode CPU kecil (1–4%);
celah sinkron total−GPU adalah GPU menganggur → target pipelining. Kerugian terbesar ada di kerja kernel:
0.3B 1.96 ms vs ideal 0.53 ms, MoE 8.7 vs 4.6 ms. qwen35 dense tidak layak diubah di sisi dispatch.

### 10.2 Fase 0b (1.7.0.dev2): biaya barrier nyata + atribusi per kelompok kernel

- (e) `ovh_mode 3`: kernel asli, barrier dilewati (output salah, hanya waktu). a − e = biaya barrier
  termasuk *drain* (menunggu threadgroup paling lambat) yang tidak tertangkap kernel kosong.
- (f) `ovh_mode 4`: satu kelompok kernel dikosongkan per run. a − f = biaya kelompok itu.
  Kelompok dari nama fungsi Metal (dicatat di `make_pso`); lm_head dipisah lewat `cur_head`.
- Jalur produksi: `set_pso` langsung `setComputePipelineState` bila `ovh_mode == 0` (tanpa lookup).

### 10.3 Hasil Fase 0b (biaya barrier nyata a−e; atribusi f)

| Kelas | a−e (barrier + drain) | per barrier | terbesar di (f) |
|---|---|---|---|
| transformer 0.3B | 1.40 ms (59%) | 6.2 µs | proyeksi 28%, ffn 25%, rope 8%, lm_head 2% |
| transformer 1.7B | 1.58 ms (23%) | 7.0 µs | proyeksi 34%, ffn 35%, lm_head 11% |
| qwen35 dense | 2.63 ms (12%) | 10.2 µs | proyeksi 41%, ffn 42%, lm_head 9% |
| qwen35 MoE topK=4 | 3.35 ms (35%) | 7.6 µs | proyeksi 35%, expert 19%, lm_head 10%, deltanet 8% |

Config: 0.3B vocab 65536 (tied); MoE `num_experts_per_tok` bawaan = **8** (bench/acuan memakai 4 = mode cepat).
Pipelining N=1 bernilai kecil di chat nyata: draft n-gram step t+1 bergantung token t (CPU).

### 10.4 HASIL NEGATIF: fusi RMSNorm ke prolog proyeksi (dicoba, dibuang)

Kernel `qmv_*_rn`/`dense_swiglu_*_rn`: tiap threadgroup menghitung ulang RMSNorm dengan urutan reduksi identik
`rms_norm` → **bit-identik** (greedy 128/128, jawaban chat identik). Tetapi lebih lambat: 0.3B QKV +2% (noise),
FFN −3%; 1.7B QKV −3%, FFN −7%, keduanya −10% (`uji_semua` GAGAL). Sebab: prolog redundan di setiap threadgroup
(±1.500 TG gate/up per layer di 1.7B) lebih mahal dari barrier yang dihapus. Norm sekali tanpa dispatch terpisah
butuh atomik lintas TG → urutan tidak tetap → tidak bit-identik. Jangan dicoba ulang dalam bentuk ini.

## 11. 1.7.0.dev3: checkpoint per pesan (engine qwen35, dense + MoE)

- Default qwen35 kembali **kanonik** (`SESSION_CACHE_BY_ENGINE["qwen35"] = False`) → jawaban = acuan.
  `--session-cache` tetap memaksa sesi token-persis (TTFT ~0.3 s, tidak kanonik).
- Python mengirim `ckpt_at` = posisi tepat setelah header asisten terakhir (`<|im_start|>assistant\n`).
- Engine mengambil checkpoint (DeltaNet + KV s/d pos) di titik grid chunk g ≤ ckpt_at, dengan grid = kelipatan
  `prefill_chunk` dari awal segmen jalur penuh (prefix_len atau 0). Restore di g + prefill sisa memakai batas
  chunk yang sama dengan prefill penuh → **identik secara konstruksi** dengan jalur kanonik tanpa checkpoint.
- Pemilihan: live (bila seluruh cache prefix) ≥ checkpoint terpanjang > snapshot bake > reset.
- LRU di RAM, `BIHABA_CKPT_MB` (default 2048, 0 = mati); `clear_state` mengosongkan semua.
  Stats `done`: `ckpt` (posisi restore), `ckpts`, `ckpt_mb`. Engine transformer TIDAK diubah.

### 11.1 Hasil uji manual dan keputusan (1.7.0.dev4)

Qwythos-9B, Web Console, jalur kanonik + checkpoint: `reuse 512`, prefill 4.63 s untuk jawaban lama ±1000 token.
Checkpoint bekerja dan identik dengan acuan, tetapi jalur kanonik wajib mem-prefill ulang jawaban lama → jauh lebih
lambat dari sesi token-persis (~0.3 s). Keputusan: qwen35 kembali ke sesi token-persis sebagai default; checkpoint
tetap aktif sebagai cadangan jalur kanonik (sesi meleset, riwayat diedit, klien API, `regress_chat`/`uji_semua`).

## 12. Adu langsung dengan mlx-lm 0.32.2 (M1 Max yang sama, 2026-11-28)

| Kelas | Prefill bihaba / MLX | Decode bihaba / MLX |
|---|---|---|
| 1.7B | 1143 / 754 | 133 / 128 |
| Qwythos 9B | 276 / 184 | 46 / 55.4 |
| MoE top-k 8 (bawaan) | 289 / 433 | 86.9 / 70.1 |

(prompt MLX 570–1009 token, bihaba 512; decode MLX setelah prompt.) Celah: decode 9B (−17%) dan prefill MoE (−33%).
Kebijakan baru (disetujui pemilik): kernel baru boleh tidak bit-identik bila (1) tidak ada angka turun,
(2) selisih logit kecil (dicek otomatis), (3) jawaban chat diperiksa manual; lalu acuan baru disimpan,
acuan v1.6.2 diarsipkan. Target: unggul di semua kelas.

### 12.1 1.7.0.dev5: microbench kernel per tensor
`BIHABA_BENCH_KERNELS=1 bihaba-serve bench --model …` (qwen35) vs `scripts/bench_mlx_kernels.py <model>`
(lingkungan MLX): µs dan GB/s untuk ffn gate+up, ffn down, deltanet out_proj, lm_head pada tensor asli.
Menentukan apakah celah decode 9B ada di kernel qmv atau di tempat lain.

### 12.2 Hasil microbench Qwythos (M=1, tensor asli) dan 1.7.0.dev6

| Tensor | bihaba | MLX 0.32.5 |
|---|---|---|
| ffn gate+up (N=12288, K=4096) | 277.4 µs (204 GB/s, fused) | 208.6 µs (2 × qmm, ~271 GB/s) |
| ffn down (N=4096, K=12288) | 111.0 µs (255 GB/s) | 105.5 µs (268 GB/s) |
| deltanet out_proj (4096×4096) | 40.8 µs (231 GB/s) | 46.8 µs (201 GB/s) |
| lm_head (248320×4096) | 1985.6 µs (288 GB/s) | 1745.0 µs (328 GB/s) |

Celah terbesar: SwiGLU M=1 (−33%). Dugaan: kernel generik `dense_swiglu_m_impl` memakai `float ag[8], au[8]`
yang diindeks `m` runtime (pelajaran lama: array akumulator → spill). 1.7.0.dev6 menambah `dense_swiglu_{b}b_m1`
(akumulator skalar; loop, ekspresi, dan simd_sum sama → diharapkan bit-identik) untuk M=1 di kedua engine.
`BIHABA_SW1=0` → kernel lama. Microbench menampilkan lama vs m1.

### 12.3 Hasil 1.7.0.dev6 (SwiGLU M=1 skalar) — LULUS, bit-identik

gate+up 9B: 277.4 → 199.9 µs (283 GB/s, kini lebih cepat dari MLX 208.6 µs). Decode `uji_semua`:
0.3B 341→354, 1.7B 133→141, Qwythos 46→52, MoE 98 (tetap). Jawaban keempat kelas identik.
Sisa celah Qwythos vs MLX (52 vs 55.4): lm_head (1987 vs 1745 µs), down (109 vs 105.5 µs), dan proyeksi lain.

### 12.4 1.7.0.dev7: microbench diperluas
`enc_dn_inproj` / `enc_full_qkv` dipisah dari `enc_layer_attn` (isi tidak berubah) agar bisa diukur:
deltanet in_proj (qkv+z+a+b fused), attn q+k+v (fused), attn o_proj. Skrip MLX mengukur tensor yang sama.

### 12.5 Hasil microbench lengkap Qwythos (1.7.0.dev7)

Per kernel bihaba ≥ MLX untuk gate+up (199 vs 205 µs), in_proj DeltaNet fused (112 vs 166 µs untuk 4 qmm),
attn q+k+v fused (93 vs 155 µs), o_proj/out_proj (41 µs; angka MLX untuk tensor kecil tergelembung overhead).
Masih kalah: lm_head 1987 vs 1744 µs (−243 µs/token) dan down 108 vs 104 µs (−0.13 ms/token).
Sisa celah decode (52 vs 55.4 tok/s) = lm_head + down + celah sinkron CPU (~0.6 ms; MLX menyembunyikannya
dengan evaluasi asinkron).

### 12.6 1.7.0.dev8: `qmv_4b_f` (pola affine_qmv_fast MLX, MIT) — opt-in
`BIHABA_QMV4F` = 0 mati (default), 1 lm_head saja, 2 semua qmv 4-bit M=1. Tidak bit-identik (kebijakan B).
Microbench (`BIHABA_BENCH_KERNELS=1`) menampilkan waktu per tensor, selisih logit lm_head + argmax,
dan decode A/B 128 token (identik berapa token) untuk memutuskan default.

### 12.7 Hasil 1.7.0.dev8: `qmv_4b_f` (pola MLX) TIDAK lebih cepat di engine kita
lm_head 2010 → 2087 µs, down 110 → 112 µs, decode 52.3 → 51.8 tok/s (greedy tetap identik 128/128,
selisih logit maks 0.0078). Default tetap mati. Jadi keunggulan MLX di lm_head bukan dari bentuk kernelnya.
Hipotesis baru (1.7.0.dev9): bobot kita dipetakan zero-copy dari file (mmap + newBufferWithBytesNoCopy),
MLX memuat ke buffer Metal miliknya; `bw_probe` pada buffer Private baru = 322–327 GB/s ≈ lm_head MLX 328 GB/s,
sedangkan kernel kita di atas mmap mentok ±285 GB/s. Microbench kini menyalin tensor ke buffer Private/Shared
baru dan mengukur kernel yang sama.

### 12.8 Hasil 1.7.0.dev9: bobot mmap vs buffer Metal baru — TIDAK ada beda
lm_head private/shared 1986/1987 µs = mmap 1987 µs. Lokasi memori bukan penyebab selisih lm_head dengan MLX.

### 12.9 1.7.0.dev10: expert MoE berkelompok untuk prefill (4-bit, bit-identik)
Prefill MoE (M > 8) dulu: grid (N/8, top_k, M) → bobot expert dibaca ulang untuk setiap pasangan (token, slot).
Kini: `moe_group` (1 TG) mengelompokkan pasangan per expert secara deterministik (urut id pasangan), lalu
`moe_swiglu_g4` / `moe_down_g4` (grid (N/rows, E)) membaca bobot sekali per langkah K untuk ubin 4 token.
Urutan akumulasi per (pasangan, baris) sama dengan kernel lama → bit-identik. Decode (M=1) dan verifikasi
spekulatif (M ≤ 8) tetap jalur lama. `BIHABA_MOE_GROUP=0` → jalur lama. Bench: baris "MoE expert berkelompok".
