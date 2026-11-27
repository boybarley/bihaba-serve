// ============================================================================
// bihaba-serve — Native Apple Silicon Metal Engine (Standard GQA Transformers)
// Powered by Taalas-Metal Engine | Created by Boy Barley from Bihaba (Team Pasai Aceh)
//
// Supports: Qwen2.5 (0.5B-7B), Qwen3 (0.6B-8B, Bihaba-1.7B), Llama-3.2 (1B-3B), MNN-Pocket (0.3B)
//   • Direct Zero-Copy .safetensors Multi-Shard Loader (2b/3b/4b/6b/8b, Native BF16/FP16)
//   • Optional QKV Biases (Qwen2.5) + Optional Per-Head QK-Norm (Qwen3) + hd=64/128/256
//   • BM=8 Hardware MMA (M=4..8) + Adaptive Suffix Speculative Decoding (k=0..7)
//   • 2-Level Baked Prefix KV-Cache State (.tstate) + Exact GPU Top-64 Sampling
//
// Compile (~1s, 0 external deps):
//   clang++ -O3 -std=c++17 -fobjc-arc engines/transformer/main.mm \
//     -framework Metal -framework Foundation -o build/bihaba-transformer
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <mach-o/dyld.h>
#include <libgen.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// SIGUSR1 dari daemon Python = hentikan generasi saat ini (stop sequence cocok / klien putus).
// Handler hanya menyetel flag atomik; loop generasi memeriksanya tiap langkah.
static std::atomic<bool> g_cancel{false};
static void on_sigusr1(int) { g_cancel.store(true, std::memory_order_relaxed); }

struct NormParams      { int32_t dim; float eps; };
struct QMVParams       { int32_t K, N, gs_shift; };
struct QMVMultiParams  { int32_t M, K, N, gs_shift; };
struct TrRopeParams    { int32_t M, n_heads, n_kv_heads, head_dim, pos, max_seq, has_qkv_bias, has_qk_norm; float eps; };
struct TrGQAParams     { int32_t M, n_heads, n_kv_heads, pos, max_seq; };
struct AttnSplitParams  { int32_t M, n_heads, n_kv_heads, pos, max_seq, n_blk; };
// Repetition penalty PER BARIS (lihat build_rep_rows). setBytes ≤ 4 KB: 8 baris × 65 int = 2.1 KB.
constexpr int REP_WIN = 64, REP_STRIDE = REP_WIN + 1;
struct RepRowsParams   { int32_t M, K_len, stride; float penalty; int32_t rows[8 * REP_STRIDE]; };

struct BufSlice { id<MTLBuffer> buf = nil; uint64_t off = 0; };

struct QTensor {
    id<MTLBuffer> w_buf = nil, s_buf = nil, b_buf = nil;
    uint64_t w = 0, s = 0, b = 0;
    int bits = 4, gs = 64, gs_shift = 6, N = 0, K = 0;
};

struct LayerMeta {
    BufSlice in_norm, post_norm;
    QTensor q, k, v, o;
    BufSlice q_norm, k_norm;
    BufSlice q_bias, k_bias, v_bias;
    bool has_qkv_bias = false;
    bool has_qk_norm  = false;
    QTensor gate, up, down;
};

struct MappedFile {
    int fd = -1;
    void* mapped = nullptr;
    size_t fsize = 0, map_len = 0;
    id<MTLBuffer> mtl_buf = nil;
    uint64_t data_off = 0;
    NSDictionary* header = nil;
};

struct SamplingConfig {
    float temperature = 0.0f;
    float top_p       = 0.95f;
    int   top_k       = 40;
    float min_p       = 0.05f;
    float rep_pen     = 1.0f;
    uint32_t seed     = 0;
};

struct StateSnapshot {
    std::vector<int32_t> ids;
    int pos = 0;
    std::vector<std::vector<uint8_t>> k_data;
    std::vector<std::vector<uint8_t>> v_data;

    bool valid() const { return pos > 0 && !ids.empty(); }
    size_t bytes() const {
        size_t s = 0;
        for (const auto& v : k_data) s += v.size();
        for (const auto& v : v_data) s += v.size();
        return s;
    }
};

static inline float bf16_to_f32(uint16_t u) {
    uint32_t x = uint32_t(u) << 16; float f = 0.0f; memcpy(&f, &x, 4); return f;
}
static inline uint16_t f32_to_f16_bits(float f) {
    __fp16 h = (__fp16)f; uint16_t u = 0; memcpy(&u, &h, 2); return u;
}

struct FastSuffixSpeculator {
    std::vector<int32_t> toks;
    std::unordered_map<uint64_t, std::vector<int32_t>> idx2, idx3, idx4;

    static uint64_t h2(int32_t a, int32_t b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }
    static uint64_t h3(int32_t a, int32_t b, int32_t c) {
        return (uint64_t(uint32_t(a)) * 1315423911ULL) ^ (uint64_t(uint32_t(b)) * 2654435761ULL) ^ uint32_t(c);
    }
    static uint64_t h4(int32_t a, int32_t b, int32_t c, int32_t d) {
        return h3(a, b, c) * 1099511628211ULL ^ uint32_t(d);
    }
    void push(int32_t t) {
        toks.push_back(t);
        int n = int(toks.size());
        auto add = [](std::unordered_map<uint64_t, std::vector<int32_t>>& mp, uint64_t k, int pos) {
            auto& v = mp[k]; v.push_back(pos);
            if (v.size() > 12) v.erase(v.begin());
        };
        if (n >= 2) add(idx2, h2(toks[n-2], toks[n-1]), n);
        if (n >= 3) add(idx3, h3(toks[n-3], toks[n-2], toks[n-1]), n);
        if (n >= 4) add(idx4, h4(toks[n-4], toks[n-3], toks[n-2], toks[n-1]), n);
    }
    // Orde n-gram tertinggi yang punya kecocokan valid (= orde yang dipakai propose); 0 = tidak ada.
    int match_order() const {
        const int n = int(toks.size());
        auto has = [&](const std::unordered_map<uint64_t, std::vector<int32_t>>& mp, uint64_t k, int order) {
            auto it = mp.find(k);
            if (it == mp.end()) return false;
            for (int pos : it->second) {
                if (pos >= n) continue;
                bool ok = true;
                for (int j = 0; j < order; ++j) if (toks[pos - order + j] != toks[n - order + j]) { ok = false; break; }
                if (ok) return true;
            }
            return false;
        };
        if (n >= 4 && has(idx4, h4(toks[n-4], toks[n-3], toks[n-2], toks[n-1]), 4)) return 4;
        if (n >= 3 && has(idx3, h3(toks[n-3], toks[n-2], toks[n-1]), 3)) return 3;
        if (n >= 2 && has(idx2, h2(toks[n-2], toks[n-1]), 2)) return 2;
        return 0;
    }

    std::vector<int32_t> propose(int max_len) const {
        std::vector<int32_t> out;
        int n = int(toks.size());
        if (max_len <= 0 || n < 2) return out;
        auto scan = [&](const std::unordered_map<uint64_t, std::vector<int32_t>>& mp, uint64_t k, int order) -> bool {
            auto it = mp.find(k);
            if (it == mp.end()) return false;
            const auto& v = it->second;
            for (int i = int(v.size()) - 1; i >= 0; --i) {
                int pos = v[i];
                if (pos >= n) continue;
                bool ok = true;
                for (int j = 0; j < order; ++j)
                    if (toks[pos - order + j] != toks[n - order + j]) { ok = false; break; }
                if (!ok) continue;
                int take = std::min(max_len, n - pos);
                if (take > 0) { out.assign(toks.begin() + pos, toks.begin() + pos + take); return true; }
            }
            return false;
        };
        if (n >= 4 && scan(idx4, h4(toks[n-4], toks[n-3], toks[n-2], toks[n-1]), 4)) return out;
        if (n >= 3 && scan(idx3, h3(toks[n-3], toks[n-2], toks[n-1]), 3)) return out;
        if (n >= 2 && scan(idx2, h2(toks[n-2], toks[n-1]), 2)) return out;
        return out;
    }
};

// ─── Fase 0b: kelompok kernel untuk atribusi waktu (dari nama fungsi Metal, dicatat di make_pso) ───
enum { G_NORM, G_PROJ, G_FFN, G_ROPE, G_ATTN, G_DN, G_ROUTER, G_EXPERT, G_COMBINE, G_HEAD, G_OTHER, G_N };
static const char* const G_NAMES[G_N] = {"norm", "proyeksi qmv", "ffn swiglu", "qknorm/rope", "attention",
                                         "deltanet", "moe router", "moe expert", "moe combine", "lm_head", "lainnya"};
static std::unordered_map<const void*, int> g_pso_grp;
static int kernel_group(const char* fn) {
    auto pre = [&](const char* p) { return strncmp(fn, p, strlen(p)) == 0; };
    if (pre("rms_norm")) return G_NORM;
    if (pre("dense_swiglu")) return G_FFN;
    if (pre("qmv_") || pre("qmm_") || pre("q35_qkv_proj") || pre("deltanet_in_proj")) return G_PROJ;
    if (pre("q35_qknorm_rope") || pre("tr_qknorm_rope")) return G_ROPE;
    if (pre("gqa_attention") || pre("attn_split") || pre("attn_merge")) return G_ATTN;
    if (pre("deltanet_")) return G_DN;
    if (pre("moe_router") || pre("moe_topk")) return G_ROUTER;
    if (pre("moe_swiglu") || pre("moe_down")) return G_EXPERT;
    if (pre("moe_combine")) return G_COMBINE;
    return G_OTHER;
}

static id<MTLComputePipelineState> make_pso(
    id<MTLDevice> dev, id<MTLLibrary> lib, const char* fn, bool add_res = false, bool is_bf16 = false)
{
    NSError* err = nil;
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    [fc setConstantValue:&add_res type:MTLDataTypeBool atIndex:0];
    [fc setConstantValue:&is_bf16 type:MTLDataTypeBool atIndex:1];
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn] constantValues:fc error:&err];
    if (!f) { fprintf(stderr, "Missing/invalid Metal fn %s: %s\n", fn, [[err localizedDescription] UTF8String]); exit(1); }
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!pso) { fprintf(stderr, "PSO error %s: %s\n", fn, [[err localizedDescription] UTF8String]); exit(1); }
    g_pso_grp[(__bridge const void*)pso] = kernel_group(fn);
    return pso;
}

// Baris m memverifikasi token SETELAH (riwayat + draft 1..m). Himpunan penalty-nya = REP_WIN token
// terakhir dari (hist ++ batch[1..m]), unik. Dengan himpunan yang SAMA untuk semua baris (versi lama),
// spekulasi + penalty ≠ decode berurutan + penalty. penalty = 1 → tidak aktif.
static RepRowsParams build_rep_rows(const std::vector<int32_t>& hist, const int32_t* batch, int M, int vocab, float pen) {
    RepRowsParams rp{};
    rp.M = M; rp.K_len = vocab; rp.stride = REP_STRIDE; rp.penalty = pen;
    if (!(pen > 0.0f && (pen > 1.0001f || pen < 0.9999f))) return rp;
    for (int m = 0; m < M; ++m) {
        int32_t* row = rp.rows + m * REP_STRIDE;
        const int total = int(hist.size()) + m;
        int n = 0;
        for (int i = std::max(0, total - REP_WIN); i < total; ++i) {
            const int32_t t = (i < int(hist.size())) ? hist[size_t(i)] : batch[1 + i - int(hist.size())];
            if (t < 0 || t >= vocab) continue;
            bool dup = false;
            for (int j = 0; j < n; ++j) if (row[1 + j] == t) { dup = true; break; }
            if (!dup) row[1 + n++] = t;
        }
        row[0] = n;
    }
    return rp;
}
static inline bool rep_active(const RepRowsParams& rp) { return rp.penalty > 0.0f && (rp.penalty > 1.0001f || rp.penalty < 0.9999f); }

// Peluang draft diterima per token, PER ORDE n-gram, sebagai proses geometrik:
//   α = token diterima / (token diterima + penolakan).
// Estimator lama (EMA dari acc/(M−1)) mencatat 5/7 sebagai 0.71 padahal draft ke-6 yang ditolak berarti
// peluang per token ≈ 5/6, dan satu penolakan langsung memangkas k. Di hardcore-on-mac perbaikan ini
// menaikkan mode adaptif 609 → 780 tok/s (Qwen2.5 1.5B, edit).
struct OrderAlpha {
    double A[5] = {0, 0, 0, 0, 0}, R[5] = {0, 0, 0, 0, 0};
    static double prior(int o) { return o <= 1 ? 0.4 : (o == 2 ? 0.6 : 0.85); }
    double alpha(int o) const {
        const double a = (A[o] + 4.0 * prior(o)) / (A[o] + R[o] + 4.0);
        return std::min(0.995, std::max(0.05, a));
    }
    void update(int o, int acc, bool rejected) { A[o] = 0.9 * A[o] + acc; R[o] = 0.9 * R[o] + (rejected ? 1.0 : 0.0); }
};

static inline uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

// Kernel mengasumsikan grup kuantisasi 32/64 dan K kelipatan 16 (3-bit: 32). Model di luar itu akan
// menghasilkan teks rusak tanpa error, jadi tolak sejak load.
// Ambang M untuk jalur MMA. 4/2-bit: dekuantisasi bit-trick murah → MMA menang mulai M=4.
// 3/6/8-bit: dekuantisasi generik lebih mahal, biayanya ~tetap per M (M1 Max: 6-bit n4=23.1ms MMA vs 18.6ms qmv,
// n6=22.8ms vs 25.3ms) → MMA baru menang mulai M=6.
static inline int mma_min_m(int bits) { return (bits == 4 || bits == 2) ? 4 : 6; }

static void check_qtensor(const char* name, int bits, int gs, int K, int N) {
    const bool ok = (gs == 32 || gs == 64) && K > 0 && N > 0 && (K % gs) == 0 && (K % (bits == 3 ? 32 : 16)) == 0;
    if (!ok) {
        fprintf(stderr, "✗ Tensor %s tidak didukung (bits=%d gs=%d K=%d N=%d): butuh gs 32/64 dan K kelipatan 16\n",
                name, bits, gs, K, N);
        exit(1);
    }
}

struct TransformerEngine {
    static constexpr int MAX_M = 8;     // batas verifikasi spekulatif (draft ≤ 7)
    static constexpr int MAX_P = 128;   // batas chunk prefill (GEMM: grid y = M/32 → GPU penuh terisi)
    int prefill_chunk = MAX_P;          // --prefill-chunk (1..128); 8 = perilaku lama
    uint64_t model_fp = 0;              // sidik jari model untuk validasi .tstate

    std::vector<MappedFile> mapped_files;
    std::vector<id<MTLBuffer>> fallback_bufs;
    size_t total_mapped_bytes = 0;
    bool scales_bf16 = false;

    int n_layers=0, dim=0, vocab=0, n_heads=0, n_kv_heads=0, head_dim=128, dense_inter=0, max_seq=4096;
    int q_full_dim=0, kv_full_dim=0;
    float rms_eps = 1e-6f;
    std::vector<int32_t> eos_ids;
    std::string arch_name, model_dir;

    QTensor embed, lm_head;
    id<MTLComputePipelineState> pso_bw = nil;
    // Attention decode split-sequence: dipakai bila konteks ≥ attn_split_min (env BIHABA_ATTN_SPLIT=0 → mati)
    id<MTLComputePipelineState> pso_attn_split = nil, pso_attn_merge = nil;
    id<MTLBuffer> b_part_ml = nil, b_part_o = nil;
    int attn_split_min = 1024;
    static constexpr int ATTN_MAX_BLK = 32;
    double last_gpu_ms = 0.0;           // durasi GPU command buffer run_step terakhir

    // ─── Fase 0: instrumen overhead (ovh_mode=0 di luar bench_overhead → jalur produksi tidak berubah) ───
    //   ovh_mode 1: semua kernel diganti null_kernel (grid, buffer, barrier sama) → sisa = biaya dispatch+barrier
    //   ovh_mode 2: seperti 1 tetapi barrier dilewati → sisa = biaya dispatch saja
    int ovh_mode = 0;
    id<MTLComputePipelineState> pso_null = nil;
    long n_disp = 0, n_bar = 0;         // penghitung kumulatif (dibaca oleh bench_overhead)
    double last_enc_ms = 0.0;           // waktu CPU: awal run_step_m → tepat sebelum commit
    //   ovh_mode 3: kernel ASLI tanpa barrier (output salah; hanya waktu) → a−e = biaya barrier nyata termasuk drain
    //   ovh_mode 4: hanya kelompok ovh_grp yang dikosongkan → a−f = biaya kelompok itu
    int ovh_grp = -1;
    bool ovh_count = false, cur_head = false;
    long n_disp_grp[G_N] = {0};
    inline void set_pso(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p) {
        ++n_disp;
        if (ovh_mode == 0 && !ovh_count) { [e setComputePipelineState:p]; return; }   // jalur produksi
        int g = G_OTHER;
        if (cur_head) g = G_HEAD;
        else { auto it = g_pso_grp.find((__bridge const void*)p); if (it != g_pso_grp.end()) g = it->second; }
        ++n_disp_grp[g];
        const bool nul = ovh_mode == 1 || ovh_mode == 2 || (ovh_mode == 4 && g == ovh_grp);
        [e setComputePipelineState:(nul ? pso_null : p)];
    }

    // ─── Roofline: byte yang WAJIB dibaca per token decode vs bandwidth memori terukur ───
    static double qt_bytes(const QTensor& q) {
        if (q.N <= 0 || q.K <= 0) return 0.0;
        return double(q.N) * double(q.K) * q.bits / 8.0 + 2.0 * double(q.N) * double(q.K >> q.gs_shift) * 2.0;
    }
    double probe_bandwidth_gbs() {
        if (!pso_bw) return 0.0;
        const NSUInteger bytes = NSUInteger(1) << 30;            // 1 GiB: jauh lebih besar dari SLC
        id<MTLBuffer> src = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        const uint32_t threads = 64 * 1024;
        id<MTLBuffer> out = [dev newBufferWithLength:threads * sizeof(float) options:MTLResourceStorageModePrivate];
        if (!src || !out) return 0.0;
        const uint32_t n4 = uint32_t(bytes / 16);
        double best = 1e30;
        for (int it = 0; it < 6; ++it) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso_bw];
            [e setBuffer:src offset:0 atIndex:0];
            [e setBuffer:out offset:0 atIndex:1];
            [e setBytes:&n4 length:sizeof(n4) atIndex:2];
            [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            const double ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
            if (it > 0 && ms > 0 && ms < best) best = ms;           // iterasi 0 = pemanasan
        }
        return best < 1e29 ? double(bytes) / (best * 1e6) : 0.0;
    }

    double decode_bytes_per_token(int pos) const {
        double b = qt_bytes(lm_head);
        for (int l = 0; l < n_layers; ++l) {
            const LayerMeta& L = layers[size_t(l)];
            b += qt_bytes(L.q) + qt_bytes(L.k) + qt_bytes(L.v) + qt_bytes(L.o);
            b += qt_bytes(L.gate) + qt_bytes(L.up) + qt_bytes(L.down);
            b += double(pos) * n_kv_heads * head_dim * 2.0 * 2.0;
        }
        return b;
    }
    BufSlice cos_s, sin_s, fnorm_s, zero_dummy;
    std::vector<LayerMeta> layers;

    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> pso_embed=nil, pso_rms=nil, pso_rep=nil, pso_argmax=nil, pso_topk64=nil;
    id<MTLComputePipelineState> pso_qmv1[9], pso_qmv1_add[9];
    id<MTLComputePipelineState> pso_qmvm[9], pso_qmvm_add[9], pso_sw[9];
    id<MTLComputePipelineState> pso_qmm4_mma8=nil, pso_qmm4_mma8_add=nil, pso_qmm2_mma8=nil, pso_qmm2_mma8_add=nil;
    id<MTLComputePipelineState> pso_sw4_mma8=nil, pso_sw2_mma8=nil;
    // MMA generik per bit-width: [bits][0=mma8,1=mmaP][add_res] dan swiglu [bits][mma8/mmaP]. nil = tidak ada.
    id<MTLComputePipelineState> pso_mm[9][2][2] = {}, pso_swm[9][2] = {};
    id<MTLComputePipelineState> pso_qmm4_mmaP=nil, pso_qmm4_mmaP_add=nil, pso_qmm2_mmaP=nil, pso_qmm2_mmaP_add=nil;
    id<MTLComputePipelineState> pso_sw4_mmaP=nil, pso_sw2_mmaP=nil;
    id<MTLComputePipelineState> pso_qk_rope=nil, pso_gqa=nil;

    id<MTLBuffer> b_tok=nil, b_x=nil, b_xn=nil;
    id<MTLBuffer> b_q=nil, b_k=nil, b_v=nil, b_att=nil, b_dense_h=nil;
    id<MTLBuffer> b_logits=nil, b_out_tok=nil, b_top64_ids=nil, b_top64_vals=nil, b_top64_flag=nil;

    std::vector<id<MTLBuffer>> k_caches, v_caches;
    std::vector<int32_t> cached_ids;
    StateSnapshot ram_snap;

    bool is_eos(int32_t t) const {
        for (int32_t e : eos_ids) if (t == e) return true;
        return false;
    }

    ~TransformerEngine() {
        for (auto& mf : mapped_files) {
            if (mf.mapped && mf.mapped != MAP_FAILED) munmap(mf.mapped, mf.fsize);
            if (mf.fd >= 0) close(mf.fd);
        }
    }

    bool mmap_file_mtl(const std::string& fpath, bool willneed, MappedFile& out_mf) {
        out_mf.fd = open(fpath.c_str(), O_RDONLY);
        if (out_mf.fd < 0) { perror(("open " + fpath).c_str()); return false; }
        struct stat st; fstat(out_mf.fd, &st);
        out_mf.fsize = size_t(st.st_size);
        size_t pg = size_t(getpagesize());
        out_mf.map_len = (out_mf.fsize + pg - 1) & ~(pg - 1);
        out_mf.mapped = mmap(nullptr, out_mf.fsize, PROT_READ, MAP_PRIVATE, out_mf.fd, 0);
        if (out_mf.mapped == MAP_FAILED) { perror(("mmap " + fpath).c_str()); return false; }
        madvise(out_mf.mapped, out_mf.fsize, willneed ? MADV_WILLNEED : MADV_NORMAL);
        out_mf.mtl_buf = [dev newBufferWithBytesNoCopy:out_mf.mapped length:out_mf.map_len
                                              options:MTLResourceStorageModeShared deallocator:nil];
        if (!out_mf.mtl_buf) { fprintf(stderr, "newBufferWithBytesNoCopy failed on %s\n", fpath.c_str()); return false; }
        total_mapped_bytes += out_mf.fsize;
        return true;
    }

    bool load(const std::string& dir_path, bool force_willneed = false, int req_max_seq = 4096) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        model_dir = dir_path;
        max_seq = req_max_seq;

        auto read_json = [&](const std::string& rel) -> NSDictionary* {
            NSString* p = [NSString stringWithUTF8String:(dir_path + "/" + rel).c_str()];
            NSData* d = [NSData dataWithContentsOfFile:p];
            if (!d) return nil;
            return [NSJSONSerialization JSONObjectWithData:d options:0 error:nil];
        };

        NSDictionary* cfg = read_json("config.json");
        if (NSData* cfg_raw = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:(dir_path + "/config.json").c_str()]])
            model_fp = fnv1a(1469598103934665603ULL, [cfg_raw bytes], [cfg_raw length]);
        if (!cfg) { fprintf(stderr, "Cannot read %s/config.json\n", dir_path.c_str()); return false; }
        NSDictionary* tc = cfg[@"text_config"] ?: cfg;

        arch_name   = tc[@"model_type"] ? [tc[@"model_type"] UTF8String] : "transformer";
        n_layers    = [tc[@"num_hidden_layers"] intValue];
        dim         = [tc[@"hidden_size"] intValue];
        vocab       = [tc[@"vocab_size"] intValue];
        rms_eps     = tc[@"rms_norm_eps"] ? [tc[@"rms_norm_eps"] floatValue] : 1e-6f;
        n_heads     = [tc[@"num_attention_heads"] intValue];
        n_kv_heads  = tc[@"num_key_value_heads"] ? [tc[@"num_key_value_heads"] intValue] : n_heads;
        head_dim    = tc[@"head_dim"] ? [tc[@"head_dim"] intValue] : (dim / n_heads);
        {   // Konteks di atas panjang latih model (tanpa rope_scaling) → output rusak tanpa error. Batasi di sini;
            // daemon lalu menolak percakapan yang terlalu panjang dengan pesan jelas. Override: BIHABA_IGNORE_MAX_POS=1
            const int mpe = tc[@"max_position_embeddings"] ? [tc[@"max_position_embeddings"] intValue] : 0;
            const bool scaled = [tc[@"rope_scaling"] isKindOfClass:[NSDictionary class]];
            if (mpe >= 64 && !scaled && max_seq > mpe && !getenv("BIHABA_IGNORE_MAX_POS")) {
                fprintf(stderr, "  [note] --max-seq %d > panjang latih model (max_position_embeddings=%d) → dibatasi ke %d\n",
                        max_seq, mpe, mpe);
                max_seq = mpe;
            }
        }
        dense_inter = [tc[@"intermediate_size"] intValue];
        q_full_dim  = n_heads * head_dim;
        kv_full_dim = n_kv_heads * head_dim;

        if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
            fprintf(stderr, "Unsupported head_dim=%d (supported: 64, 128, 256)\n", head_dim);
            return false;
        }

        auto add_eos = [&](id val) {
            if ([val isKindOfClass:[NSArray class]]) {
                for (NSNumber* n in (NSArray*)val) {
                    int32_t v = [n intValue];
                    if (std::find(eos_ids.begin(), eos_ids.end(), v) == eos_ids.end()) eos_ids.push_back(v);
                }
            } else if ([val isKindOfClass:[NSNumber class]]) {
                int32_t v = [(NSNumber*)val intValue];
                if (std::find(eos_ids.begin(), eos_ids.end(), v) == eos_ids.end()) eos_ids.push_back(v);
            }
        };
        add_eos(cfg[@"eos_token_id"]); add_eos(tc[@"eos_token_id"]);
        if (NSDictionary* gc = read_json("generation_config.json")) add_eos(gc[@"eos_token_id"]);
        if (NSDictionary* tj = read_json("tokenizer.json")) {
            for (NSDictionary* at in tj[@"added_tokens"]) {
                NSString* c = at[@"content"];
                if ([c isEqualToString:@"<|im_end|>"] || [c isEqualToString:@"<|endoftext|>"] || [c isEqualToString:@"<|eot_id|>"])
                    add_eos(at[@"id"]);
            }
        }
        if (eos_ids.empty()) eos_ids = {151645, 151643, 128009};

        NSDictionary* idx = read_json("model.safetensors.index.json");
        NSDictionary* wmap = nil;
        std::unordered_map<std::string, int> shard_idx_map;

        auto open_shard = [&](const std::string& fname) -> int {
            auto it = shard_idx_map.find(fname);
            if (it != shard_idx_map.end()) return it->second;
            MappedFile mf;
            if (!mmap_file_mtl(dir_path + "/" + fname, force_willneed, mf)) exit(1);
            uint64_t hlen = 0; memcpy(&hlen, mf.mapped, 8);
            mf.data_off = 8 + hlen;
            NSData* hdata = [NSData dataWithBytesNoCopy:(uint8_t*)mf.mapped + 8 length:hlen freeWhenDone:NO];
            mf.header = [NSJSONSerialization JSONObjectWithData:hdata options:0 error:nil];
            if (!mf.header) { fprintf(stderr, "Bad safetensors header in %s\n", fname.c_str()); exit(1); }
            int id_i = int(mapped_files.size());
            mapped_files.push_back(mf);
            shard_idx_map[fname] = id_i;
            return id_i;
        };

        if (idx && idx[@"weight_map"]) {
            wmap = idx[@"weight_map"];
        } else {
            int s0 = open_shard("model.safetensors");
            NSMutableDictionary* m = [NSMutableDictionary new];
            for (NSString* k in mapped_files[s0].header) {
                if (![k isEqualToString:@"__metadata__"]) m[k] = @"model.safetensors";
            }
            wmap = m;
        }

        auto has_key = [&](const std::string& k) -> bool {
            return wmap[[NSString stringWithUTF8String:k.c_str()]] != nil;
        };
        std::string pfx = has_key("language_model.model.norm.weight") ? "language_model." : "";

        struct RawRef { int shard; uint64_t off, len; std::string dtype; std::vector<int64_t> shape; };
        auto find_tensor = [&](const std::string& k) -> RawRef {
            NSString* nk = [NSString stringWithUTF8String:k.c_str()];
            NSString* fn = wmap[nk];
            if (!fn) { fprintf(stderr, "Missing safetensors key: %s\n", k.c_str()); exit(1); }
            int si = open_shard([fn UTF8String]);
            NSDictionary* info = mapped_files[si].header[nk];
            NSArray* offs = info[@"data_offsets"];
            uint64_t s = [offs[0] unsignedLongLongValue], e = [offs[1] unsignedLongLongValue];
            RawRef r{si, mapped_files[si].data_off + s, e - s, [info[@"dtype"] UTF8String], {}};
            for (NSNumber* d in info[@"shape"]) r.shape.push_back([d longLongValue]);
            return r;
        };

        std::vector<uint8_t> small_blob(256, 0);  // offset 0 = 256-byte zero dummy!
        auto append_small = [&](const void* src, size_t nbytes) -> uint64_t {
            size_t cur = (small_blob.size() + 255) & ~size_t(255);
            small_blob.resize(cur + nbytes);
            memcpy(small_blob.data() + cur, src, nbytes);
            return uint64_t(cur);
        };

        auto load_small_f16 = [&](const std::string& k) -> uint64_t {
            RawRef r = find_tensor(k);
            const uint8_t* ptr = (const uint8_t*)mapped_files[r.shard].mapped + r.off;
            size_t n = 1; for (auto d : r.shape) n *= size_t(d);
            std::vector<uint16_t> out(n);
            if (r.dtype == "F16") memcpy(out.data(), ptr, n * 2);
            else if (r.dtype == "BF16") {
                const uint16_t* p16 = (const uint16_t*)ptr;
                for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16_bits(bf16_to_f32(p16[i]));
            } else {
                const float* pf = (const float*)ptr;
                for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16_bits(pf[i]);
            }
            return append_small(out.data(), n * 2);
        };

        // Full Split-Half RoPE (supports Standard, Llama-3 frequency scaling, and YaRN!)
        double theta = tc[@"rope_theta"] ? [tc[@"rope_theta"] doubleValue] : 1000000.0;
        int half_r = head_dim / 2;
        std::vector<double> inv_freq(half_r);
        for (int i = 0; i < half_r; ++i) inv_freq[i] = 1.0 / std::pow(theta, double(2 * i) / double(head_dim));
        double mscale = 1.0;

        NSDictionary* rs_cfg = tc[@"rope_scaling"];
        if (rs_cfg && [rs_cfg isKindOfClass:[NSDictionary class]]) {
            NSString* rtype = rs_cfg[@"rope_type"] ?: rs_cfg[@"type"];
            if (rtype && [rtype isEqualToString:@"llama3"]) {
                double factor = rs_cfg[@"factor"] ? [rs_cfg[@"factor"] doubleValue] : 8.0;
                double low_f  = rs_cfg[@"low_freq_factor"]  ? [rs_cfg[@"low_freq_factor"] doubleValue]  : 1.0;
                double high_f = rs_cfg[@"high_freq_factor"] ? [rs_cfg[@"high_freq_factor"] doubleValue] : 4.0;
                double old_ctx = rs_cfg[@"original_max_position_embeddings"] ? [rs_cfg[@"original_max_position_embeddings"] doubleValue] : 8192.0;
                double low_w = old_ctx / low_f, high_w = old_ctx / high_f;
                for (int i = 0; i < half_r; ++i) {
                    double wlen = 2.0 * M_PI / inv_freq[i];
                    if (wlen > low_w) inv_freq[i] /= factor;
                    else if (wlen >= high_w) {
                        double sm = (old_ctx / wlen - low_f) / (high_f - low_f);
                        inv_freq[i] = (1.0 - sm) * inv_freq[i] / factor + sm * inv_freq[i];
                    }
                }
            }
        }
        std::vector<uint16_t> cos_f16(size_t(max_seq) * half_r), sin_f16(size_t(max_seq) * half_r);
        for (int p_i = 0; p_i < max_seq; ++p_i) {
            for (int j = 0; j < half_r; ++j) {
                double f = double(p_i) * inv_freq[j];
                cos_f16[size_t(p_i) * half_r + j] = f32_to_f16_bits(float(std::cos(f) * mscale));
                sin_f16[size_t(p_i) * half_r + j] = f32_to_f16_bits(float(std::sin(f) * mscale));
            }
        }
        uint64_t cos_off_sm = append_small(cos_f16.data(), cos_f16.size() * 2);
        uint64_t sin_off_sm = append_small(sin_f16.data(), sin_f16.size() * 2);
        uint64_t fn_off_sm  = load_small_f16(pfx + "model.norm.weight");

        int misaligned_copies = 0;
        auto bind_zero_copy = [&](const RawRef& r, size_t req_align) -> BufSlice {
            if ((r.off % req_align) == 0) return {mapped_files[r.shard].mtl_buf, r.off};
            misaligned_copies++;
            id<MTLBuffer> fb = [dev newBufferWithBytes:(const uint8_t*)mapped_files[r.shard].mapped + r.off
                                                length:r.len options:MTLResourceStorageModeShared];
            fallback_bufs.push_back(fb);
            return {fb, 0};
        };

        bool dtype_detected = false;
        auto qt_from_st = [&](const std::string& st_pfx, int K_expect) -> QTensor {
            RawRef rw = find_tensor(st_pfx + ".weight");
            RawRef rs = find_tensor(st_pfx + ".scales");
            RawRef rb = find_tensor(st_pfx + ".biases");
            if (!dtype_detected) { scales_bf16 = (rs.dtype == "BF16"); dtype_detected = true; }
            int kw = int(rw.shape.back()), kg = int(rs.shape.back());
            int N  = int(rw.shape[rw.shape.size() - 2]);
            int gs = (kg > 0) ? K_expect / kg : 0;
            int bits = (kw * 32) / K_expect;
            check_qtensor(st_pfx.c_str(), bits, (kg > 0 && K_expect % kg == 0) ? gs : 0, K_expect, N);
            if (bits != 2 && bits != 3 && bits != 4 && bits != 6 && bits != 8) {
                fprintf(stderr, "Unsupported bits=%d for %s\n", bits, st_pfx.c_str()); exit(1);
            }
            BufSlice sw = bind_zero_copy(rw, 4), ss = bind_zero_copy(rs, 2), sb = bind_zero_copy(rb, 2);
            return QTensor{sw.buf, ss.buf, sb.buf, sw.off, ss.off, sb.off, bits, gs, (gs == 32) ? 5 : 6, N, K_expect};
        };

        embed   = qt_from_st(pfx + "model.embed_tokens", dim);
        lm_head = has_key(pfx + "lm_head.weight") ? qt_from_st(pfx + "lm_head", dim) : embed;

        struct SmallRef { BufSlice* dst; uint64_t off; };
        std::vector<SmallRef> small_refs;
        small_refs.push_back({&zero_dummy, 0});
        small_refs.push_back({&cos_s, cos_off_sm});
        small_refs.push_back({&sin_s, sin_off_sm});
        small_refs.push_back({&fnorm_s, fn_off_sm});

        layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            std::string sp = pfx + "model.layers." + std::to_string(i);
            std::string sa = sp + ".self_attn", mp = sp + ".mlp";
            auto& L = layers[i];
            small_refs.push_back({&L.in_norm,   load_small_f16(sp + ".input_layernorm.weight")});
            small_refs.push_back({&L.post_norm, load_small_f16(sp + ".post_attention_layernorm.weight")});
            L.q = qt_from_st(sa + ".q_proj", dim);
            L.k = qt_from_st(sa + ".k_proj", dim);
            L.v = qt_from_st(sa + ".v_proj", dim);
            L.o = qt_from_st(sa + ".o_proj", q_full_dim);

            L.has_qk_norm = has_key(sa + ".q_norm.weight") && has_key(sa + ".k_norm.weight");
            small_refs.push_back({&L.q_norm, L.has_qk_norm ? load_small_f16(sa + ".q_norm.weight") : 0});
            small_refs.push_back({&L.k_norm, L.has_qk_norm ? load_small_f16(sa + ".k_norm.weight") : 0});

            L.has_qkv_bias = has_key(sa + ".q_proj.bias") && has_key(sa + ".k_proj.bias") && has_key(sa + ".v_proj.bias");
            small_refs.push_back({&L.q_bias, L.has_qkv_bias ? load_small_f16(sa + ".q_proj.bias") : 0});
            small_refs.push_back({&L.k_bias, L.has_qkv_bias ? load_small_f16(sa + ".k_proj.bias") : 0});
            small_refs.push_back({&L.v_bias, L.has_qkv_bias ? load_small_f16(sa + ".v_proj.bias") : 0});

            L.gate = qt_from_st(mp + ".gate_proj", dim);
            L.up   = qt_from_st(mp + ".up_proj",   dim);
            L.down = qt_from_st(mp + ".down_proj", dense_inter);
        }

        id<MTLBuffer> small_buf = [dev newBufferWithBytes:small_blob.data() length:small_blob.size() options:MTLResourceStorageModeShared];
        fallback_bufs.push_back(small_buf);
        for (auto& sr : small_refs) { sr.dst->buf = small_buf; sr.dst->off = sr.off; }

        if (misaligned_copies > 0) {
            fprintf(stderr, "  [note] %d tensors copied for alignment (run `bihaba-serve align` for instant ~50ms boot)\n", misaligned_copies);
        }

        // Locate transformer kernels.metal relative to binary or BIHABA_SHADER_DIR
        auto try_read = [](const std::string& p) -> NSString* {
            if (p.empty()) return nil;
            return [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:p.c_str()] encoding:NSUTF8StringEncoding error:nil];
        };
        NSString* shaderSrc = nil;
        if (const char* env_dir = getenv("BIHABA_SHADER_DIR")) {
            shaderSrc = try_read(std::string(env_dir) + "/transformer_kernels.metal");
            if (!shaderSrc) shaderSrc = try_read(std::string(env_dir) + "/kernels.metal");
        }
        if (!shaderSrc) {
            char exe_buf[4096]; uint32_t exe_sz = sizeof(exe_buf);
            if (_NSGetExecutablePath(exe_buf, &exe_sz) == 0) {
                char real_buf[4096];
                const char* resolved = realpath(exe_buf, real_buf) ? real_buf : exe_buf;
                std::string exe_dir = dirname(const_cast<char*>(resolved));
                shaderSrc = try_read(exe_dir + "/transformer_kernels.metal");
                if (!shaderSrc) shaderSrc = try_read(exe_dir + "/../engines/transformer/kernels.metal");
                if (!shaderSrc) shaderSrc = try_read(exe_dir + "/../share/bihaba-serve/transformer_kernels.metal");
            }
        }
        if (!shaderSrc) shaderSrc = try_read("engines/transformer/kernels.metal");
        if (!shaderSrc) { fprintf(stderr, "Cannot locate transformer kernels.metal\n"); return false; }

        NSError* serr = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) { opts.mathMode = MTLMathModeFast; }
        id<MTLLibrary> lib = [dev newLibraryWithSource:shaderSrc options:opts error:&serr];
        if (!lib) { fprintf(stderr, "Shader compile error:\n%s\n", [[serr localizedDescription] UTF8String]); return false; }

        if (head_dim != 64 && head_dim != 128 && head_dim != 256) { fprintf(stderr, "✗ head_dim %d tidak didukung (64/128/256)\n", head_dim); return false; }
        if (n_kv_heads <= 0 || n_heads % n_kv_heads) { fprintf(stderr, "✗ num_attention_heads harus kelipatan num_key_value_heads\n"); return false; }
        if (dim % 16) { fprintf(stderr, "✗ hidden_size harus kelipatan 16\n"); return false; }

        const bool bf = scales_bf16;
        pso_embed         = make_pso(dev, lib, "embed_lookup_q",          false, bf);
        pso_rms           = make_pso(dev, lib, "rms_norm",                false, bf);
        pso_null          = make_pso(dev, lib, "null_kernel",             false, bf);
        pso_rep           = make_pso(dev, lib, "apply_rep_penalty_rows",  false, bf);
        pso_qmm4_mmaP     = make_pso(dev, lib, "qmm_4b_mmaP",             false, bf);
        pso_qmm4_mmaP_add = make_pso(dev, lib, "qmm_4b_mmaP",             true,  bf);
        pso_qmm2_mmaP     = make_pso(dev, lib, "qmm_2b_mmaP",             false, bf);
        pso_qmm2_mmaP_add = make_pso(dev, lib, "qmm_2b_mmaP",             true,  bf);
        pso_sw4_mmaP      = make_pso(dev, lib, "dense_swiglu_4b_mmaP",    false, bf);
        pso_sw2_mmaP      = make_pso(dev, lib, "dense_swiglu_2b_mmaP",    false, bf);
        pso_argmax        = make_pso(dev, lib, "argmax_full",             false, bf);
        pso_bw            = make_pso(dev, lib, "bw_probe",                false, bf);
        {
            const char* sn = head_dim == 64 ? "attn_split_64" : (head_dim == 128 ? "attn_split_128" : "attn_split_256");
            const char* mn = head_dim == 64 ? "attn_merge_64" : (head_dim == 128 ? "attn_merge_128" : "attn_merge_256");
            pso_attn_split = make_pso(dev, lib, sn, false, bf);
            pso_attn_merge = make_pso(dev, lib, mn, false, bf);
        }
        if (const char* av = getenv("BIHABA_ATTN_SPLIT")) { const int v = atoi(av); attn_split_min = v <= 0 ? (1 << 30) : v; }
        pso_topk64        = make_pso(dev, lib, "topk_logits_64",          false, bf);
        pso_qmm4_mma8     = make_pso(dev, lib, "qmm_4b_mma8",             false, bf);
        pso_qmm4_mma8_add = make_pso(dev, lib, "qmm_4b_mma8",             true,  bf);
        pso_qmm2_mma8     = make_pso(dev, lib, "qmm_2b_mma8",             false, bf);
        pso_qmm2_mma8_add = make_pso(dev, lib, "qmm_2b_mma8",             true,  bf);
        pso_sw4_mma8      = make_pso(dev, lib, "dense_swiglu_4b_mma8",    false, bf);
        pso_sw2_mma8      = make_pso(dev, lib, "dense_swiglu_2b_mma8",    false, bf);
        for (int bb : {2, 3, 4, 6, 8}) {
            for (int P = 0; P < 2; ++P) {
                char qn[48], sn[48];
                // P=1 (prefill, M>8): kernel GEMM 32×64; P=0 (verifikasi, M≤8): MMA 8 token
                snprintf(qn, sizeof(qn), "qmm_%db_%s", bb, P ? "gemm" : "mma8");
                snprintf(sn, sizeof(sn), "dense_swiglu_%db_%s", bb, P ? "gemm" : "mma8");
                pso_mm[bb][P][0] = make_pso(dev, lib, qn, false, bf);
                pso_mm[bb][P][1] = make_pso(dev, lib, qn, true,  bf);
                pso_swm[bb][P]   = make_pso(dev, lib, sn, false, bf);
            }
        }
        pso_qk_rope       = make_pso(dev, lib, "tr_qknorm_rope_kv",       false, bf);
        pso_gqa           = make_pso(dev, lib, (head_dim == 64 ? "gqa_attention_64" : (head_dim == 128 ? "gqa_attention_128" : "gqa_attention_256")), false, bf);

        const int bvals[5] = {2, 3, 4, 6, 8};
        const char* q1_n[5] = {"qmv_2b", "qmv_3b", "qmv_4b", "qmv_6b", "qmv_8b"};
        const char* qm_n[5] = {"qmv_2b_m", "qmv_3b_m", "qmv_4b_m", "qmv_6b_m", "qmv_8b_m"};
        const char* sw_n[5] = {"dense_swiglu_2b", "dense_swiglu_3b", "dense_swiglu_4b", "dense_swiglu_6b", "dense_swiglu_8b"};
        for (int i = 0; i < 5; ++i) {
            int b = bvals[i];
            pso_qmv1[b]     = make_pso(dev, lib, q1_n[i], false, bf);
            pso_qmv1_add[b] = make_pso(dev, lib, q1_n[i], true,  bf);
            pso_qmvm[b]     = make_pso(dev, lib, qm_n[i], false, bf);
            pso_qmvm_add[b] = make_pso(dev, lib, qm_n[i], true,  bf);
            pso_sw[b]       = make_pso(dev, lib, sw_n[i], false, bf);
        }

        auto mkbuf = [&](size_t bytes) -> id<MTLBuffer> {
            return [dev newBufferWithLength:std::max<size_t>(bytes, 256) options:MTLResourceStorageModeShared];
        };
        b_tok        = mkbuf(MAX_P * sizeof(int32_t));
        b_x          = mkbuf(MAX_P * dim * 2);
        b_xn         = mkbuf(MAX_P * dim * 2);
        b_q          = mkbuf(MAX_P * q_full_dim * 2);
        b_k          = mkbuf(MAX_P * kv_full_dim * 2);
        b_v          = mkbuf(MAX_P * kv_full_dim * 2);
        b_att        = mkbuf(MAX_P * q_full_dim * 2);
        b_dense_h    = mkbuf(MAX_P * dense_inter * 2);
        b_logits     = mkbuf(MAX_M * vocab * 2);
        b_out_tok    = mkbuf(MAX_M * sizeof(int32_t));
        b_top64_ids  = mkbuf(MAX_M * 64 * sizeof(int32_t));
        b_top64_vals = mkbuf(MAX_M * 64 * sizeof(float));
        b_top64_flag = mkbuf(MAX_M * sizeof(int32_t));
        b_part_ml    = mkbuf(size_t(MAX_M) * n_heads * ATTN_MAX_BLK * 2 * sizeof(float));
        b_part_o     = mkbuf(size_t(MAX_M) * n_heads * ATTN_MAX_BLK * head_dim * sizeof(float));

        model_fp = fnv1a(model_fp, &head_dim, sizeof(head_dim));
        k_caches.resize(n_layers, nil); v_caches.resize(n_layers, nil);
        size_t kv_bytes = size_t(n_kv_heads) * size_t(max_seq) * size_t(head_dim) * 2;
        for (int i = 0; i < n_layers; ++i) { k_caches[i] = mkbuf(kv_bytes); v_caches[i] = mkbuf(kv_bytes); }

        double load_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        fprintf(stderr, "▸ bihaba-serve [Taalas-Metal Transformer v1.0] (%s, Direct 0-Dup) Ready on %s (cold: %.1f ms | %.2f GB)\n",
                scales_bf16 ? "Native-BF16" : "FP16", [[dev name] UTF8String], load_ms, total_mapped_bytes / 1e9);
        fprintf(stderr, "  arch=%s | %dL dim=%d heads=%d/%d hd=%d inter=%d | vocab=%d | qk_norm=%d qkv_bias=%d\n",
                arch_name.c_str(), n_layers, dim, n_heads, n_kv_heads, head_dim, dense_inter, vocab,
                layers[0].has_qk_norm ? 1 : 0, layers[0].has_qkv_bias ? 1 : 0);
        return true;
    }

    void reset_states() { cached_ids.clear(); }
    inline void bar(id<MTLComputeCommandEncoder> e) {
        ++n_bar;
        if (ovh_mode == 2 || ovh_mode == 3) return;
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    void capture_snapshot(StateSnapshot& snap, const std::vector<int32_t>& ids, int pos) {
        snap.ids = ids; snap.pos = pos;
        snap.k_data.assign(n_layers, {}); snap.v_data.assign(n_layers, {});
        const size_t head_stride_bytes = size_t(max_seq) * size_t(head_dim) * 2;
        const size_t pos_bytes = size_t(pos) * size_t(head_dim) * 2;
        for (int l = 0; l < n_layers; ++l) {
            snap.k_data[l].resize(size_t(n_kv_heads) * pos_bytes);
            snap.v_data[l].resize(size_t(n_kv_heads) * pos_bytes);
            const uint8_t* kp = (const uint8_t*)[k_caches[l] contents];
            const uint8_t* vp = (const uint8_t*)[v_caches[l] contents];
            for (int h = 0; h < n_kv_heads; ++h) {
                memcpy(snap.k_data[l].data() + size_t(h) * pos_bytes, kp + size_t(h) * head_stride_bytes, pos_bytes);
                memcpy(snap.v_data[l].data() + size_t(h) * pos_bytes, vp + size_t(h) * head_stride_bytes, pos_bytes);
            }
        }
    }

    void restore_snapshot(const StateSnapshot& snap) {
        cached_ids = snap.ids;
        const size_t head_stride_bytes = size_t(max_seq) * size_t(head_dim) * 2;
        const size_t pos_bytes = size_t(snap.pos) * size_t(head_dim) * 2;
        for (int l = 0; l < n_layers; ++l) {
            uint8_t* kp = (uint8_t*)[k_caches[l] contents];
            uint8_t* vp = (uint8_t*)[v_caches[l] contents];
            for (int h = 0; h < n_kv_heads; ++h) {
                memcpy(kp + size_t(h) * head_stride_bytes, snap.k_data[l].data() + size_t(h) * pos_bytes, pos_bytes);
                memcpy(vp + size_t(h) * head_stride_bytes, snap.v_data[l].data() + size_t(h) * pos_bytes, pos_bytes);
            }
        }
    }

    // Format .tstate v2 (transformer): "TSTATET2" | int32[6] {n_layers, dim, n_kv_heads, head_dim, pos, n_ids}
    //   | int32 vocab | uint64 model_fp | ids | per layer uint64[2] + K/V. Setiap ukuran divalidasi saat load.
    bool save_snapshot_file(const StateSnapshot& snap, const std::string& path) const {
        const std::string tmp = path + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) { perror(("save " + path).c_str()); return false; }
        const char magic[8] = {'T','S','T','A','T','E','T','2'};
        int32_t hdr[7] = {n_layers, dim, n_kv_heads, head_dim, snap.pos, int32_t(snap.ids.size()), vocab};
        bool ok = fwrite(magic, 1, 8, f) == 8 && fwrite(hdr, sizeof(int32_t), 7, f) == 7 &&
                  fwrite(&model_fp, sizeof(model_fp), 1, f) == 1 &&
                  fwrite(snap.ids.data(), sizeof(int32_t), snap.ids.size(), f) == snap.ids.size();
        for (int l = 0; ok && l < n_layers; ++l) {
            uint64_t sz[2] = {snap.k_data[l].size(), snap.v_data[l].size()};
            ok = fwrite(sz, sizeof(uint64_t), 2, f) == 2 && fwrite(snap.k_data[l].data(), 1, sz[0], f) == sz[0] &&
                 fwrite(snap.v_data[l].data(), 1, sz[1], f) == sz[1];
        }
        ok = (fclose(f) == 0) && ok;
        if (ok) ok = rename(tmp.c_str(), path.c_str()) == 0;
        if (!ok) { remove(tmp.c_str()); fprintf(stderr, "✗ Gagal menulis %s\n", path.c_str()); }
        return ok;
    }

    bool load_snapshot_file(StateSnapshot& snap, const std::string& path) const {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { perror(("open " + path).c_str()); return false; }
        auto fail = [&](const char* why) { fprintf(stderr, "✗ .tstate ditolak (%s): %s\n", path.c_str(), why); fclose(f); return false; };
        char magic[8] = {0}; int32_t hdr[7] = {0}; uint64_t fp = 0;
        if (fread(magic, 1, 8, f) != 8) return fail("file terlalu pendek");
        if (memcmp(magic, "TSTATET1", 8) == 0) return fail("format lama TSTATET1 tanpa validasi — bake ulang");
        if (memcmp(magic, "TSTATET2", 8) != 0) return fail("bukan file .tstate engine transformer");
        if (fread(hdr, sizeof(int32_t), 7, f) != 7 || fread(&fp, sizeof(fp), 1, f) != 1) return fail("header terpotong");
        if (fp != model_fp) return fail("dibuat dari model lain (sidik jari berbeda)");
        if (hdr[0] != n_layers || hdr[1] != dim || hdr[2] != n_kv_heads || hdr[3] != head_dim || hdr[6] != vocab)
            return fail("arsitektur berbeda");
        const int pos = hdr[4], n_ids = hdr[5];
        if (pos <= 0 || pos + 9 > max_seq || n_ids != pos) return fail("panjang prefix tidak valid untuk --max-seq ini");
        snap.pos = pos; snap.ids.resize(size_t(n_ids));
        if (fread(snap.ids.data(), sizeof(int32_t), snap.ids.size(), f) != snap.ids.size()) return fail("token terpotong");
        for (int32_t t : snap.ids) if (t < 0 || t >= vocab) return fail("token di luar vocab");
        const uint64_t kv_bytes = uint64_t(n_kv_heads) * uint64_t(pos) * uint64_t(head_dim) * 2;
        snap.k_data.assign(n_layers, {}); snap.v_data.assign(n_layers, {});
        for (int l = 0; l < n_layers; ++l) {
            uint64_t sz[2] = {0, 0};
            if (fread(sz, sizeof(uint64_t), 2, f) != 2) return fail("data layer terpotong");
            if (sz[0] != kv_bytes || sz[1] != kv_bytes) return fail("ukuran KV layer tidak cocok");
            snap.k_data[l].resize(size_t(sz[0])); snap.v_data[l].resize(size_t(sz[1]));
            if (fread(snap.k_data[l].data(), 1, sz[0], f) != sz[0] || fread(snap.v_data[l].data(), 1, sz[1], f) != sz[1])
                return fail("data layer terpotong");
        }
        fclose(f); return true;
    }

    // Proyeksi terkuantisasi untuk M baris (M ≤ MAX_P). in_row/out_row = offset dalam BARIS (stride K / N).
    //   M=1        : qmv dual-row (decode)
    //   M=4..8     : MMA 1 tile   (verifikasi spekulatif)
    //   M=9..128   : GEMM 32×64   (prefill) — tile bobot dipakai ulang untuk 32 token, grid y = M/32
    //   lainnya    : qmv multi-token; M>8 tanpa MMA (3/8-bit) dipecah per 8 baris (kernelnya hanya 8 akumulator)
    void enc_qmv(id<MTLComputeCommandEncoder> e, int M, id<MTLBuffer> inp, const QTensor& qt, id<MTLBuffer> outp,
                 bool add_res, int in_row = 0, int out_row = 0) {
        const bool mma_ok = qt.bits >= 2 && qt.bits <= 8 && pso_mm[qt.bits][0][0] != nil && (qt.K % 64 == 0);
        if (M > MAX_M && !mma_ok) {
            for (int r0 = 0; r0 < M; r0 += MAX_M)
                enc_qmv(e, std::min(MAX_M, M - r0), inp, qt, outp, add_res, in_row + r0, out_row + r0);
            return;
        }
        const NSUInteger ioff = NSUInteger(in_row) * NSUInteger(qt.K) * 2;
        const NSUInteger ooff = NSUInteger(out_row) * NSUInteger(qt.N) * 2;
        id<MTLComputePipelineState> pso; int rows_per_tg; bool multi = true;
        if (M == 1) {
            pso = add_res ? pso_qmv1_add[qt.bits] : pso_qmv1[qt.bits]; rows_per_tg = 8; multi = false;
        } else if (M >= mma_min_m(qt.bits) && mma_ok) {
            pso = pso_mm[qt.bits][M > MAX_M ? 1 : 0][add_res ? 1 : 0];
            rows_per_tg = (M > MAX_M) ? 64 : 32;
        } else {
            pso = add_res ? pso_qmvm_add[qt.bits] : pso_qmvm[qt.bits]; rows_per_tg = 4;
        }
        set_pso(e, pso);
        [e setBuffer:inp offset:ioff atIndex:0];
        [e setBuffer:qt.w_buf offset:qt.w atIndex:1]; [e setBuffer:qt.s_buf offset:qt.s atIndex:2]; [e setBuffer:qt.b_buf offset:qt.b atIndex:3];
        [e setBuffer:outp offset:ooff atIndex:4];
        if (multi) { QMVMultiParams p{M, qt.K, qt.N, qt.gs_shift}; [e setBytes:&p length:sizeof(p) atIndex:5]; }
        else       { QMVParams p{qt.K, qt.N, qt.gs_shift};         [e setBytes:&p length:sizeof(p) atIndex:5]; }
        const NSUInteger gy = (multi && rows_per_tg == 64) ? NSUInteger((M + 31) / 32) : 1;
        [e dispatchThreadgroups:MTLSizeMake((qt.N + rows_per_tg - 1) / rows_per_tg, gy, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void enc_swiglu(id<MTLComputeCommandEncoder> e, int M, id<MTLBuffer> inp, const QTensor& gp, const QTensor& up, id<MTLBuffer> outp,
                    int row0 = 0) {
        const bool mma_ok = gp.bits >= 2 && gp.bits <= 8 && pso_swm[gp.bits][0] != nil && gp.bits == up.bits && (gp.K % 64 == 0);
        if (M > MAX_M && !mma_ok) {
            for (int r0 = 0; r0 < M; r0 += MAX_M) enc_swiglu(e, std::min(MAX_M, M - r0), inp, gp, up, outp, row0 + r0);
            return;
        }
        QMVMultiParams p{M, gp.K, gp.N, gp.gs_shift};
        const bool use_mma = (M >= mma_min_m(gp.bits) && mma_ok);
        id<MTLComputePipelineState> pso = use_mma
            ? pso_swm[gp.bits][M > MAX_M ? 1 : 0]
            : pso_sw[gp.bits];
        const int rows_per_tg = use_mma ? (M > MAX_M ? 64 : 32) : 4;
        set_pso(e, pso);
        [e setBuffer:inp offset:NSUInteger(row0) * NSUInteger(gp.K) * 2 atIndex:0];
        [e setBuffer:gp.w_buf offset:gp.w atIndex:1]; [e setBuffer:gp.s_buf offset:gp.s atIndex:2]; [e setBuffer:gp.b_buf offset:gp.b atIndex:3];
        [e setBuffer:up.w_buf offset:up.w atIndex:4]; [e setBuffer:up.s_buf offset:up.s atIndex:5]; [e setBuffer:up.b_buf offset:up.b atIndex:6];
        [e setBuffer:outp offset:NSUInteger(row0) * NSUInteger(gp.N) * 2 atIndex:7]; [e setBytes:&p length:sizeof(p) atIndex:8];
        const NSUInteger gy = (rows_per_tg == 64) ? NSUInteger((M + 31) / 32) : 1;
        [e dispatchThreadgroups:MTLSizeMake((gp.N + rows_per_tg - 1) / rows_per_tg, gy, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    void enc_layer(id<MTLComputeCommandEncoder> e, int M, int l, int pos) {
        const auto& L = layers[l];
        NormParams np{dim, rms_eps};
        set_pso(e, pso_rms);
        [e setBuffer:b_x offset:0 atIndex:0]; [e setBuffer:L.in_norm.buf offset:L.in_norm.off atIndex:1];
        [e setBuffer:b_xn offset:0 atIndex:2]; [e setBytes:&np length:sizeof(np) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        bar(e);

        // Concurrent Q, K, V projections!
        enc_qmv(e, M, b_xn, L.q, b_q, false);
        enc_qmv(e, M, b_xn, L.k, b_k, false);
        enc_qmv(e, M, b_xn, L.v, b_v, false);
        bar(e);

        TrRopeParams rp{M, n_heads, n_kv_heads, head_dim, pos, max_seq, L.has_qkv_bias ? 1 : 0, L.has_qk_norm ? 1 : 0, rms_eps};
        set_pso(e, pso_qk_rope);
        [e setBuffer:b_q offset:0 atIndex:0]; [e setBuffer:b_k offset:0 atIndex:1]; [e setBuffer:b_v offset:0 atIndex:2];
        [e setBuffer:k_caches[l] offset:0 atIndex:3]; [e setBuffer:v_caches[l] offset:0 atIndex:4];
        [e setBuffer:cos_s.buf offset:cos_s.off atIndex:5]; [e setBuffer:sin_s.buf offset:sin_s.off atIndex:6];
        [e setBuffer:L.q_norm.buf offset:L.q_norm.off atIndex:7]; [e setBuffer:L.k_norm.buf offset:L.k_norm.off atIndex:8];
        [e setBuffer:L.q_bias.buf offset:L.q_bias.off atIndex:9]; [e setBuffer:L.k_bias.buf offset:L.k_bias.off atIndex:10];
        [e setBuffer:L.v_bias.buf offset:L.v_bias.off atIndex:11];
        [e setBytes:&rp length:sizeof(rp) atIndex:12];
        [e dispatchThreadgroups:MTLSizeMake(n_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        bar(e);

        if (M <= MAX_M && pos + M >= attn_split_min && pso_attn_split) {
            // Konteks panjang: bagi sekuens ke n_blk threadgroup per head (≈256 token per blok), lalu gabungkan.
            const int nblk = std::min(int(ATTN_MAX_BLK), (pos + M + 255) / 256);
            AttnSplitParams ap{M, n_heads, n_kv_heads, pos, max_seq, nblk};
            set_pso(e, pso_attn_split);
            [e setBuffer:b_q offset:0 atIndex:0]; [e setBuffer:k_caches[l] offset:0 atIndex:1]; [e setBuffer:v_caches[l] offset:0 atIndex:2];
            [e setBuffer:b_part_ml offset:0 atIndex:3]; [e setBuffer:b_part_o offset:0 atIndex:4]; [e setBytes:&ap length:sizeof(ap) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(n_heads, nblk, M) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            bar(e);
            set_pso(e, pso_attn_merge);
            [e setBuffer:b_part_ml offset:0 atIndex:0]; [e setBuffer:b_part_o offset:0 atIndex:1]; [e setBuffer:b_q offset:0 atIndex:2];
            [e setBuffer:b_att offset:0 atIndex:3]; [e setBytes:&ap length:sizeof(ap) atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(n_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            bar(e);
        } else {
            TrGQAParams gp{M, n_heads, n_kv_heads, pos, max_seq};
            set_pso(e, pso_gqa);
            [e setBuffer:b_q offset:0 atIndex:0]; [e setBuffer:k_caches[l] offset:0 atIndex:1];
            [e setBuffer:v_caches[l] offset:0 atIndex:2]; [e setBuffer:b_att offset:0 atIndex:3];
            [e setBytes:&gp length:sizeof(gp) atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(n_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            bar(e);
        }

        enc_qmv(e, M, b_att, L.o, b_x, true);
        bar(e);

        set_pso(e, pso_rms);
        [e setBuffer:b_x offset:0 atIndex:0]; [e setBuffer:L.post_norm.buf offset:L.post_norm.off atIndex:1];
        [e setBuffer:b_xn offset:0 atIndex:2]; [e setBytes:&np length:sizeof(np) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        bar(e);

        enc_swiglu(e, M, b_xn, L.gate, L.up, b_dense_h);
        bar(e);
        enc_qmv(e, M, b_dense_h, L.down, b_x, true);
        bar(e);
    }

    // head_last_only: lm_head + argmax/top-64 hanya untuk baris terakhir (prefill). Hasil di indeks 0.
    void run_step_m(const int32_t* ids, int M, int pos, int32_t* out_ids,
                    bool run_head, const RepRowsParams& rp, double* gpu_ms_out = nullptr,
                    bool want_top64 = false, int32_t* top64_ids_out = nullptr, float* top64_vals_out = nullptr,
                    bool prefill = false, bool head_last_only = false) {
        if (M < 1 || M > (prefill ? MAX_P : MAX_M) || pos < 0 || pos + M > max_seq) {
            fprintf(stderr, "✗ run_step di luar batas: pos=%d M=%d max_seq=%d\n", pos, M, max_seq);
            exit(1);
        }
        const auto t_enc0 = std::chrono::steady_clock::now();
        memcpy([b_tok contents], ids, size_t(M) * sizeof(int32_t));
        @autoreleasepool {   // daemon jalan berjam-jam: lepas objek Metal sementara tiap langkah
        id<MTLCommandBuffer> cb = [queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

        QMVParams ep{dim, vocab, embed.gs_shift};
        int ebits = embed.bits;
        set_pso(e, pso_embed);
        [e setBuffer:b_tok offset:0 atIndex:0];
        [e setBuffer:embed.w_buf offset:embed.w atIndex:1]; [e setBuffer:embed.s_buf offset:embed.s atIndex:2]; [e setBuffer:embed.b_buf offset:embed.b atIndex:3];
        [e setBuffer:b_x offset:0 atIndex:4]; [e setBytes:&ep length:sizeof(ep) atIndex:5]; [e setBytes:&ebits length:sizeof(ebits) atIndex:6];
        [e dispatchThreads:MTLSizeMake(dim, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        bar(e);

        for (int l = 0; l < n_layers; ++l) enc_layer(e, M, l, pos);

        const int HM = head_last_only ? 1 : M;
        if (run_head) {
            NormParams np{dim, rms_eps};
            set_pso(e, pso_rms);
            [e setBuffer:b_x offset:(head_last_only ? NSUInteger(M - 1) * NSUInteger(dim) * 2 : 0) atIndex:0];
            [e setBuffer:fnorm_s.buf offset:fnorm_s.off atIndex:1];
            [e setBuffer:b_xn offset:0 atIndex:2]; [e setBytes:&np length:sizeof(np) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(HM, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            bar(e);

            cur_head = true;
            enc_qmv(e, HM, b_xn, lm_head, b_logits, false);
            cur_head = false;
            bar(e);
            if (!head_last_only && rep_active(rp) && rp.M == M) {
                set_pso(e, pso_rep);
                [e setBuffer:b_logits offset:0 atIndex:0]; [e setBytes:&rp length:sizeof(rp) atIndex:1];
                [e dispatchThreads:MTLSizeMake(REP_WIN, M, 1) threadsPerThreadgroup:MTLSizeMake(REP_WIN, 1, 1)];
                bar(e);
            }
            if (want_top64) {
                set_pso(e, pso_topk64);
                [e setBuffer:b_logits offset:0 atIndex:0]; [e setBuffer:b_top64_ids offset:0 atIndex:1];
                [e setBuffer:b_top64_vals offset:0 atIndex:2]; [e setBytes:&vocab length:sizeof(vocab) atIndex:3];
                [e setBuffer:b_top64_flag offset:0 atIndex:4];
                [e dispatchThreadgroups:MTLSizeMake(HM, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                set_pso(e, pso_argmax);
                [e setBuffer:b_logits offset:0 atIndex:0]; [e setBuffer:b_out_tok offset:0 atIndex:1];
                [e setBytes:&vocab length:sizeof(vocab) atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(HM, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
        [e endEncoding];
        last_enc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc0).count();
        [cb commit]; [cb waitUntilCompleted];
        if (gpu_ms_out) *gpu_ms_out = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        last_gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        if (run_head) {
            if (want_top64) {
                int32_t* ids64 = (int32_t*)[b_top64_ids contents];
                float* vals64  = (float*)[b_top64_vals contents];
                const int32_t* flags = (const int32_t*)[b_top64_flag contents];
                for (int m = 0; m < HM; ++m) if (flags[m]) cpu_top64(m, ids64 + m * 64, vals64 + m * 64);
                if (top64_ids_out)  memcpy(top64_ids_out,  ids64,  size_t(HM) * 64 * sizeof(int32_t));
                if (top64_vals_out) memcpy(top64_vals_out, vals64, size_t(HM) * 64 * sizeof(float));
                if (out_ids) for (int m = 0; m < HM; ++m) out_ids[m] = ids64[m * 64];
            } else if (out_ids) {
                memcpy(out_ids, [b_out_tok contents], size_t(HM) * sizeof(int32_t));
            }
        }
        }
    }


    // ─── Fase 0: biaya dispatch/barrier per token decode M=1 ───
    //   (a) step normal  (b) semua kernel kosong, grid & barrier sama  (c) seperti (b) tanpa barrier
    //   (d) waktu encode CPU per step. b = batas ATAS overhead GPU (di (a) sebagian bisa tumpang-tindih dengan kerja).
    void bench_overhead(int n) {
        if (!pso_null) { printf("▸ Overhead: null_kernel tidak tersedia\n"); return; }
        n = std::max(8, std::min(n, max_seq - 8));
        const RepRowsParams rp{};
        struct R { double wall = 0, gpu = 0, enc = 0; };
        long nd = 0, nb = 0;
        long ng[G_N] = {0};
        {   int32_t t = 9707; reset_states();
            const long d0 = n_disp, b0 = n_bar;
            for (int g = 0; g < G_N; ++g) n_disp_grp[g] = 0;
            ovh_count = true;
            run_step_m(&t, 1, 0, &t, true, rp);
            ovh_count = false;
            for (int g = 0; g < G_N; ++g) ng[g] = n_disp_grp[g];
            nd = n_disp - d0; nb = n_bar - b0; }
        auto run = [&](int mode) {
            R r; int32_t tok = 9707;
            ovh_mode = mode;
            for (int i = 0; i < 4; ++i) run_step_m(&tok, 1, i, &tok, true, rp);   // pemanasan
            reset_states(); tok = 9707;
            for (int i = 0; i < n; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                run_step_m(&tok, 1, i, &tok, true, rp);
                r.wall += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                r.gpu += last_gpu_ms; r.enc += last_enc_ms;
                if (mode) tok = 9707;          // kernel kosong tidak menulis token
            }
            ovh_mode = 0; reset_states();
            r.wall /= n; r.gpu /= n; r.enc /= n;
            return r;
        };
        const R a = run(0), b = run(1), c = run(2);
        printf("▸ Overhead/token [kelas: transformer, %d layer] (decode M=1, %d tok): %ld dispatch, %ld barrier per step\n", n_layers, n, nd, nb);
        printf("   (a) normal        : GPU %.3f ms | encode CPU %.3f ms | total %.3f ms\n", a.gpu, a.enc, a.wall);
        printf("   (b) kernel kosong : GPU %.3f ms -> %.2f us/dispatch (termasuk barrier) | total %.3f ms\n",
               b.gpu, nd ? 1e3 * b.gpu / double(nd) : 0.0, b.wall);
        printf("   (c) tanpa barrier : GPU %.3f ms -> barrier ~ %.2f us/buah, dispatch ~ %.2f us/buah\n",
               c.gpu, nb ? 1e3 * (b.gpu - c.gpu) / double(nb) : 0.0, nd ? 1e3 * c.gpu / double(nd) : 0.0);
        printf("   (d) encode CPU    : %.3f ms/token = %.0f%% dari total | celah total-GPU %.3f ms\n",
               a.enc, a.wall > 0 ? 100.0 * a.enc / a.wall : 0.0, a.wall - a.gpu);
        printf("   => kerja nyata ~ a-b = %.3f ms | overhead GPU (b) = %.0f%% dari GPU normal\n",
               a.gpu - b.gpu, a.gpu > 0 ? 100.0 * b.gpu / a.gpu : 0.0);
        // Fase 0b
        const R ee = run(3);
        printf("   (e) asli tanpa barrier: GPU %.3f ms -> biaya barrier nyata (termasuk drain) a-e = %.3f ms = %.2f us/barrier (%.0f%% dari GPU)\n",
               ee.gpu, a.gpu - ee.gpu, nb ? 1e3 * (a.gpu - ee.gpu) / double(nb) : 0.0,
               a.gpu > 0 ? 100.0 * (a.gpu - ee.gpu) / a.gpu : 0.0);
        printf("   (f) atribusi: kelompok dikosongkan -> GPU yang hilang (a-f), per dispatch\n");
        for (int g = 0; g < G_N; ++g) {
            if (ng[g] == 0) continue;
            ovh_grp = g; const R f = run(4); ovh_grp = -1;
            const double ms = a.gpu - f.gpu;
            printf("       %-13s %4ld disp : %7.3f ms (%3.0f%%) | %.2f us/disp\n", G_NAMES[g], ng[g], ms,
                   a.gpu > 0 ? 100.0 * ms / a.gpu : 0.0, 1e3 * ms / double(ng[g]));
        }
    }

    // Jalur cadangan exact untuk top-64 (dipakai hanya bila kernel menandai kemungkinan kandidat terlewat).
    void cpu_top64(int m, int32_t* ids, float* vals) const {
        const __fp16* row = (const __fp16*)[b_logits contents] + size_t(m) * size_t(vocab);
        std::vector<int32_t> idx(static_cast<size_t>(vocab));
        for (int i = 0; i < vocab; ++i) idx[size_t(i)] = i;
        const int k = std::min(64, vocab);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int32_t a, int32_t b) {
            const float fa = float(row[a]), fb = float(row[b]);
            return fa > fb || (fa == fb && a < b);
        });
        for (int i = 0; i < 64; ++i) {
            ids[i]  = (i < k) ? idx[size_t(i)] : idx[0];
            vals[i] = (i < k) ? float(row[idx[size_t(i)]]) : -INFINITY;
        }
    }


    void prefill_range(const std::vector<int32_t>& pids, size_t start, size_t end, int& pos,
                       bool run_last_head, bool want_top64, int32_t* last_out_id,
                       int32_t* last_top64_ids, float* last_top64_vals) {
        const RepRowsParams no_rp{};
        const int chunk_max = std::max(1, std::min(prefill_chunk, int(MAX_P)));
        int32_t out_id = 0, t64_i[64]; float t64_v[64];
        for (size_t i = start; i < end; i += size_t(chunk_max)) {
            const int chunk = int(std::min<size_t>(size_t(chunk_max), end - i));
            const bool is_last = run_last_head && (i + size_t(chunk) == end);
            run_step_m(pids.data() + i, chunk, pos, &out_id, is_last, no_rp, nullptr,
                       is_last && want_top64, t64_i, t64_v, /*prefill=*/true, /*head_last_only=*/true);
            for (int c = 0; c < chunk; ++c) cached_ids.push_back(pids[i + size_t(c)]);
            pos += chunk;
            if (is_last) {
                if (last_out_id) *last_out_id = out_id;
                if (want_top64 && last_top64_ids && last_top64_vals) {
                    memcpy(last_top64_ids,  t64_i, 64 * sizeof(int32_t));
                    memcpy(last_top64_vals, t64_v, 64 * sizeof(float));
                }
            }
        }
    }

    int32_t prefill_incremental(const std::vector<int32_t>& pids, int prefix_len, int& pos_out, int& reused_out,
                                bool want_top64 = false, int32_t* last_top64_ids = nullptr, float* last_top64_vals = nullptr) {
        // Pure KV-Cache Transformer supports arbitrary-length LCP reuse directly (no DeltaNet rewind constraint!)
        size_t lcp_live = 0;
        while (lcp_live < cached_ids.size() && lcp_live < pids.size() && cached_ids[lcp_live] == pids[lcp_live]) lcp_live++;
        size_t lcp_snap = 0;
        if (ram_snap.valid()) {
            while (lcp_snap < ram_snap.ids.size() && lcp_snap < pids.size() && ram_snap.ids[lcp_snap] == pids[lcp_snap]) lcp_snap++;
        }
        bool hit_snap = (ram_snap.valid() && lcp_snap == ram_snap.ids.size() && lcp_snap > 0 && lcp_snap < pids.size());

        size_t start = 0;
        if (lcp_live > 0 && lcp_live < pids.size() && (!hit_snap || lcp_live >= lcp_snap)) {
            cached_ids.resize(lcp_live);
            start = lcp_live; reused_out = int(lcp_live);
        } else if (hit_snap) {
            restore_snapshot(ram_snap);
            start = lcp_snap; reused_out = int(lcp_snap);
        } else {
            reset_states(); start = 0; reused_out = 0;
        }

        int pos = int(start);
        int32_t out_tok = 0;
        if (prefix_len > 0 && size_t(prefix_len) < pids.size() && size_t(prefix_len) > start && !hit_snap) {
            prefill_range(pids, start, size_t(prefix_len), pos, false, false, nullptr, nullptr, nullptr);
            std::vector<int32_t> pfx_ids(pids.begin(), pids.begin() + prefix_len);
            capture_snapshot(ram_snap, pfx_ids, pos);
            start = size_t(prefix_len);
        }
        prefill_range(pids, start, pids.size(), pos, true, want_top64, &out_tok, last_top64_ids, last_top64_vals);
        pos_out = pos;
        return out_tok;
    }

    void measure_step_costs(double cost[9]) {
        int32_t d_in[8] = {16, 11, 220, 17, 11, 220, 18, 11}, d_out[8];
        const RepRowsParams no_rp{};
        for (int w = 0; w < 2; ++w) run_step_m(d_in, 4, 32, d_out, true, no_rp);
        for (int m : {1, 2, 4, 6, 8}) {
            std::vector<double> ts;
            for (int it = 0; it < 5; ++it) {
                auto a = std::chrono::high_resolution_clock::now();
                run_step_m(d_in, m, 32, d_out, true, no_rp);
                ts.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - a).count());
            }
            std::sort(ts.begin(), ts.end());
            cost[m] = ts[ts.size() / 2];
        }
        for (int m = 3; m <= 7; m += 2) cost[m] = 0.5 * (cost[m-1] + cost[m+1]);
        reset_states();
    }
};

struct SpecStats {
    std::vector<int32_t> tokens;
    double prefill_ms = 0.0, decode_ms = 0.0;
    int reused = 0, steps = 0, proposed = 0, accepted = 0;
};

static int choose_best_k(double alpha, const double cost[9], int max_k) {
    int best_k = 0; double best_tps = 1.0 / cost[1];
    for (int k = 1; k <= max_k; ++k) {
        double et = (alpha >= 0.999) ? (k + 1.0) : (1.0 - std::pow(alpha, k + 1)) / (1.0 - alpha);
        double tps = et / cost[k + 1];
        if (tps > best_tps) { best_tps = tps; best_k = k; }
    }
    return best_k;
}

static void build_target_probs(const float vals[64], const SamplingConfig& sc, float probs[64]) {
    if (sc.temperature <= 1e-4f) { probs[0] = 1.0f; for (int i = 1; i < 64; ++i) probs[i] = 0.0f; return; }
    const float inv_t = 1.0f / sc.temperature, mx = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) { float p = (vals[i] > -1e20f) ? std::exp((vals[i] - mx) * inv_t) : 0.0f; probs[i] = p; sum += p; }
    const float inv_s = 1.0f / std::max(sum, 1e-12f);
    for (int i = 0; i < 64; ++i) probs[i] *= inv_s;
    int k_lim = (sc.top_k > 0 && sc.top_k < 64) ? sc.top_k : 64;
    for (int i = k_lim; i < 64; ++i) probs[i] = 0.0f;
    if (sc.min_p > 0.0f) { float th = probs[0] * sc.min_p; for (int i = 1; i < k_lim; ++i) if (probs[i] < th) probs[i] = 0.0f; }
    if (sc.top_p > 0.0f && sc.top_p < 1.0f) {
        float csum = 0.0f; bool cut = false;
        for (int i = 0; i < k_lim; ++i) { if (cut) { probs[i] = 0.0f; continue; } csum += probs[i]; if (csum >= sc.top_p) cut = true; }
    }
    float fsum = 0.0f; for (int i = 0; i < 64; ++i) fsum += probs[i];
    if (fsum <= 1e-12f) { probs[0] = 1.0f; for (int i = 1; i < 64; ++i) probs[i] = 0.0f; }
    else { float inv_f = 1.0f / fsum; for (int i = 0; i < 64; ++i) probs[i] *= inv_f; }
}

static int32_t sample_categorical(const int32_t ids[64], const float probs[64], std::mt19937& rng, int exclude_idx = -1) {
    float total = 0.0f;
    for (int i = 0; i < 64; ++i) if (i != exclude_idx) total += probs[i];
    if (total <= 1e-12f) return ids[0];
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng), acc = 0.0f;
    for (int i = 0; i < 64; ++i) { if (i == exclude_idx) continue; acc += probs[i]; if (r <= acc) return ids[i]; }
    return ids[0];
}

static SpecStats generate_adaptive(
    TransformerEngine& eng, const std::vector<int32_t>& pids, int prefix_len, int max_tokens,
    int max_spec, const SamplingConfig& sc, const double cost[9], bool stream_stdout, bool ignore_eos = false)
{
    SpecStats st;
    const bool do_sample = (sc.temperature > 1e-4f);
    std::mt19937 rng(sc.seed ? sc.seed : uint32_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    int pos = 0; int32_t t64_ids[8 * 64]; float t64_vals[8 * 64];
    auto tp0 = std::chrono::high_resolution_clock::now();
    int32_t nxt = eng.prefill_incremental(pids, prefix_len, pos, st.reused, do_sample, t64_ids, t64_vals);
    st.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tp0).count();

    if (do_sample) { float probs[64]; build_target_probs(t64_vals, sc, probs); nxt = sample_categorical(t64_ids, probs, rng); }
    FastSuffixSpeculator sfx;
    for (int32_t t : pids) sfx.push(t);
    sfx.push(nxt);
    st.tokens.push_back(nxt);
    if (stream_stdout) { printf("{\"tok\":%d}\n", nxt); fflush(stdout); }

    OrderAlpha oa; int32_t batch[8], out_ids[8];
    auto td0 = std::chrono::high_resolution_clock::now();

    while ((ignore_eos || !eng.is_eos(nxt)) && int(st.tokens.size()) < max_tokens && pos < eng.max_seq - 9 && !g_cancel.load(std::memory_order_relaxed)) {
        int rem = max_tokens - int(st.tokens.size());
        const int ord = (max_spec > 0) ? sfx.match_order() : 0;
        const int kk = ord ? std::min(choose_best_k(oa.alpha(ord), cost, max_spec), rem) : 0;
        auto draft = sfx.propose(kk);
        int M = 1 + int(draft.size());
        batch[0] = nxt;
        for (int i = 0; i < int(draft.size()); ++i) batch[1 + i] = draft[i];
        st.proposed += (M - 1);

        const RepRowsParams rp = build_rep_rows(st.tokens, batch, M, eng.vocab, sc.rep_pen);
        eng.run_step_m(batch, M, pos, out_ids, true, rp, nullptr, do_sample, t64_ids, t64_vals);
        st.steps++;

        std::vector<int32_t> emitted; int acc = 0; bool was_rejected = false;
        if (!do_sample) {
            while (acc < M - 1 && batch[1 + acc] == out_ids[acc]) { emitted.push_back(out_ids[acc]); acc++; }
            emitted.push_back(out_ids[acc]);
            was_rejected = (acc < M - 1);
        } else {
            bool rejected = false;
            for (int i = 0; i < M - 1; ++i) {
                float probs[64]; build_target_probs(t64_vals + i * 64, sc, probs);
                int32_t d_tok = batch[1 + i]; int d_idx = -1;
                for (int j = 0; j < 64; ++j) if (t64_ids[i * 64 + j] == d_tok) { d_idx = j; break; }
                float p_d = (d_idx >= 0) ? probs[d_idx] : 0.0f;
                if (u01(rng) < p_d) { emitted.push_back(d_tok); acc++; }
                else { emitted.push_back(sample_categorical(t64_ids + i * 64, probs, rng, d_idx)); rejected = true; was_rejected = true; break; }
            }
            if (!rejected) {
                float probs[64]; build_target_probs(t64_vals + (M - 1) * 64, sc, probs);
                emitted.push_back(sample_categorical(t64_ids + (M - 1) * 64, probs, rng));
            }
        }
        st.accepted += acc;
        int n_emit = int(emitted.size());
        // Potong di EOS / max_tokens SEBELUM mencatat state: input batch[0..n_keep-1] = yang resmi masuk
        // KV, cached_ids dan pos. Token terakhir yang dipancarkan belum menjadi input → percakapan
        // berikutnya (yang memuat token itu) tetap cocok sebagai prefix dan cache bisa dipakai ulang.
        int n_keep = n_emit;
        for (int i = 0; i < n_emit; ++i) {
            if ((!ignore_eos && eng.is_eos(emitted[i])) || int(st.tokens.size()) + i + 1 >= max_tokens) { n_keep = i + 1; break; }
        }
        if (ord > 0 && M > 1) oa.update(ord, acc, was_rejected);
        for (int i = 0; i < n_keep; ++i) eng.cached_ids.push_back(batch[i]);

        for (int i = 0; i < n_keep; ++i) {
            int32_t tok = emitted[i];
            st.tokens.push_back(tok); sfx.push(tok); nxt = tok;
            if (stream_stdout) { printf("{\"tok\":%d}\n", tok); fflush(stdout); }
        }
        pos += n_keep;
    }
    st.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - td0).count();
    return st;
}

int main(int argc, char** argv) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    {   // SA_RESTART: read() stdin & write() stdout yang terinterupsi dilanjutkan, bukan gagal EINTR
        struct sigaction sa{};
        sa.sa_handler = on_sigusr1; sa.sa_flags = SA_RESTART; sigemptyset(&sa.sa_mask);
        sigaction(SIGUSR1, &sa, nullptr);
    }
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage: %s <model_dir> [--bench 128] [--daemon] [--spec 7] [--max-seq 4096] [--prefill-chunk 128] [--rep-penalty 1.0]\n", argv[0]);
            return 1;
        }
        std::string model_path = argv[1], state_path = "";
        bool daemon_mode = false, willneed = false;
        int bench_toks = 0, max_spec = 7, req_max_seq = 4096, prefill_chunk = TransformerEngine::MAX_P;
        SamplingConfig def_sc;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--daemon") == 0) daemon_mode = true;
            else if (strcmp(argv[i], "--willneed") == 0) willneed = true;
            else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) bench_toks = atoi(argv[++i]);
            else if (strcmp(argv[i], "--spec") == 0 && i + 1 < argc) max_spec = std::min(7, std::max(0, atoi(argv[++i])));
            else if (strcmp(argv[i], "--max-seq") == 0 && i + 1 < argc) req_max_seq = atoi(argv[++i]);
            else if (strcmp(argv[i], "--state") == 0 && i + 1 < argc) state_path = argv[++i];
            else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) def_sc.rep_pen = atof(argv[++i]);
            else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) def_sc.temperature = atof(argv[++i]);
            else if (strcmp(argv[i], "--prefill-chunk") == 0 && i + 1 < argc)
                prefill_chunk = std::min(int(TransformerEngine::MAX_P), std::max(1, atoi(argv[++i])));
        }
        if (req_max_seq < 64) { fprintf(stderr, "--max-seq minimal 64\n"); return 1; }

        TransformerEngine eng;
        eng.prefill_chunk = prefill_chunk;
        if (!eng.load(model_path, willneed, req_max_seq)) return 1;
        if (!state_path.empty() && eng.load_snapshot_file(eng.ram_snap, state_path)) {
            fprintf(stderr, "▸ Loaded Baked KV State: %s (%d toks, %.1f MB)\n", state_path.c_str(), eng.ram_snap.pos, eng.ram_snap.bytes() / 1e6);
        }

        double cost[9] = {0};
        eng.measure_step_costs(cost);

        if (bench_toks > 0 && !daemon_mode) {
            printf("▸ Speculative Step Latency (M=1..8): ");
            for (int m : {1, 2, 4, 6, 8}) printf("n%d=%.2fms(ceil %.0f tok/s)  ", m, cost[m], m * 1000.0 / cost[m]);
            printf("\n");

            eng.reset_states();
            const RepRowsParams rp1{};
            int32_t tok = 9707;
            auto t0 = std::chrono::high_resolution_clock::now();
            double gpu_sum = 0.0;
            for (int i = 0; i < bench_toks; ++i) { eng.run_step_m(&tok, 1, i, &tok, true, rp1); gpu_sum += eng.last_gpu_ms; }
            double ms1 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            printf("▸ Live Decode Bench (Greedy M=1):      %d toks in %.1f ms => %.1f tok/s\n", bench_toks, ms1, bench_toks * 1000.0 / ms1);

            {   // Roofline: seberapa dekat decode M=1 ke batas fisik bandwidth memori
                const double gbs = eng.probe_bandwidth_gbs();
                const double mb_tok = eng.decode_bytes_per_token(bench_toks / 2) / 1e6;
                const double eff_gbs = mb_tok * (bench_toks * 1000.0 / ms1) / 1e3;
                if (gbs > 0)
                    printf("▸ Roofline: %.0f MB/token → decode %.0f GB/s = %.0f%% dari bandwidth terukur %.0f GB/s (batas fisik ≈ %.0f tok/s)\n",
                           mb_tok, eff_gbs, 100.0 * eff_gbs / gbs, gbs, gbs * 1e3 / mb_tok);
                const double wall_tok = ms1 / bench_toks, gpu_tok = gpu_sum / bench_toks;
                printf("▸ Waktu/token: total %.2f ms | GPU %.2f ms | CPU+sinkron %.2f ms (%.0f%%) | ideal-bandwidth %.2f ms\n",
                       wall_tok, gpu_tok, wall_tok - gpu_tok, 100.0 * (wall_tok - gpu_tok) / wall_tok,
                       gbs > 0 ? mb_tok / gbs : 0.0);
            }

            if (const char* ov = getenv("BIHABA_BENCH_OVERHEAD")) {   // Fase 0 (opt-in): biaya dispatch/barrier
                if (atoi(ov) > 0) eng.bench_overhead(std::max(64, bench_toks / 2));
            }

            {   // Decode di konteks panjang: attention lama vs split-sequence (hasil greedy harus identik)
                const int ctx = std::min(3072, eng.max_seq - 200);
                if (ctx >= 1024 && eng.pso_attn_split) {
                    std::vector<int32_t> lp(static_cast<size_t>(ctx));
                    for (size_t i = 0; i < lp.size(); ++i) lp[i] = int32_t((i * 7919u + 13u) % uint32_t(std::min(eng.vocab, 150000)));
                    std::vector<int32_t> seqs[2];
                    std::vector<float> marg[2];          // selisih logit top-1 − top-2 tiap langkah (bukti seri fp16)
                    double ms_v[2] = {0, 0};
                    int32_t i64[64]; float v64[64];
                    const int saved = eng.attn_split_min;
                    for (int v = 0; v < 2; ++v) {
                        eng.attn_split_min = v ? 1024 : (1 << 30);
                        eng.reset_states(); eng.ram_snap = StateSnapshot{};
                        int lpos = 0; int32_t t = 0;
                        eng.prefill_range(lp, 0, lp.size(), lpos, true, false, &t, nullptr, nullptr);
                        auto ta = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < 64; ++i) {
                            eng.run_step_m(&t, 1, lpos, &t, true, rp1, nullptr, true, i64, v64);
                            seqs[v].push_back(t); marg[v].push_back(v64[0] - v64[1]); lpos++;
                        }
                        ms_v[v] = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - ta).count();
                    }
                    eng.attn_split_min = saved;
                    int same = 0;
                    while (same < 64 && seqs[0][size_t(same)] == seqs[1][size_t(same)]) same++;
                    printf("▸ Decode @%d token konteks: attention lama %.1f tok/s | split %.1f tok/s (%+.0f%%) | greedy identik %d/64",
                           ctx, 64000.0 / ms_v[0], 64000.0 / ms_v[1], 100.0 * (ms_v[0] / ms_v[1] - 1.0), same);
                    if (same < 64)   // seri: selisih top-1/top-2 di titik beda ≈ resolusi fp16 (≤ ~0.06) → bukan bug
                        printf(" | selisih logit top1-top2 di token %d: lama %.4f, split %.4f", same, marg[0][size_t(same)], marg[1][size_t(same)]);
                    printf(" (aktif default: %s)\n", saved < (1 << 30) ? "ya" : "tidak");
                }
            }

            {   // Prefill A/B: chunk 8 (perilaku lama) vs chunk aktif, 512 token
                std::vector<int32_t> pp(512);
                for (size_t i = 0; i < pp.size(); ++i) pp[i] = int32_t((i * 7919u + 13u) % uint32_t(std::min(eng.vocab, 150000)));
                for (int ch : {8, 32, eng.prefill_chunk}) {
                    const int saved = eng.prefill_chunk; eng.prefill_chunk = ch;
                    double best = 1e30;
                    for (int it = 0; it < 2; ++it) {
                        eng.reset_states(); int pos = 0; int32_t o = 0;
                        auto a = std::chrono::high_resolution_clock::now();
                        eng.prefill_range(pp, 0, pp.size(), pos, true, false, &o, nullptr, nullptr);
                        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - a).count());
                    }
                    eng.prefill_chunk = saved;
                    printf("▸ Prefill Bench (512 tok, chunk %2d):   %.1f ms => %.0f tok/s\n", ch, best, 512 * 1000.0 / best);
                    if (ch == eng.prefill_chunk || ch > eng.prefill_chunk) break;
                }
                // Kebenaran prefill: 32 token greedy setelah prompt yang sama, chunk 8 (jalur MMA-8 v1.0) vs chunk aktif.
                if (eng.prefill_chunk != 8) {
                    std::vector<int32_t> ref, got;
                    for (int ch : {8, eng.prefill_chunk}) {
                        const int saved = eng.prefill_chunk; eng.prefill_chunk = ch;
                        eng.reset_states();
                        SpecStats gg = generate_adaptive(eng, pp, 0, 32, 0, SamplingConfig{}, cost, false, true);
                        eng.prefill_chunk = saved;
                        (ch == 8 ? ref : got) = gg.tokens;
                    }
                    int same = 0;
                    while (same < int(std::min(ref.size(), got.size())) && ref[size_t(same)] == got[size_t(same)]) same++;
                    printf("▸ Prefill check (chunk %d vs 8):          %d/%zu token greedy identik%s\n", eng.prefill_chunk, same, ref.size(),
                           same == int(ref.size()) ? "" : "  (selisih awal kecil bisa karena urutan penjumlahan fp16; selisih di token 0–1 = bug)");
                }
            }
            // CATATAN: pola sintetis (deret angka berulang) → angka spekulatif = batas atas.
            std::vector<int32_t> pattern = {16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220, 22, 11, 220, 23, 198};
            std::vector<int32_t> prompt;
            for (int r = 0; r < 8; ++r) prompt.insert(prompt.end(), pattern.begin(), pattern.end());
            SamplingConfig sc_g{};
            eng.reset_states(); SpecStats g0 = generate_adaptive(eng, prompt, 0, bench_toks, 0, sc_g, cost, false, true);
            eng.reset_states(); SpecStats gs = generate_adaptive(eng, prompt, 0, bench_toks, max_spec, sc_g, cost, false, true);
            int match = 0, lim = int(std::min(g0.tokens.size(), gs.tokens.size()));
            while (match < lim && g0.tokens[match] == gs.tokens[match]) match++;
            int n_dec = std::max(0, int(gs.tokens.size()) - 1);
            printf("▸ Live Speculative Bench (Greedy k=%d, pola sintetis): %d toks in %.1f ms => %.1f tok/s (%.2f tok/step, =greedy %d/%d)\n",
                   max_spec, n_dec, gs.decode_ms, n_dec * 1000.0 / std::max(gs.decode_ms, 0.01),
                   double(n_dec) / std::max(gs.steps, 1), match, lim);
            return 0;
        }

        if (daemon_mode) {
            printf("{\"ready\":true,\"cancel\":true,\"version\":\"1.1\",\"arch\":\"%s\",\"model_dir\":\"%s\",\"top_k\":0,\"max_spec\":%d,\"baked_toks\":%d,\"max_seq\":%d,\"prefill_chunk\":%d}\n",
                   eng.arch_name.c_str(), eng.model_dir.c_str(), max_spec, eng.ram_snap.pos, eng.max_seq, eng.prefill_chunk);
            fflush(stdout);
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                NSData* reqData = [NSData dataWithBytes:line.data() length:line.size()];
                NSDictionary* req = [NSJSONSerialization JSONObjectWithData:reqData options:0 error:nil];
                if (!req) continue;

                NSString* cmd = req[@"cmd"];
                if (cmd && [cmd isEqualToString:@"clear_state"]) {
                    // Buang snapshot prefix di RAM + cache percakapan. File .tstate di disk tidak disentuh.
                    const double freed = eng.ram_snap.bytes() / 1e6;
                    eng.ram_snap = StateSnapshot{};
                    eng.reset_states();
                    printf("{\"cleared\":true,\"mb\":%.1f}\n", freed);
                    fflush(stdout);
                    continue;
                }
                if (cmd && [cmd isEqualToString:@"bake"]) {
                    std::vector<int32_t> pids;
                    for (NSNumber* num in req[@"prompt_ids"]) pids.push_back([num intValue]);
                    std::string out_p = req[@"path"] ? [req[@"path"] UTF8String] : "";
                    bool bad_bake = false;
                    for (int32_t t : pids) if (t < 0 || t >= eng.vocab) { bad_bake = true; break; }
                    if (pids.empty() || bad_bake || int(pids.size()) + 9 > eng.max_seq) {
                        printf("{\"baked\":false,\"error\":\"prefix %zu token kosong/di luar vocab/melebihi max_seq %d\"}\n", pids.size(), eng.max_seq);
                        fflush(stdout); continue;
                    }
                    auto tb0 = std::chrono::high_resolution_clock::now();
                    eng.reset_states(); int pos = 0;
                    eng.prefill_range(pids, 0, pids.size(), pos, false, false, nullptr, nullptr, nullptr);
                    eng.capture_snapshot(eng.ram_snap, pids, pos);
                    bool saved = out_p.empty() ? true : eng.save_snapshot_file(eng.ram_snap, out_p);
                    double bms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tb0).count();
                    printf("{\"baked\":%s,\"tokens\":%d,\"mb\":%.1f,\"ms\":%.1f}\n", saved ? "true" : "false", pos, eng.ram_snap.bytes() / 1e6, bms);
                    fflush(stdout); continue;
                }

                std::vector<int32_t> pids;
                for (NSNumber* num in req[@"prompt_ids"]) pids.push_back([num intValue]);
                int mt         = req[@"max_tokens"] ? [req[@"max_tokens"] intValue] : 512;
                int prefix_len = req[@"prefix_len"] ? [req[@"prefix_len"] intValue] : 0;
                int sp_k       = req[@"max_spec"]   ? std::min(7, std::max(0, [req[@"max_spec"] intValue])) : max_spec;
                if (mt < 1) mt = 1;
                if (prefix_len < 0 || size_t(prefix_len) >= pids.size()) prefix_len = 0;
                bool bad_tok = false;
                for (int32_t t : pids) if (t < 0 || t >= eng.vocab) { bad_tok = true; break; }
                if (pids.empty() || bad_tok || int(pids.size()) + mt + 9 > eng.max_seq) {
                    printf("{\"done\":true,\"error\":\"prompt (%zu token) kosong/di luar vocab, atau + max_tokens (%d) melebihi max_seq (%d)\",\"tokens\":0}\n",
                           pids.size(), mt, eng.max_seq);
                    fflush(stdout); continue;
                }
                SamplingConfig sc = def_sc;
                if (req[@"temperature"]) sc.temperature = [req[@"temperature"] floatValue];
                if (req[@"top_p"])       sc.top_p       = [req[@"top_p"] floatValue];
                if (req[@"top_k"])       sc.top_k       = [req[@"top_k"] intValue];
                if (req[@"min_p"])       sc.min_p       = [req[@"min_p"] floatValue];
                if (req[@"rep_penalty"]) sc.rep_pen     = [req[@"rep_penalty"] floatValue];
                if (req[@"seed"])        sc.seed        = uint32_t([req[@"seed"] unsignedIntValue]);

                g_cancel.store(false, std::memory_order_relaxed);
                SpecStats st = generate_adaptive(eng, pids, prefix_len, mt, sp_k, sc, cost, true);
                int n_dec = std::max(0, int(st.tokens.size()) - 1);
                double tps = (st.decode_ms > 0.0) ? (n_dec * 1000.0 / st.decode_ms) : 0.0;
                printf("{\"done\":true,\"tokens\":%d,\"ms\":%.1f,\"prefill_ms\":%.1f,\"reused\":%d,\"baked_toks\":%d,\"tps\":%.1f,\"tok_per_step\":%.2f}\n",
                       n_dec, st.decode_ms, st.prefill_ms, st.reused, eng.ram_snap.pos, tps, double(n_dec) / std::max(st.steps, 1));
                fflush(stdout);
            }
        }
    }
    return 0;
}
