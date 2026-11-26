// ============================================================================
// bihaba-serve — Native Apple Silicon Metal Inference Engine (Qwen3.5 & MoE)
// Powered by Taalas-Metal Engine | Created by Boy Barley from Bihaba (Team Pasai Aceh)
//
// Compile (~1s, 0 external deps):
//   clang++ -O3 -std=c++17 -fobjc-arc engines/qwen35/main.mm \
//     -framework Metal -framework Foundation -o build/bihaba-qwen35
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mach-o/dyld.h>
#include <libgen.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <random>
#include <unordered_map>
#include <vector>

struct NormParams       { int32_t dim; float eps; };
struct QMVParams        { int32_t K, N, gs_shift; };
struct QMVMultiParams   { int32_t M, K, N, gs_shift; };
struct Conv1dParams     { int32_t M, C, k_size; };
struct DeltaNetParams   { int32_t M, n_kh, n_vh, k_hd, v_hd; float eps; };
struct QKNormRopeParams { int32_t M, n_heads, n_kv_heads, head_dim, rope_dim, pos, max_seq; float eps; };
struct GQA256Params     { int32_t M, n_heads, n_kv_heads, pos, max_seq; };
struct TopKParams       { int32_t M, E, top_k; };
struct MoEProjParams    { int32_t M, K, N, top_k, gs_shift; uint64_t stride_w, stride_s; };
struct SparseRepParams  { int32_t M, K_len, n_slots; float penalty; int32_t slots[64]; };
struct DeltaNetInParams { int32_t M, K, gs_shift, n_qkv, n_z, n_a, n_b; };
struct FullQKVParams    { int32_t M, K, gs_shift, n_q, n_k, n_v; };

struct BufSlice {
    id<MTLBuffer> buf = nil;
    uint64_t off = 0;
};

struct QTensor {
    id<MTLBuffer> w_buf = nil, s_buf = nil, b_buf = nil;
    uint64_t w = 0, s = 0, b = 0;
    int bits = 4, gs = 64, gs_shift = 6, N = 0, K = 0;
    uint64_t stride_w = 0, stride_s = 0;
};

struct LayerMeta {
    int type = 0;
    BufSlice in_norm, post_norm;
    QTensor in_qkv, in_z, in_a, in_b, out_p;
    BufSlice conv_w, A_log, dt_bias, dn_norm;
    QTensor q, k, v, o;
    BufSlice q_norm, k_norm;
    QTensor router, sh_gate, sh_gp, sh_up, sh_dp, moe_gp, moe_up, moe_dp;
    QTensor gate, up, down;
};

struct MappedFile {
    int fd = -1;
    void* mapped = nullptr;
    size_t fsize = 0;
    size_t map_len = 0;
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
    std::vector<std::vector<uint8_t>> conv_data;
    std::vector<std::vector<uint8_t>> dn_data;
    // Compact KV storage: for each full-attn layer, stores [n_kv_heads, pos, head_dim] half (NOT max_seq!)
    std::vector<std::vector<uint8_t>> k_data;
    std::vector<std::vector<uint8_t>> v_data;

    bool valid() const { return pos > 0 && !ids.empty(); }
    size_t bytes() const {
        size_t s = 0;
        for (const auto& v : conv_data) s += v.size();
        for (const auto& v : dn_data)   s += v.size();
        for (const auto& v : k_data)    s += v.size();
        for (const auto& v : v_data)    s += v.size();
        return s;
    }
};

static inline float bf16_to_f32(uint16_t u) {
    uint32_t x = uint32_t(u) << 16;
    float f = 0.0f;
    memcpy(&f, &x, 4);
    return f;
}

static inline uint16_t f32_to_f16_bits(float f) {
    __fp16 h = (__fp16)f;
    uint16_t u = 0;
    memcpy(&u, &h, 2);
    return u;
}

static inline float f16_bits_to_f32(uint16_t u) {
    __fp16 h;
    memcpy(&h, &u, 2);
    return float(h);
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

    void clear() { toks.clear(); idx2.clear(); idx3.clear(); idx4.clear(); }

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
                if (take > 0) {
                    out.assign(toks.begin() + pos, toks.begin() + pos + take);
                    return true;
                }
            }
            return false;
        };
        if (n >= 4 && scan(idx4, h4(toks[n-4], toks[n-3], toks[n-2], toks[n-1]), 4)) return out;
        if (n >= 3 && scan(idx3, h3(toks[n-3], toks[n-2], toks[n-1]), 3)) return out;
        if (n >= 2 && scan(idx2, h2(toks[n-2], toks[n-1]), 2)) return out;
        return out;
    }
};

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
    return pso;
}

static SparseRepParams build_rep(const std::vector<int32_t>& gen, int M, int vocab, float pen) {
    SparseRepParams rp{};
    rp.M = M; rp.K_len = vocab; rp.penalty = pen; rp.n_slots = 0;
    if (pen <= 1.001f || gen.empty()) return rp;
    int start = std::max(0, int(gen.size()) - 64);
    for (int i = start; i < int(gen.size()); ++i) {
        int32_t t = gen[i];
        if (t < 0 || t >= vocab) continue;
        bool dup = false;
        for (int j = 0; j < rp.n_slots; ++j) if (rp.slots[j] == t) { dup = true; break; }
        if (!dup && rp.n_slots < 64) rp.slots[rp.n_slots++] = t;
    }
    return rp;
}

struct Qwen35Engine {
    static constexpr int MAX_M = 8;

    std::vector<MappedFile> mapped_files;
    std::vector<id<MTLBuffer>> fallback_bufs;  // small tensors + any rare misaligned tensor fallback
    size_t total_mapped_bytes = 0;
    bool scales_bf16 = false;
    bool direct_st_mode = false;

    int n_layers=0, dim=0, vocab=0, n_heads=0, n_kv_heads=0, head_dim=256, rope_dim=64;
    int lin_kh=16, lin_vh=32, lin_khd=128, lin_vhd=128, conv_k=4;
    int qkv_lin_dim=0, val_lin_dim=0, q_full_dim=0, kv_full_dim=0;
    bool is_moe = false;
    int num_experts=0, top_k=4, moe_inter=512, shared_inter=512, dense_inter=0, max_seq=4096;
    float rms_eps = 1e-6f;
    std::vector<int32_t> eos_ids;
    std::string arch_name, model_dir;

    QTensor embed, lm_head;
    BufSlice cos_s, sin_s, fnorm_s;
    std::vector<LayerMeta> layers;

    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> pso_embed=nil, pso_rms=nil, pso_rep=nil, pso_argmax=nil, pso_topk64=nil;
    id<MTLComputePipelineState> pso_qmv1[9], pso_qmv1_add[9];
    id<MTLComputePipelineState> pso_qmvm[9], pso_qmvm_add[9], pso_sw[9];
    id<MTLComputePipelineState> pso_dn_in4=nil, pso_dn_in4_m1=nil, pso_dn_in4_mma8=nil;
    id<MTLComputePipelineState> pso_full_qkv4=nil, pso_full_qkv4_m1=nil, pso_full_qkv4_mma8=nil;
    id<MTLComputePipelineState> pso_qmm4_mma8=nil, pso_qmm4_mma8_add=nil, pso_qmm2_mma8=nil, pso_qmm2_mma8_add=nil;
    id<MTLComputePipelineState> pso_sw4_mma8=nil, pso_sw2_mma8=nil;
    id<MTLComputePipelineState> pso_conv1d_m1=nil, pso_conv1d_spec=nil;
    id<MTLComputePipelineState> pso_dn_m1=nil, pso_dn_spec=nil;
    id<MTLComputePipelineState> pso_commit_conv=nil, pso_commit_dn=nil;
    id<MTLComputePipelineState> pso_qk_rope=nil, pso_gqa256=nil;
    id<MTLComputePipelineState> pso_router_sh4=nil, pso_router_sh8=nil, pso_topk=nil, pso_combine=nil;
    id<MTLComputePipelineState> pso_moe_sw1[5], pso_moe_dn1[5];

    id<MTLBuffer> b_tok=nil, b_x=nil, b_xn=nil;
    id<MTLBuffer> b_qkv=nil, b_z=nil, b_a=nil, b_b=nil, b_dn_out=nil;
    id<MTLBuffer> b_q_raw=nil, b_q=nil, b_gate=nil, b_k=nil, b_v=nil, b_att=nil;
    id<MTLBuffer> b_router_l=nil, b_sh_gl=nil, b_top_ids=nil, b_top_w=nil, b_sh_g=nil;
    id<MTLBuffer> b_uniq_cnt=nil, b_uniq_e=nil, b_grp_cnt=nil, b_grp_slots=nil;
    id<MTLBuffer> b_sh_h=nil, b_sh_out=nil, b_moe_h=nil, b_moe_out=nil, b_dense_h=nil;
    id<MTLBuffer> b_logits=nil, b_out_tok=nil, b_top64_ids=nil, b_top64_vals=nil;

    std::vector<id<MTLBuffer>> conv_states, dn_states, k_caches, v_caches;
    std::vector<id<MTLBuffer>> conv_hist, dn_hist_k, dn_hist_d, dn_hist_dec;

    int pending_dn_commit = 0;
    std::vector<int32_t> cached_ids;
    StateSnapshot ram_snap;

    bool is_eos(int32_t t) const {
        for (int32_t e : eos_ids) if (t == e) return true;
        return false;
    }

    ~Qwen35Engine() {
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
        if (!out_mf.mtl_buf) {
            fprintf(stderr, "newBufferWithBytesNoCopy failed on %s (%.2f GB)\n", fpath.c_str(), out_mf.fsize / 1e9);
            return false;
        }
        total_mapped_bytes += out_mf.fsize;
        return true;
    }

    bool load_taalas2(const std::string& path, int override_topk, bool force_willneed) {
        direct_st_mode = false;
        scales_bf16 = false;
        MappedFile mf;
        if (!mmap_file_mtl(path, force_willneed, mf)) return false;
        if (memcmp(mf.mapped, "TAALAS02", 8) != 0) { fprintf(stderr, "Invalid magic (expected TAALAS02)\n"); return false; }

        uint64_t meta_len = 0; memcpy(&meta_len, (uint8_t*)mf.mapped + 8, 8);
        NSData* metaData = [NSData dataWithBytesNoCopy:(uint8_t*)mf.mapped + 16 length:meta_len freeWhenDone:NO];
        NSError* jerr = nil;
        NSDictionary* meta = [NSJSONSerialization JSONObjectWithData:metaData options:0 error:&jerr];
        if (!meta) { fprintf(stderr, "JSON error: %s\n", [[jerr localizedDescription] UTF8String]); return false; }

        id<MTLBuffer> W = mf.mtl_buf;
        mapped_files.push_back(mf);

        arch_name    = [meta[@"arch"] UTF8String];
        model_dir    = meta[@"model_dir"] ? [meta[@"model_dir"] UTF8String] : "";
        n_layers     = [meta[@"n_layers"] intValue];
        dim          = [meta[@"dim"] intValue];
        vocab        = [meta[@"vocab"] intValue];
        rms_eps      = [meta[@"rms_eps"] floatValue];
        n_heads      = [meta[@"n_heads"] intValue];
        n_kv_heads   = [meta[@"n_kv_heads"] intValue];
        head_dim     = [meta[@"head_dim"] intValue];
        rope_dim     = [meta[@"rope_dim"] intValue];
        lin_kh       = [meta[@"lin_kh"] intValue];
        lin_vh       = [meta[@"lin_vh"] intValue];
        lin_khd      = [meta[@"lin_khd"] intValue];
        lin_vhd      = [meta[@"lin_vhd"] intValue];
        conv_k       = [meta[@"conv_k"] intValue];
        is_moe       = [meta[@"is_moe"] boolValue];
        num_experts  = [meta[@"num_experts"] intValue];
        top_k        = (override_topk > 0) ? override_topk : [meta[@"top_k"] intValue];
        moe_inter    = [meta[@"moe_inter"] intValue];
        shared_inter = [meta[@"shared_inter"] intValue];
        dense_inter  = [meta[@"dense_inter"] intValue];
        max_seq      = [meta[@"max_seq"] intValue];
        for (NSNumber* e in meta[@"eos_ids"]) eos_ids.push_back([e intValue]);
        if (eos_ids.empty()) eos_ids.push_back([meta[@"eos_id"] intValue]);

        NSDictionary* tensors = meta[@"tensors"];
        NSDictionary* qmeta   = meta[@"qmeta"];
        NSArray* ltypes       = meta[@"layer_types"];

        auto sl_of = [&](const std::string& k) -> BufSlice {
            NSArray* arr = tensors[[NSString stringWithUTF8String:k.c_str()]];
            if (!arr) { fprintf(stderr, "Missing tensor %s\n", k.c_str()); exit(1); }
            return {W, [arr[0] unsignedLongLongValue]};
        };
        auto qt_of = [&](const std::string& k, bool is_switch = false) -> QTensor {
            QTensor q{};
            q.w_buf = W; q.s_buf = W; q.b_buf = W;
            q.w = sl_of(k + ".w").off; q.s = sl_of(k + ".s").off; q.b = sl_of(k + ".b").off;
            NSArray* qm = qmeta[[NSString stringWithUTF8String:k.c_str()]];
            if (!qm) { fprintf(stderr, "Missing qmeta %s\n", k.c_str()); exit(1); }
            q.bits = [qm[0] intValue]; q.gs = [qm[1] intValue];
            q.N    = [qm[2] intValue]; q.K  = [qm[3] intValue];
            q.gs_shift = (q.gs == 32) ? 5 : 6;
            if (is_switch) {
                uint64_t kw = (q.bits == 4) ? uint64_t(q.K >> 3) :
                              (q.bits == 2) ? uint64_t(q.K >> 4) :
                              (q.bits == 8) ? uint64_t(q.K >> 2) : uint64_t((q.K * 3) >> 5);
                q.stride_w = uint64_t(q.N) * kw;
                q.stride_s = uint64_t(q.N) * uint64_t(q.K >> q.gs_shift);
            }
            return q;
        };

        cos_s = sl_of("rope.cos"); sin_s = sl_of("rope.sin"); fnorm_s = sl_of("final_norm");
        embed = qt_of("embed"); lm_head = qt_of("lm_head");

        layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            std::string lp = "L" + std::to_string(i);
            auto& L = layers[i];
            L.type = [ltypes[i] intValue];
            L.in_norm = sl_of(lp + ".in_norm"); L.post_norm = sl_of(lp + ".post_norm");
            if (L.type == 0) {
                L.in_qkv = qt_of(lp + ".in_qkv"); L.in_z = qt_of(lp + ".in_z");
                L.in_a   = qt_of(lp + ".in_a");   L.in_b = qt_of(lp + ".in_b");
                L.out_p  = qt_of(lp + ".out_p");
                L.conv_w = sl_of(lp + ".conv_w"); L.A_log = sl_of(lp + ".A_log");
                L.dt_bias = sl_of(lp + ".dt_bias"); L.dn_norm = sl_of(lp + ".dn_norm");
            } else {
                L.q = qt_of(lp + ".q"); L.k = qt_of(lp + ".k");
                L.v = qt_of(lp + ".v"); L.o = qt_of(lp + ".o");
                L.q_norm = sl_of(lp + ".q_norm"); L.k_norm = sl_of(lp + ".k_norm");
            }
            if (is_moe) {
                L.router = qt_of(lp + ".router"); L.sh_gate = qt_of(lp + ".sh_gate");
                L.sh_gp  = qt_of(lp + ".sh_gp");  L.sh_up   = qt_of(lp + ".sh_up"); L.sh_dp = qt_of(lp + ".sh_dp");
                L.moe_gp = qt_of(lp + ".moe_gp", true); L.moe_up = qt_of(lp + ".moe_up", true); L.moe_dp = qt_of(lp + ".moe_dp", true);
            } else {
                L.gate = qt_of(lp + ".gate"); L.up = qt_of(lp + ".up"); L.down = qt_of(lp + ".down");
            }
        }
        return true;
    }

    // Direct Zero-Copy .safetensors Multi-Shard Loader (0 GB Disk Duplication!)
    bool load_safetensors_dir(const std::string& dir_path, int override_topk, bool force_willneed, int req_max_seq) {
        direct_st_mode = true;
        model_dir = dir_path;
        max_seq = req_max_seq;

        auto read_json = [&](const std::string& rel) -> NSDictionary* {
            NSString* p = [NSString stringWithUTF8String:(dir_path + "/" + rel).c_str()];
            NSData* d = [NSData dataWithContentsOfFile:p];
            if (!d) return nil;
            return [NSJSONSerialization JSONObjectWithData:d options:0 error:nil];
        };

        NSDictionary* cfg = read_json("config.json");
        if (!cfg) { fprintf(stderr, "Cannot read %s/config.json\n", dir_path.c_str()); return false; }
        NSDictionary* tc = cfg[@"text_config"] ?: cfg;

        n_layers     = [tc[@"num_hidden_layers"] intValue];
        dim          = [tc[@"hidden_size"] intValue];
        vocab        = [tc[@"vocab_size"] intValue];
        rms_eps      = tc[@"rms_norm_eps"] ? [tc[@"rms_norm_eps"] floatValue] : 1e-6f;
        n_heads      = [tc[@"num_attention_heads"] intValue];
        n_kv_heads   = [tc[@"num_key_value_heads"] intValue];
        head_dim     = tc[@"head_dim"] ? [tc[@"head_dim"] intValue] : 256;
        lin_kh       = tc[@"linear_num_key_heads"] ? [tc[@"linear_num_key_heads"] intValue] : 16;
        lin_vh       = tc[@"linear_num_value_heads"] ? [tc[@"linear_num_value_heads"] intValue] : 32;
        lin_khd      = tc[@"linear_key_head_dim"] ? [tc[@"linear_key_head_dim"] intValue] : 128;
        lin_vhd      = tc[@"linear_value_head_dim"] ? [tc[@"linear_value_head_dim"] intValue] : 128;
        conv_k       = tc[@"linear_conv_kernel_dim"] ? [tc[@"linear_conv_kernel_dim"] intValue] : 4;
        num_experts  = tc[@"num_experts"] ? [tc[@"num_experts"] intValue] : 0;
        is_moe       = (num_experts > 0);
        top_k        = (override_topk > 0) ? override_topk : (tc[@"num_experts_per_tok"] ? [tc[@"num_experts_per_tok"] intValue] : 4);
        moe_inter    = tc[@"moe_intermediate_size"] ? [tc[@"moe_intermediate_size"] intValue] : 512;
        shared_inter = tc[@"shared_expert_intermediate_size"] ? [tc[@"shared_expert_intermediate_size"] intValue] : 512;
        dense_inter  = tc[@"intermediate_size"] ? [tc[@"intermediate_size"] intValue] : 0;
        arch_name    = is_moe ? "qwen3_5_moe" : "qwen3_5";

        // Layer types
        std::vector<int> ltypes(n_layers, 0);
        NSArray* raw_lt = tc[@"layer_types"];
        if (raw_lt && int([raw_lt count]) == n_layers) {
            for (int i = 0; i < n_layers; ++i) ltypes[i] = [raw_lt[i] isEqualToString:@"full_attention"] ? 1 : 0;
        } else {
            int interval = tc[@"full_attention_interval"] ? [tc[@"full_attention_interval"] intValue] : 4;
            for (int i = 0; i < n_layers; ++i) ltypes[i] = (((i + 1) % interval) == 0) ? 1 : 0;
        }

        // EOS detection (config + generation_config + tokenizer.json)
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
                if ([c isEqualToString:@"<|im_end|>"] || [c isEqualToString:@"<|endoftext|>"]) add_eos(at[@"id"]);
            }
        }
        if (eos_ids.empty()) { eos_ids = {248046, 248044}; }

        // Load shard index & mmap every shard once
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

        struct RawRef {
            int shard;
            uint64_t off, len;
            std::string dtype;
            std::vector<int64_t> shape;
        };
        auto find_tensor = [&](const std::string& k) -> RawRef {
            NSString* nk = [NSString stringWithUTF8String:k.c_str()];
            NSString* fn = wmap[nk];
            if (!fn) { fprintf(stderr, "Missing safetensors key: %s\n", k.c_str()); exit(1); }
            int si = open_shard([fn UTF8String]);
            NSDictionary* info = mapped_files[si].header[nk];
            if (!info) { fprintf(stderr, "Missing header entry: %s\n", k.c_str()); exit(1); }
            NSArray* offs = info[@"data_offsets"];
            uint64_t s = [offs[0] unsignedLongLongValue], e = [offs[1] unsignedLongLongValue];
            RawRef r{si, mapped_files[si].data_off + s, e - s, [info[@"dtype"] UTF8String], {}};
            for (NSNumber* d in info[@"shape"]) r.shape.push_back([d longLongValue]);
            return r;
        };

        // Accumulate small converted 1D/conv/RoPE tensors into a single compact RAM MTLBuffer (<40 MB!)
        std::vector<uint8_t> small_blob;
        small_blob.reserve(48 * 1024 * 1024);
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
            if (r.dtype == "F16") {
                memcpy(out.data(), ptr, n * 2);
            } else if (r.dtype == "BF16") {
                const uint16_t* p16 = (const uint16_t*)ptr;
                for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16_bits(bf16_to_f32(p16[i]));
            } else if (r.dtype == "F32") {
                const float* pf = (const float*)ptr;
                for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16_bits(pf[i]);
            } else {
                fprintf(stderr, "Unsupported small tensor dtype %s for %s\n", r.dtype.c_str(), k.c_str()); exit(1);
            }
            return append_small(out.data(), n * 2);
        };

        auto load_small_f32 = [&](const std::string& k) -> uint64_t {
            RawRef r = find_tensor(k);
            const uint8_t* ptr = (const uint8_t*)mapped_files[r.shard].mapped + r.off;
            size_t n = 1; for (auto d : r.shape) n *= size_t(d);
            std::vector<float> out(n);
            if (r.dtype == "F32") {
                memcpy(out.data(), ptr, n * 4);
            } else if (r.dtype == "BF16") {
                const uint16_t* p16 = (const uint16_t*)ptr;
                for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p16[i]);
            } else if (r.dtype == "F16") {
                const uint16_t* p16 = (const uint16_t*)ptr;
                for (size_t i = 0; i < n; ++i) out[i] = f16_bits_to_f32(p16[i]);
            } else {
                fprintf(stderr, "Unsupported f32 tensor dtype %s for %s\n", r.dtype.c_str(), k.c_str()); exit(1);
            }
            return append_small(out.data(), n * 4);
        };

        // Build Standard or YaRN Partial RoPE cos/sin table directly in C++!
        NSDictionary* rp = tc[@"rope_parameters"];
        double prf = tc[@"partial_rotary_factor"] ? [tc[@"partial_rotary_factor"] doubleValue]
                   : (rp && rp[@"partial_rotary_factor"] ? [rp[@"partial_rotary_factor"] doubleValue] : 0.25);
        rope_dim = int(std::llround(head_dim * prf));
        double theta = (rp && rp[@"rope_theta"]) ? [rp[@"rope_theta"] doubleValue]
                     : (tc[@"rope_theta"] ? [tc[@"rope_theta"] doubleValue] : 10000000.0);
        int half_r = rope_dim / 2;
        std::vector<double> inv_freq(half_r);
        for (int i = 0; i < half_r; ++i) inv_freq[i] = 1.0 / std::pow(theta, double(2 * i) / double(rope_dim));
        double mscale = 1.0;
        if (rp && rp[@"rope_type"] && [rp[@"rope_type"] isEqualToString:@"yarn"] && rp[@"factor"] && [rp[@"factor"] doubleValue] > 1.0) {
            double factor   = [rp[@"factor"] doubleValue];
            double orig_max = rp[@"original_max_position_embeddings"] ? [rp[@"original_max_position_embeddings"] doubleValue] : 262144.0;
            double b_fast   = rp[@"beta_fast"] ? [rp[@"beta_fast"] doubleValue] : 32.0;
            double b_slow   = rp[@"beta_slow"] ? [rp[@"beta_slow"] doubleValue] : 1.0;
            auto corr = [&](double nr) { return (rope_dim * std::log(orig_max / (nr * 2.0 * M_PI))) / (2.0 * std::log(theta)); };
            double low = std::max(std::floor(corr(b_fast)), 0.0);
            double high = std::min(std::ceil(corr(b_slow)), double(rope_dim - 1));
            if (low == high) high += 0.001;
            for (int i = 0; i < half_r; ++i) {
                double ramp = 1.0 - std::min(1.0, std::max(0.0, (double(i) - low) / (high - low)));
                inv_freq[i] = (inv_freq[i] / factor) * (1.0 - ramp) + inv_freq[i] * ramp;
            }
            mscale = rp[@"attention_factor"] ? [rp[@"attention_factor"] doubleValue] : (0.1 * std::log(factor) + 1.0);
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

        // Zero-copy slice binding with automatic alignment guard (8-byte for weight, 2-byte for scales/biases)
        int misaligned_copies = 0;
        auto bind_zero_copy = [&](const RawRef& r, size_t req_align) -> BufSlice {
            if ((r.off % req_align) == 0) {
                return {mapped_files[r.shard].mtl_buf, r.off};
            }
            misaligned_copies++;
            id<MTLBuffer> fb = [dev newBufferWithBytes:(const uint8_t*)mapped_files[r.shard].mapped + r.off
                                                length:r.len options:MTLResourceStorageModeShared];
            fallback_bufs.push_back(fb);
            return {fb, 0};
        };

        bool dtype_detected = false;
        auto qt_from_st = [&](const std::string& st_pfx, int K_expect, bool is_switch = false) -> QTensor {
            RawRef rw = find_tensor(st_pfx + ".weight");
            RawRef rs = find_tensor(st_pfx + ".scales");
            RawRef rb = find_tensor(st_pfx + ".biases");
            if (!dtype_detected) {
                scales_bf16 = (rs.dtype == "BF16");
                dtype_detected = true;
            }
            int kw = int(rw.shape.back());
            int kg = int(rs.shape.back());
            int N  = int(rw.shape[rw.shape.size() - 2]);
            int gs = K_expect / kg;
            int bits = (kw * 32) / K_expect;
            if (bits != 2 && bits != 3 && bits != 4 && bits != 8) {
                fprintf(stderr, "Unsupported bits=%d for %s\n", bits, st_pfx.c_str()); exit(1);
            }
            BufSlice sw = bind_zero_copy(rw, 4);
            BufSlice ss = bind_zero_copy(rs, 2);
            BufSlice sb = bind_zero_copy(rb, 2);
            QTensor q{};
            q.w_buf = sw.buf; q.w = sw.off;
            q.s_buf = ss.buf; q.s = ss.off;
            q.b_buf = sb.buf; q.b = sb.off;
            q.bits = bits; q.gs = gs; q.gs_shift = (gs == 32) ? 5 : 6;
            q.N = N; q.K = K_expect;
            if (is_switch) {
                q.stride_w = uint64_t(N) * uint64_t(kw);
                q.stride_s = uint64_t(N) * uint64_t(kg);
            }
            return q;
        };

        embed = qt_from_st(pfx + "model.embed_tokens", dim);
        lm_head = has_key(pfx + "lm_head.weight") ? qt_from_st(pfx + "lm_head", dim) : embed;

        int val_lin = lin_vh * lin_vhd;
        int q_full  = n_heads * head_dim;

        // Track small-buffer offsets and patch `buf` pointer once small_buf is created
        struct SmallRef { BufSlice* dst; uint64_t off; };
        std::vector<SmallRef> small_refs;
        small_refs.push_back({&cos_s, cos_off_sm});
        small_refs.push_back({&sin_s, sin_off_sm});
        small_refs.push_back({&fnorm_s, fn_off_sm});

        layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            std::string sp = pfx + "model.layers." + std::to_string(i);
            auto& L = layers[i];
            L.type = ltypes[i];
            small_refs.push_back({&L.in_norm,   load_small_f16(sp + ".input_layernorm.weight")});
            small_refs.push_back({&L.post_norm, load_small_f16(sp + ".post_attention_layernorm.weight")});
            if (L.type == 0) {
                std::string la = sp + ".linear_attn";
                L.in_qkv = qt_from_st(la + ".in_proj_qkv", dim);
                L.in_z   = qt_from_st(la + ".in_proj_z",   dim);
                L.in_a   = qt_from_st(la + ".in_proj_a",   dim);
                L.in_b   = qt_from_st(la + ".in_proj_b",   dim);
                L.out_p  = qt_from_st(la + ".out_proj",    val_lin);
                small_refs.push_back({&L.conv_w,  load_small_f16(la + ".conv1d.weight")});
                small_refs.push_back({&L.A_log,   load_small_f32(la + ".A_log")});
                small_refs.push_back({&L.dt_bias, load_small_f32(la + ".dt_bias")});
                small_refs.push_back({&L.dn_norm, load_small_f16(la + ".norm.weight")});
            } else {
                std::string sa = sp + ".self_attn";
                L.q = qt_from_st(sa + ".q_proj", dim);
                L.k = qt_from_st(sa + ".k_proj", dim);
                L.v = qt_from_st(sa + ".v_proj", dim);
                L.o = qt_from_st(sa + ".o_proj", q_full);
                small_refs.push_back({&L.q_norm, load_small_f16(sa + ".q_norm.weight")});
                small_refs.push_back({&L.k_norm, load_small_f16(sa + ".k_norm.weight")});
            }
            std::string mp = sp + ".mlp";
            if (is_moe) {
                L.router  = qt_from_st(mp + ".gate", dim);
                L.sh_gate = qt_from_st(mp + ".shared_expert_gate", dim);
                L.sh_gp   = qt_from_st(mp + ".shared_expert.gate_proj", dim);
                L.sh_up   = qt_from_st(mp + ".shared_expert.up_proj",   dim);
                L.sh_dp   = qt_from_st(mp + ".shared_expert.down_proj", shared_inter);
                L.moe_gp  = qt_from_st(mp + ".switch_mlp.gate_proj", dim, true);
                L.moe_up  = qt_from_st(mp + ".switch_mlp.up_proj",   dim, true);
                L.moe_dp  = qt_from_st(mp + ".switch_mlp.down_proj", moe_inter, true);
            } else {
                L.gate = qt_from_st(mp + ".gate_proj", dim);
                L.up   = qt_from_st(mp + ".up_proj",   dim);
                L.down = qt_from_st(mp + ".down_proj", dense_inter);
            }
        }

        id<MTLBuffer> small_buf = [dev newBufferWithBytes:small_blob.data() length:small_blob.size() options:MTLResourceStorageModeShared];
        fallback_bufs.push_back(small_buf);
        for (auto& sr : small_refs) { sr.dst->buf = small_buf; sr.dst->off = sr.off; }

        if (misaligned_copies > 0) {
            fprintf(stderr, "  [note] %d tensors copied for alignment (shard data_off%%4: ", misaligned_copies);
            for (size_t si = 0; si < mapped_files.size(); ++si) {
                fprintf(stderr, "s%zu=%llu%s", si, (unsigned long long)(mapped_files[si].data_off % 4),
                        (si + 1 < mapped_files.size()) ? "," : "");
            }
            fprintf(stderr, ")\n");
        }
        return true;
    }

    bool load(const std::string& path, int override_topk = 0, bool force_willneed = false, int req_max_seq = 4096) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];

        struct stat st{};
        if (stat(path.c_str(), &st) != 0) { perror(("stat " + path).c_str()); return false; }

        bool ok = S_ISDIR(st.st_mode)
            ? load_safetensors_dir(path, override_topk, force_willneed, req_max_seq)
            : load_taalas2(path, override_topk, force_willneed);
        if (!ok) return false;

        qkv_lin_dim = 2 * lin_kh * lin_khd + lin_vh * lin_vhd;
        val_lin_dim = lin_vh * lin_vhd;
        q_full_dim  = n_heads * head_dim;
        kv_full_dim = n_kv_heads * head_dim;

        auto try_read = [](const std::string& p) -> NSString* {
            if (p.empty()) return nil;
            return [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:p.c_str()] encoding:NSUTF8StringEncoding error:nil];
        };
        NSString* shaderSrc = nil;
        if (const char* env_dir = getenv("BIHABA_SHADER_DIR")) {
            shaderSrc = try_read(std::string(env_dir) + "/kernels.metal");
            if (!shaderSrc) shaderSrc = try_read(std::string(env_dir) + "/qwen35_kernels.metal");
        }
        if (!shaderSrc) {
            char exe_buf[4096]; uint32_t exe_sz = sizeof(exe_buf);
            if (_NSGetExecutablePath(exe_buf, &exe_sz) == 0) {
                char real_buf[4096];
                const char* resolved = realpath(exe_buf, real_buf) ? real_buf : exe_buf;
                std::string exe_dir = dirname(const_cast<char*>(resolved));
                shaderSrc = try_read(exe_dir + "/qwen35_kernels.metal");
                if (!shaderSrc) shaderSrc = try_read(exe_dir + "/../engines/qwen35/kernels.metal");
                if (!shaderSrc) shaderSrc = try_read(exe_dir + "/../share/bihaba-serve/qwen35_kernels.metal");
            }
        }
        if (!shaderSrc) shaderSrc = try_read("engines/qwen35/kernels.metal");
        if (!shaderSrc) shaderSrc = try_read("kernels.metal");
        if (!shaderSrc) {
            fprintf(stderr, "Cannot locate qwen35 kernels.metal (set BIHABA_SHADER_DIR or run via bihaba-serve)\n");
            return false;
        }

        NSError* serr = nil;
        MTLCompileOptions* opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) { opts.mathMode = MTLMathModeFast; }
        id<MTLLibrary> lib = [dev newLibraryWithSource:shaderSrc options:opts error:&serr];
        if (!lib) { fprintf(stderr, "Shader compile error:\n%s\n", [[serr localizedDescription] UTF8String]); return false; }

        const bool bf = scales_bf16;
        pso_embed         = make_pso(dev, lib, "embed_lookup_q",          false, bf);
        pso_rms           = make_pso(dev, lib, "rms_norm",                false, bf);
        pso_rep           = make_pso(dev, lib, "apply_rep_penalty_sparse",false, bf);
        pso_argmax        = make_pso(dev, lib, "argmax_full",             false, bf);
        pso_topk64        = make_pso(dev, lib, "topk_logits_64",          false, bf);
        pso_dn_in4        = make_pso(dev, lib, "deltanet_in_proj_4b",     false, bf);
        pso_dn_in4_m1     = make_pso(dev, lib, "deltanet_in_proj_4b_m1",  false, bf);
        pso_dn_in4_mma8   = make_pso(dev, lib, "deltanet_in_proj_4b_mma8",false, bf);
        pso_full_qkv4     = make_pso(dev, lib, "q35_qkv_proj_4b",         false, bf);
        pso_full_qkv4_m1  = make_pso(dev, lib, "q35_qkv_proj_4b_m1",      false, bf);
        pso_full_qkv4_mma8= make_pso(dev, lib, "q35_qkv_proj_4b_mma8",    false, bf);
        pso_qmm4_mma8     = make_pso(dev, lib, "qmm_4b_mma8",             false, bf);
        pso_qmm4_mma8_add = make_pso(dev, lib, "qmm_4b_mma8",             true,  bf);
        pso_qmm2_mma8     = make_pso(dev, lib, "qmm_2b_mma8",             false, bf);
        pso_qmm2_mma8_add = make_pso(dev, lib, "qmm_2b_mma8",             true,  bf);
        pso_sw4_mma8      = make_pso(dev, lib, "dense_swiglu_4b_mma8",    false, bf);
        pso_sw2_mma8      = make_pso(dev, lib, "dense_swiglu_2b_mma8",    false, bf);
        pso_conv1d_m1     = make_pso(dev, lib, "deltanet_conv1d_silu_m1", false, bf);
        pso_conv1d_spec   = make_pso(dev, lib, "deltanet_conv1d_silu_spec",false,bf);
        pso_dn_m1         = make_pso(dev, lib, "deltanet_recurrent_m1",   false, bf);
        pso_dn_spec       = make_pso(dev, lib, "deltanet_recurrent_spec", false, bf);
        pso_commit_conv   = make_pso(dev, lib, "deltanet_commit_conv1d",  false, bf);
        pso_commit_dn     = make_pso(dev, lib, "deltanet_commit_state",   false, bf);
        pso_qk_rope       = make_pso(dev, lib, "q35_qknorm_rope_kv",      false, bf);
        pso_gqa256        = make_pso(dev, lib, "gqa_attention_256_gated", false, bf);
        pso_router_sh4    = make_pso(dev, lib, "moe_router_shgate_4b",    false, bf);
        pso_router_sh8    = make_pso(dev, lib, "moe_router_shgate_8b",    false, bf);
        pso_topk          = make_pso(dev, lib, "moe_topk_and_shgate",     false, bf);
        pso_combine       = make_pso(dev, lib, "moe_combine_res",         false, bf);

        const int bvals[4] = {2, 3, 4, 8};
        const char* qmv1_n[4] = {"qmv_2b",   "qmv_3b",   "qmv_4b",   "qmv_8b"};
        const char* qmvm_n[4] = {"qmv_2b_m", "qmv_3b_m", "qmv_4b_m", "qmv_8b_m"};
        const char* sw_n[4]   = {"dense_swiglu_2b", "dense_swiglu_3b", "dense_swiglu_4b", "dense_swiglu_8b"};
        for (int i = 0; i < 4; ++i) {
            int b = bvals[i];
            pso_qmv1[b]     = make_pso(dev, lib, qmv1_n[i], false, bf);
            pso_qmv1_add[b] = make_pso(dev, lib, qmv1_n[i], true,  bf);
            pso_qmvm[b]     = make_pso(dev, lib, qmvm_n[i], false, bf);
            pso_qmvm_add[b] = make_pso(dev, lib, qmvm_n[i], true,  bf);
            pso_sw[b]       = make_pso(dev, lib, sw_n[i],   false, bf);
        }
        pso_moe_sw1[2] = make_pso(dev, lib, "moe_swiglu_2b",     false, bf);
        pso_moe_sw1[3] = make_pso(dev, lib, "moe_swiglu_3b",     false, bf);
        pso_moe_sw1[4] = make_pso(dev, lib, "moe_swiglu_4b",     false, bf);
        pso_moe_dn1[2] = make_pso(dev, lib, "moe_down_2d_2b",    false, bf);
        pso_moe_dn1[3] = make_pso(dev, lib, "moe_down_2d_3b",    false, bf);
        pso_moe_dn1[4] = make_pso(dev, lib, "moe_down_2d_4b",    false, bf);

        auto mkbuf = [&](size_t bytes) -> id<MTLBuffer> {
            return [dev newBufferWithLength:std::max<size_t>(bytes, 256) options:MTLResourceStorageModeShared];
        };
        b_tok       = mkbuf(MAX_M * sizeof(int32_t));
        b_x         = mkbuf(MAX_M * dim * 2);
        b_xn        = mkbuf(MAX_M * dim * 2);
        b_qkv       = mkbuf(MAX_M * qkv_lin_dim * 2);
        b_z         = mkbuf(MAX_M * val_lin_dim * 2);
        b_a         = mkbuf(MAX_M * lin_vh * 2);
        b_b         = mkbuf(MAX_M * lin_vh * 2);
        b_dn_out    = mkbuf(MAX_M * val_lin_dim * 2);
        b_q_raw     = mkbuf(MAX_M * 2 * q_full_dim * 2);
        b_q         = mkbuf(MAX_M * q_full_dim * 2);
        b_gate      = mkbuf(MAX_M * q_full_dim * 2);
        b_k         = mkbuf(MAX_M * kv_full_dim * 2);
        b_v         = mkbuf(MAX_M * kv_full_dim * 2);
        b_att       = mkbuf(MAX_M * q_full_dim * 2);
        b_router_l  = mkbuf(MAX_M * std::max(num_experts, 256) * 2);
        b_sh_gl     = mkbuf(MAX_M * 16);
        b_top_ids   = mkbuf(MAX_M * std::max(top_k, 16) * sizeof(int32_t));
        b_top_w     = mkbuf(MAX_M * std::max(top_k, 16) * sizeof(float));
        b_sh_g      = mkbuf(MAX_M * sizeof(float));
        b_uniq_cnt  = mkbuf(sizeof(int32_t));
        b_uniq_e    = mkbuf(MAX_M * 16 * sizeof(int32_t));
        b_grp_cnt   = mkbuf(MAX_M * 16 * sizeof(int32_t));
        b_grp_slots = mkbuf(MAX_M * 16 * 8 * sizeof(int32_t));
        b_sh_h      = mkbuf(MAX_M * std::max(shared_inter, 64) * 2);
        b_sh_out    = mkbuf(MAX_M * dim * 2);
        b_moe_h     = mkbuf(MAX_M * std::max(top_k * moe_inter, 64) * 2);
        b_moe_out   = mkbuf(MAX_M * std::max(top_k * dim, 64) * 2);
        b_dense_h   = mkbuf(MAX_M * std::max(dense_inter, 64) * 2);
        b_logits     = mkbuf(MAX_M * vocab * 2);
        b_out_tok    = mkbuf(MAX_M * sizeof(int32_t));
        b_top64_ids  = mkbuf(MAX_M * 64 * sizeof(int32_t));
        b_top64_vals = mkbuf(MAX_M * 64 * sizeof(float));

        conv_states.resize(n_layers, nil); dn_states.resize(n_layers, nil);
        k_caches.resize(n_layers, nil);    v_caches.resize(n_layers, nil);
        conv_hist.resize(n_layers, nil);   dn_hist_k.resize(n_layers, nil);
        dn_hist_d.resize(n_layers, nil);   dn_hist_dec.resize(n_layers, nil);

        for (int i = 0; i < n_layers; ++i) {
            if (layers[i].type == 0) {
                conv_states[i] = mkbuf(size_t(conv_k - 1) * size_t(qkv_lin_dim) * 2);
                dn_states[i]   = mkbuf(size_t(lin_vh) * size_t(lin_vhd) * size_t(lin_khd) * sizeof(float));
                conv_hist[i]   = mkbuf(size_t(MAX_M) * size_t(qkv_lin_dim) * 2);
                dn_hist_k[i]   = mkbuf(size_t(MAX_M) * size_t(lin_vh) * size_t(lin_khd) * sizeof(float));
                dn_hist_d[i]   = mkbuf(size_t(MAX_M) * size_t(lin_vh) * size_t(lin_vhd) * sizeof(float));
                dn_hist_dec[i] = mkbuf(size_t(MAX_M) * size_t(lin_vh) * sizeof(float));
            } else {
                size_t kv_bytes = size_t(n_kv_heads) * size_t(max_seq) * size_t(head_dim) * 2;
                k_caches[i] = mkbuf(kv_bytes); v_caches[i] = mkbuf(kv_bytes);
            }
        }
        reset_states();

        double load_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        fprintf(stderr, "▸ bihaba-serve [Taalas-Metal v1.0] (%s, %s, BakedState+Top64) Ready on %s (cold: %.1f ms | %.2f GB / %zu shards)\n",
                direct_st_mode ? "Direct .safetensors 0-Dup" : "TAALAS02",
                scales_bf16 ? "Native-BF16" : "FP16",
                [[dev name] UTF8String], load_ms, total_mapped_bytes / 1e9, mapped_files.size());
        fprintf(stderr, "  arch=%s | %dL dim=%d hd=%d | %s | vocab=%d\n",
                arch_name.c_str(), n_layers, dim, head_dim,
                is_moe ? (std::string("MoE E=") + std::to_string(num_experts) + " topK=" + std::to_string(top_k)).c_str()
                       : (std::string("Dense inter=") + std::to_string(dense_inter)).c_str(), vocab);
        return true;
    }

    void reset_states() {
        for (int i = 0; i < n_layers; ++i) {
            if (conv_states[i]) memset([conv_states[i] contents], 0, [conv_states[i] length]);
            if (dn_states[i])   memset([dn_states[i] contents],   0, [dn_states[i] length]);
        }
        pending_dn_commit = 0;
        cached_ids.clear();
    }

    void flush_pending_commit() {
        if (pending_dn_commit <= 0) return;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        enc_commit_deltanet(e, pending_dn_commit);
        pending_dn_commit = 0;
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    void capture_snapshot(StateSnapshot& snap, const std::vector<int32_t>& ids, int pos) {
        flush_pending_commit();
        snap.ids = ids;
        snap.pos = pos;
        snap.conv_data.assign(n_layers, {});
        snap.dn_data.assign(n_layers, {});
        snap.k_data.assign(n_layers, {});
        snap.v_data.assign(n_layers, {});
        const size_t head_stride_bytes = size_t(max_seq) * size_t(head_dim) * 2;
        const size_t pos_bytes = size_t(pos) * size_t(head_dim) * 2;
        for (int l = 0; l < n_layers; ++l) {
            if (layers[l].type == 0) {
                size_t cl = [conv_states[l] length];
                snap.conv_data[l].resize(cl);
                memcpy(snap.conv_data[l].data(), [conv_states[l] contents], cl);
                size_t dl = [dn_states[l] length];
                snap.dn_data[l].resize(dl);
                memcpy(snap.dn_data[l].data(), [dn_states[l] contents], dl);
            } else {
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
    }

    void restore_snapshot(const StateSnapshot& snap) {
        pending_dn_commit = 0;
        cached_ids = snap.ids;
        const size_t head_stride_bytes = size_t(max_seq) * size_t(head_dim) * 2;
        const size_t pos_bytes = size_t(snap.pos) * size_t(head_dim) * 2;
        for (int l = 0; l < n_layers; ++l) {
            if (layers[l].type == 0) {
                memcpy([conv_states[l] contents], snap.conv_data[l].data(), snap.conv_data[l].size());
                memcpy([dn_states[l] contents],   snap.dn_data[l].data(),   snap.dn_data[l].size());
            } else {
                uint8_t* kp = (uint8_t*)[k_caches[l] contents];
                uint8_t* vp = (uint8_t*)[v_caches[l] contents];
                for (int h = 0; h < n_kv_heads; ++h) {
                    memcpy(kp + size_t(h) * head_stride_bytes, snap.k_data[l].data() + size_t(h) * pos_bytes, pos_bytes);
                    memcpy(vp + size_t(h) * head_stride_bytes, snap.v_data[l].data() + size_t(h) * pos_bytes, pos_bytes);
                }
            }
        }
    }

    bool save_snapshot_file(const StateSnapshot& snap, const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { perror(("save " + path).c_str()); return false; }
        const char magic[8] = {'T','S','T','A','T','E','0','1'};
        int32_t hdr[6] = {n_layers, dim, n_kv_heads, head_dim, snap.pos, int32_t(snap.ids.size())};
        fwrite(magic, 1, 8, f);
        fwrite(hdr, sizeof(int32_t), 6, f);
        fwrite(snap.ids.data(), sizeof(int32_t), snap.ids.size(), f);
        for (int l = 0; l < n_layers; ++l) {
            if (layers[l].type == 0) {
                uint64_t sz[2] = {snap.conv_data[l].size(), snap.dn_data[l].size()};
                fwrite(sz, sizeof(uint64_t), 2, f);
                fwrite(snap.conv_data[l].data(), 1, sz[0], f);
                fwrite(snap.dn_data[l].data(),   1, sz[1], f);
            } else {
                uint64_t sz[2] = {snap.k_data[l].size(), snap.v_data[l].size()};
                fwrite(sz, sizeof(uint64_t), 2, f);
                fwrite(snap.k_data[l].data(), 1, sz[0], f);
                fwrite(snap.v_data[l].data(), 1, sz[1], f);
            }
        }
        fclose(f);
        return true;
    }

    bool load_snapshot_file(StateSnapshot& snap, const std::string& path) const {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) { perror(("open " + path).c_str()); return false; }
        char magic[8] = {0};
        int32_t hdr[6] = {0};
        if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "TSTATE01", 8) != 0) { fclose(f); return false; }
        if (fread(hdr, sizeof(int32_t), 6, f) != 6) { fclose(f); return false; }
        if (hdr[0] != n_layers || hdr[1] != dim || hdr[2] != n_kv_heads || hdr[3] != head_dim || hdr[4] >= max_seq) {
            fprintf(stderr, "State file architecture mismatch!\n"); fclose(f); return false;
        }
        snap.pos = hdr[4];
        snap.ids.resize(size_t(hdr[5]));
        if (fread(snap.ids.data(), sizeof(int32_t), snap.ids.size(), f) != snap.ids.size()) { fclose(f); return false; }
        snap.conv_data.assign(n_layers, {}); snap.dn_data.assign(n_layers, {});
        snap.k_data.assign(n_layers, {});    snap.v_data.assign(n_layers, {});
        for (int l = 0; l < n_layers; ++l) {
            uint64_t sz[2] = {0, 0};
            if (fread(sz, sizeof(uint64_t), 2, f) != 2) { fclose(f); return false; }
            if (layers[l].type == 0) {
                snap.conv_data[l].resize(sz[0]); snap.dn_data[l].resize(sz[1]);
                if (fread(snap.conv_data[l].data(), 1, sz[0], f) != sz[0]) { fclose(f); return false; }
                if (fread(snap.dn_data[l].data(),   1, sz[1], f) != sz[1]) { fclose(f); return false; }
            } else {
                snap.k_data[l].resize(sz[0]); snap.v_data[l].resize(sz[1]);
                if (fread(snap.k_data[l].data(), 1, sz[0], f) != sz[0]) { fclose(f); return false; }
                if (fread(snap.v_data[l].data(), 1, sz[1], f) != sz[1]) { fclose(f); return false; }
            }
        }
        fclose(f);
        return true;
    }

    inline void bar(id<MTLComputeCommandEncoder> e) {
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    void enc_commit_deltanet(id<MTLComputeCommandEncoder> e, int n_acc) {
        if (n_acc <= 0) return;
        for (int l = 0; l < n_layers; ++l) {
            if (layers[l].type != 0) continue;
            Conv1dParams cp{n_acc, qkv_lin_dim, conv_k};
            [e setComputePipelineState:pso_commit_conv];
            [e setBuffer:conv_states[l] offset:0 atIndex:0];
            [e setBuffer:conv_hist[l]   offset:0 atIndex:1];
            [e setBytes:&cp length:sizeof(cp) atIndex:2];
            [e dispatchThreads:MTLSizeMake(qkv_lin_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            DeltaNetParams dp{n_acc, lin_kh, lin_vh, lin_khd, lin_vhd, rms_eps};
            [e setComputePipelineState:pso_commit_dn];
            [e setBuffer:dn_states[l]   offset:0 atIndex:0];
            [e setBuffer:dn_hist_k[l]   offset:0 atIndex:1];
            [e setBuffer:dn_hist_d[l]   offset:0 atIndex:2];
            [e setBuffer:dn_hist_dec[l] offset:0 atIndex:3];
            [e setBytes:&dp length:sizeof(dp) atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(lin_vh, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        bar(e);
    }

    void enc_qmv(id<MTLComputeCommandEncoder> e, int M, id<MTLBuffer> inp, const QTensor& qt, id<MTLBuffer> outp, bool add_res) {
        if (M == 1) {
            QMVParams p{qt.K, qt.N, qt.gs_shift};
            [e setComputePipelineState:(add_res ? pso_qmv1_add[qt.bits] : pso_qmv1[qt.bits])];
            [e setBuffer:inp offset:0 atIndex:0];
            [e setBuffer:qt.w_buf offset:qt.w atIndex:1]; [e setBuffer:qt.s_buf offset:qt.s atIndex:2]; [e setBuffer:qt.b_buf offset:qt.b atIndex:3];
            [e setBuffer:outp offset:0 atIndex:4]; [e setBytes:&p length:sizeof(p) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((qt.N + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (M >= 4 && (qt.bits == 4 || qt.bits == 2) && (qt.K % 64 == 0)) {
            QMVMultiParams p{M, qt.K, qt.N, qt.gs_shift};
            id<MTLComputePipelineState> pso = (qt.bits == 4)
                ? (add_res ? pso_qmm4_mma8_add : pso_qmm4_mma8)
                : (add_res ? pso_qmm2_mma8_add : pso_qmm2_mma8);
            [e setComputePipelineState:pso];
            [e setBuffer:inp offset:0 atIndex:0];
            [e setBuffer:qt.w_buf offset:qt.w atIndex:1]; [e setBuffer:qt.s_buf offset:qt.s atIndex:2]; [e setBuffer:qt.b_buf offset:qt.b atIndex:3];
            [e setBuffer:outp offset:0 atIndex:4]; [e setBytes:&p length:sizeof(p) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((qt.N + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            QMVMultiParams p{M, qt.K, qt.N, qt.gs_shift};
            [e setComputePipelineState:(add_res ? pso_qmvm_add[qt.bits] : pso_qmvm[qt.bits])];
            [e setBuffer:inp offset:0 atIndex:0];
            [e setBuffer:qt.w_buf offset:qt.w atIndex:1]; [e setBuffer:qt.s_buf offset:qt.s atIndex:2]; [e setBuffer:qt.b_buf offset:qt.b atIndex:3];
            [e setBuffer:outp offset:0 atIndex:4]; [e setBytes:&p length:sizeof(p) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake((qt.N + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }

    void enc_swiglu(id<MTLComputeCommandEncoder> e, int M, id<MTLBuffer> inp, const QTensor& gp, const QTensor& up, id<MTLBuffer> outp) {
        QMVMultiParams p{M, gp.K, gp.N, gp.gs_shift};
        if (M >= 4 && (gp.bits == 4 || gp.bits == 2) && gp.bits == up.bits && (gp.K % 64 == 0)) {
            [e setComputePipelineState:(gp.bits == 4 ? pso_sw4_mma8 : pso_sw2_mma8)];
            [e setBuffer:inp offset:0 atIndex:0];
            [e setBuffer:gp.w_buf offset:gp.w atIndex:1]; [e setBuffer:gp.s_buf offset:gp.s atIndex:2]; [e setBuffer:gp.b_buf offset:gp.b atIndex:3];
            [e setBuffer:up.w_buf offset:up.w atIndex:4]; [e setBuffer:up.s_buf offset:up.s atIndex:5]; [e setBuffer:up.b_buf offset:up.b atIndex:6];
            [e setBuffer:outp offset:0 atIndex:7]; [e setBytes:&p length:sizeof(p) atIndex:8];
            [e dispatchThreadgroups:MTLSizeMake((gp.N + 31) / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [e setComputePipelineState:pso_sw[gp.bits]];
            [e setBuffer:inp offset:0 atIndex:0];
            [e setBuffer:gp.w_buf offset:gp.w atIndex:1]; [e setBuffer:gp.s_buf offset:gp.s atIndex:2]; [e setBuffer:gp.b_buf offset:gp.b atIndex:3];
            [e setBuffer:up.w_buf offset:up.w atIndex:4]; [e setBuffer:up.s_buf offset:up.s atIndex:5]; [e setBuffer:up.b_buf offset:up.b atIndex:6];
            [e setBuffer:outp offset:0 atIndex:7]; [e setBytes:&p length:sizeof(p) atIndex:8];
            [e dispatchThreadgroups:MTLSizeMake((gp.N + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }

    void enc_layer_attn(id<MTLComputeCommandEncoder> e, int M, int l, int pos) {
        const auto& L = layers[l];
        NormParams np{dim, rms_eps};
        [e setComputePipelineState:pso_rms];
        [e setBuffer:b_x offset:0 atIndex:0]; [e setBuffer:L.in_norm.buf offset:L.in_norm.off atIndex:1];
        [e setBuffer:b_xn offset:0 atIndex:2]; [e setBytes:&np length:sizeof(np) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        bar(e);

        if (L.type == 0) {
            if (L.in_qkv.bits == 4 && L.in_z.bits == 4 && L.in_a.bits == 4 && L.in_b.bits == 4 &&
                L.in_qkv.gs_shift == L.in_z.gs_shift && L.in_qkv.gs_shift == L.in_a.gs_shift) {
                DeltaNetInParams dp{M, dim, L.in_qkv.gs_shift, L.in_qkv.N, L.in_z.N, L.in_a.N, L.in_b.N};
                int tot_n = L.in_qkv.N + L.in_z.N + L.in_a.N + L.in_b.N;
                id<MTLComputePipelineState> pso_dn = (M == 1) ? pso_dn_in4_m1 : ((M >= 4 && (dim % 64 == 0)) ? pso_dn_in4_mma8 : pso_dn_in4);
                int rpt_dn = (M == 1) ? 8 : ((M >= 4 && (dim % 64 == 0)) ? 32 : 4);
                [e setComputePipelineState:pso_dn];
                [e setBuffer:b_xn offset:0 atIndex:0];
                [e setBuffer:L.in_qkv.w_buf offset:L.in_qkv.w atIndex:1];  [e setBuffer:L.in_qkv.s_buf offset:L.in_qkv.s atIndex:2];  [e setBuffer:L.in_qkv.b_buf offset:L.in_qkv.b atIndex:3];
                [e setBuffer:L.in_z.w_buf   offset:L.in_z.w   atIndex:4];  [e setBuffer:L.in_z.s_buf   offset:L.in_z.s   atIndex:5];  [e setBuffer:L.in_z.b_buf   offset:L.in_z.b   atIndex:6];
                [e setBuffer:L.in_a.w_buf   offset:L.in_a.w   atIndex:7];  [e setBuffer:L.in_a.s_buf   offset:L.in_a.s   atIndex:8];  [e setBuffer:L.in_a.b_buf   offset:L.in_a.b   atIndex:9];
                [e setBuffer:L.in_b.w_buf   offset:L.in_b.w   atIndex:10]; [e setBuffer:L.in_b.s_buf   offset:L.in_b.s   atIndex:11]; [e setBuffer:L.in_b.b_buf   offset:L.in_b.b   atIndex:12];
                [e setBuffer:b_qkv offset:0 atIndex:13]; [e setBuffer:b_z offset:0 atIndex:14];
                [e setBuffer:b_a   offset:0 atIndex:15]; [e setBuffer:b_b offset:0 atIndex:16];
                [e setBytes:&dp length:sizeof(dp) atIndex:17];
                [e dispatchThreadgroups:MTLSizeMake((tot_n + rpt_dn - 1) / rpt_dn, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                enc_qmv(e, M, b_xn, L.in_qkv, b_qkv, false);
                enc_qmv(e, M, b_xn, L.in_z,   b_z,   false);
                enc_qmv(e, M, b_xn, L.in_a,   b_a,   false);
                enc_qmv(e, M, b_xn, L.in_b,   b_b,   false);
            }
            bar(e);

            Conv1dParams cp{M, qkv_lin_dim, conv_k};
            if (M == 1) {
                [e setComputePipelineState:pso_conv1d_m1];
                [e setBuffer:b_qkv offset:0 atIndex:0]; [e setBuffer:conv_states[l] offset:0 atIndex:1];
                [e setBuffer:L.conv_w.buf offset:L.conv_w.off atIndex:2]; [e setBytes:&cp length:sizeof(cp) atIndex:3];
                [e dispatchThreads:MTLSizeMake(qkv_lin_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [e setComputePipelineState:pso_conv1d_spec];
                [e setBuffer:b_qkv offset:0 atIndex:0]; [e setBuffer:conv_states[l] offset:0 atIndex:1];
                [e setBuffer:L.conv_w.buf offset:L.conv_w.off atIndex:2]; [e setBuffer:conv_hist[l] offset:0 atIndex:3];
                [e setBytes:&cp length:sizeof(cp) atIndex:4];
                [e dispatchThreads:MTLSizeMake(qkv_lin_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            bar(e);

            DeltaNetParams dp{M, lin_kh, lin_vh, lin_khd, lin_vhd, rms_eps};
            if (M == 1) {
                [e setComputePipelineState:pso_dn_m1];
                [e setBuffer:b_qkv offset:0 atIndex:0]; [e setBuffer:b_z offset:0 atIndex:1];
                [e setBuffer:b_a offset:0 atIndex:2];   [e setBuffer:b_b offset:0 atIndex:3];
                [e setBuffer:L.A_log.buf offset:L.A_log.off atIndex:4]; [e setBuffer:L.dt_bias.buf offset:L.dt_bias.off atIndex:5];
                [e setBuffer:L.dn_norm.buf offset:L.dn_norm.off atIndex:6]; [e setBuffer:dn_states[l] offset:0 atIndex:7];
                [e setBuffer:b_dn_out offset:0 atIndex:8]; [e setBytes:&dp length:sizeof(dp) atIndex:9];
                [e dispatchThreadgroups:MTLSizeMake(lin_vh, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [e setComputePipelineState:pso_dn_spec];
                [e setBuffer:b_qkv offset:0 atIndex:0]; [e setBuffer:b_z offset:0 atIndex:1];
                [e setBuffer:b_a offset:0 atIndex:2];   [e setBuffer:b_b offset:0 atIndex:3];
                [e setBuffer:L.A_log.buf offset:L.A_log.off atIndex:4]; [e setBuffer:L.dt_bias.buf offset:L.dt_bias.off atIndex:5];
                [e setBuffer:L.dn_norm.buf offset:L.dn_norm.off atIndex:6]; [e setBuffer:dn_states[l] offset:0 atIndex:7];
                [e setBuffer:b_dn_out offset:0 atIndex:8];
                [e setBuffer:dn_hist_k[l] offset:0 atIndex:9]; [e setBuffer:dn_hist_d[l] offset:0 atIndex:10];
                [e setBuffer:dn_hist_dec[l] offset:0 atIndex:11];
                [e setBytes:&dp length:sizeof(dp) atIndex:12];
                [e dispatchThreadgroups:MTLSizeMake(lin_vh, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            bar(e);

            enc_qmv(e, M, b_dn_out, L.out_p, b_x, true);
            bar(e);
        } else {
            if (L.q.bits == 4 && L.k.bits == 4 && L.v.bits == 4 && L.q.gs_shift == L.k.gs_shift) {
                FullQKVParams qp{M, dim, L.q.gs_shift, L.q.N, L.k.N, L.v.N};
                int tot_n = L.q.N + L.k.N + L.v.N;
                id<MTLComputePipelineState> pso_fq = (M == 1) ? pso_full_qkv4_m1 : ((M >= 4 && (dim % 64 == 0)) ? pso_full_qkv4_mma8 : pso_full_qkv4);
                int rpt_fq = (M == 1) ? 8 : ((M >= 4 && (dim % 64 == 0)) ? 32 : 4);
                [e setComputePipelineState:pso_fq];
                [e setBuffer:b_xn offset:0 atIndex:0];
                [e setBuffer:L.q.w_buf offset:L.q.w atIndex:1]; [e setBuffer:L.q.s_buf offset:L.q.s atIndex:2]; [e setBuffer:L.q.b_buf offset:L.q.b atIndex:3];
                [e setBuffer:L.k.w_buf offset:L.k.w atIndex:4]; [e setBuffer:L.k.s_buf offset:L.k.s atIndex:5]; [e setBuffer:L.k.b_buf offset:L.k.b atIndex:6];
                [e setBuffer:L.v.w_buf offset:L.v.w atIndex:7]; [e setBuffer:L.v.s_buf offset:L.v.s atIndex:8]; [e setBuffer:L.v.b_buf offset:L.v.b atIndex:9];
                [e setBuffer:b_q_raw offset:0 atIndex:10]; [e setBuffer:b_k offset:0 atIndex:11]; [e setBuffer:b_v offset:0 atIndex:12];
                [e setBytes:&qp length:sizeof(qp) atIndex:13];
                [e dispatchThreadgroups:MTLSizeMake((tot_n + rpt_fq - 1) / rpt_fq, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                enc_qmv(e, M, b_xn, L.q, b_q_raw, false);
                enc_qmv(e, M, b_xn, L.k, b_k,     false);
                enc_qmv(e, M, b_xn, L.v, b_v,     false);
            }
            bar(e);

            QKNormRopeParams qp{M, n_heads, n_kv_heads, head_dim, rope_dim, pos, max_seq, rms_eps};
            [e setComputePipelineState:pso_qk_rope];
            [e setBuffer:b_q_raw offset:0 atIndex:0]; [e setBuffer:b_k offset:0 atIndex:1]; [e setBuffer:b_v offset:0 atIndex:2];
            [e setBuffer:b_q offset:0 atIndex:3]; [e setBuffer:b_gate offset:0 atIndex:4];
            [e setBuffer:k_caches[l] offset:0 atIndex:5]; [e setBuffer:v_caches[l] offset:0 atIndex:6];
            [e setBuffer:cos_s.buf offset:cos_s.off atIndex:7]; [e setBuffer:sin_s.buf offset:sin_s.off atIndex:8];
            [e setBuffer:L.q_norm.buf offset:L.q_norm.off atIndex:9]; [e setBuffer:L.k_norm.buf offset:L.k_norm.off atIndex:10];
            [e setBytes:&qp length:sizeof(qp) atIndex:11];
            [e dispatchThreadgroups:MTLSizeMake(n_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            bar(e);

            GQA256Params gp{M, n_heads, n_kv_heads, pos, max_seq};
            [e setComputePipelineState:pso_gqa256];
            [e setBuffer:b_q offset:0 atIndex:0]; [e setBuffer:b_gate offset:0 atIndex:1];
            [e setBuffer:k_caches[l] offset:0 atIndex:2]; [e setBuffer:v_caches[l] offset:0 atIndex:3];
            [e setBuffer:b_att offset:0 atIndex:4]; [e setBytes:&gp length:sizeof(gp) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(n_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            bar(e);

            enc_qmv(e, M, b_att, L.o, b_x, true);
            bar(e);
        }
    }

    void enc_layer_ffn(id<MTLComputeCommandEncoder> e, int M, int l) {
        const auto& L = layers[l];
        NormParams np{dim, rms_eps};
        [e setComputePipelineState:pso_rms];
        [e setBuffer:b_x offset:0 atIndex:0]; [e setBuffer:L.post_norm.buf offset:L.post_norm.off atIndex:1];
        [e setBuffer:b_xn offset:0 atIndex:2]; [e setBytes:&np length:sizeof(np) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        bar(e);

        if (is_moe) {
            // Wave 1 (Concurrent): Router+ShGate + Shared-Expert SwiGLU
            if (L.router.bits == L.sh_gate.bits && (L.router.bits == 8 || L.router.bits == 4) && L.router.gs_shift == L.sh_gate.gs_shift) {
                QMVMultiParams qp{M, L.router.K, L.router.N, L.router.gs_shift};
                [e setComputePipelineState:(L.router.bits == 8 ? pso_router_sh8 : pso_router_sh4)];
                [e setBuffer:b_xn offset:0 atIndex:0];
                [e setBuffer:L.router.w_buf  offset:L.router.w  atIndex:1]; [e setBuffer:L.router.s_buf  offset:L.router.s  atIndex:2]; [e setBuffer:L.router.b_buf  offset:L.router.b  atIndex:3];
                [e setBuffer:L.sh_gate.w_buf offset:L.sh_gate.w atIndex:4]; [e setBuffer:L.sh_gate.s_buf offset:L.sh_gate.s atIndex:5]; [e setBuffer:L.sh_gate.b_buf offset:L.sh_gate.b atIndex:6];
                [e setBuffer:b_router_l offset:0 atIndex:7]; [e setBuffer:b_sh_gl offset:0 atIndex:8];
                [e setBytes:&qp length:sizeof(qp) atIndex:9];
                [e dispatchThreadgroups:MTLSizeMake((L.router.N + 3) / 4 + 1, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                enc_qmv(e, M, b_xn, L.router,  b_router_l, false);
                enc_qmv(e, M, b_xn, L.sh_gate, b_sh_gl,    false);
            }
            enc_swiglu(e, M, b_xn, L.sh_gp, L.sh_up, b_sh_h);
            bar(e);

            // Wave 2 (Concurrent): TopK+Softmax + Shared-Expert Down
            TopKParams tp{M, num_experts, top_k};
            [e setComputePipelineState:pso_topk];
            [e setBuffer:b_router_l offset:0 atIndex:0]; [e setBuffer:b_sh_gl offset:0 atIndex:1];
            [e setBuffer:b_top_ids offset:0 atIndex:2];  [e setBuffer:b_top_w offset:0 atIndex:3];
            [e setBuffer:b_sh_g offset:0 atIndex:4];     [e setBytes:&tp length:sizeof(tp) atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            enc_qmv(e, M, b_sh_h, L.sh_dp, b_sh_out, false);
            bar(e);

            // Unified High-Occupancy Low-Register 3D Contiguous-Expert MoE (M=1..8!)
            MoEProjParams swp{M, L.moe_gp.K, L.moe_gp.N, top_k, L.moe_gp.gs_shift, L.moe_gp.stride_w, L.moe_gp.stride_s};
            [e setComputePipelineState:pso_moe_sw1[L.moe_gp.bits]];
            [e setBuffer:b_xn offset:0 atIndex:0]; [e setBuffer:b_top_ids offset:0 atIndex:1];
            [e setBuffer:L.moe_gp.w_buf offset:L.moe_gp.w atIndex:2]; [e setBuffer:L.moe_gp.s_buf offset:L.moe_gp.s atIndex:3]; [e setBuffer:L.moe_gp.b_buf offset:L.moe_gp.b atIndex:4];
            [e setBuffer:L.moe_up.w_buf offset:L.moe_up.w atIndex:5]; [e setBuffer:L.moe_up.s_buf offset:L.moe_up.s atIndex:6]; [e setBuffer:L.moe_up.b_buf offset:L.moe_up.b atIndex:7];
            [e setBuffer:b_moe_h offset:0 atIndex:8]; [e setBytes:&swp length:sizeof(swp) atIndex:9];
            [e dispatchThreadgroups:MTLSizeMake((L.moe_gp.N + 7) / 8, top_k, M) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            bar(e);

            MoEProjParams dnp{M, L.moe_dp.K, L.moe_dp.N, top_k, L.moe_dp.gs_shift, L.moe_dp.stride_w, L.moe_dp.stride_s};
            [e setComputePipelineState:pso_moe_dn1[L.moe_dp.bits]];
            [e setBuffer:b_moe_h offset:0 atIndex:0]; [e setBuffer:b_top_ids offset:0 atIndex:1];
            [e setBuffer:L.moe_dp.w_buf offset:L.moe_dp.w atIndex:2]; [e setBuffer:L.moe_dp.s_buf offset:L.moe_dp.s atIndex:3]; [e setBuffer:L.moe_dp.b_buf offset:L.moe_dp.b atIndex:4];
            [e setBuffer:b_moe_out offset:0 atIndex:5]; [e setBytes:&dnp length:sizeof(dnp) atIndex:6];
            [e dispatchThreadgroups:MTLSizeMake((L.moe_dp.N + 15) / 16, top_k, M) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            bar(e);

            TopKParams cp{M, dim >> 2, top_k};
            [e setComputePipelineState:pso_combine];
            [e setBuffer:b_moe_out offset:0 atIndex:0]; [e setBuffer:b_top_w offset:0 atIndex:1];
            [e setBuffer:b_sh_out offset:0 atIndex:2];  [e setBuffer:b_sh_g offset:0 atIndex:3];
            [e setBuffer:b_x offset:0 atIndex:4];       [e setBytes:&cp length:sizeof(cp) atIndex:5];
            [e dispatchThreads:MTLSizeMake(dim >> 2, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            bar(e);
        } else {
            enc_swiglu(e, M, b_xn, L.gate, L.up, b_dense_h);
            bar(e);
            enc_qmv(e, M, b_dense_h, L.down, b_x, true);
            bar(e);
        }
    }

    void run_step_m(const int32_t* ids, int M, int pos, int32_t* out_ids,
                    bool run_head, const SparseRepParams& rp, double* gpu_ms_out = nullptr,
                    bool want_top64 = false, int32_t* top64_ids_out = nullptr, float* top64_vals_out = nullptr) {
        memcpy([b_tok contents], ids, size_t(M) * sizeof(int32_t));
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

        if (pending_dn_commit > 0) {
            enc_commit_deltanet(e, pending_dn_commit);
            pending_dn_commit = 0;
        }

        QMVParams ep{dim, vocab, embed.gs_shift};
        int ebits = embed.bits;
        [e setComputePipelineState:pso_embed];
        [e setBuffer:b_tok offset:0 atIndex:0];
        [e setBuffer:embed.w_buf offset:embed.w atIndex:1]; [e setBuffer:embed.s_buf offset:embed.s atIndex:2]; [e setBuffer:embed.b_buf offset:embed.b atIndex:3];
        [e setBuffer:b_x offset:0 atIndex:4]; [e setBytes:&ep length:sizeof(ep) atIndex:5]; [e setBytes:&ebits length:sizeof(ebits) atIndex:6];
        [e dispatchThreads:MTLSizeMake(dim, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        bar(e);

        for (int l = 0; l < n_layers; ++l) {
            enc_layer_attn(e, M, l, pos);
            enc_layer_ffn(e, M, l);
        }

        if (run_head) {
            NormParams np{dim, rms_eps};
            [e setComputePipelineState:pso_rms];
            [e setBuffer:b_x offset:0 atIndex:0]; [e setBuffer:fnorm_s.buf offset:fnorm_s.off atIndex:1];
            [e setBuffer:b_xn offset:0 atIndex:2]; [e setBytes:&np length:sizeof(np) atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            bar(e);

            enc_qmv(e, M, b_xn, lm_head, b_logits, false);
            bar(e);
            if (rp.penalty > 1.001f && rp.n_slots > 0) {
                [e setComputePipelineState:pso_rep];
                [e setBuffer:b_logits offset:0 atIndex:0]; [e setBytes:&rp length:sizeof(rp) atIndex:1];
                [e dispatchThreads:MTLSizeMake(rp.n_slots, M, 1) threadsPerThreadgroup:MTLSizeMake(std::min(rp.n_slots, 64), 1, 1)];
                bar(e);
            }
            if (want_top64) {
                [e setComputePipelineState:pso_topk64];
                [e setBuffer:b_logits offset:0 atIndex:0];
                [e setBuffer:b_top64_ids offset:0 atIndex:1];
                [e setBuffer:b_top64_vals offset:0 atIndex:2];
                [e setBytes:&vocab length:sizeof(vocab) atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [e setComputePipelineState:pso_argmax];
                [e setBuffer:b_logits offset:0 atIndex:0]; [e setBuffer:b_out_tok offset:0 atIndex:1];
                [e setBytes:&vocab length:sizeof(vocab) atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (gpu_ms_out) *gpu_ms_out = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        if (run_head) {
            if (want_top64) {
                if (top64_ids_out)  memcpy(top64_ids_out,  [b_top64_ids contents],  size_t(M) * 64 * sizeof(int32_t));
                if (top64_vals_out) memcpy(top64_vals_out, [b_top64_vals contents], size_t(M) * 64 * sizeof(float));
                if (out_ids) {
                    const int32_t* p64 = (const int32_t*)[b_top64_ids contents];
                    for (int m = 0; m < M; ++m) out_ids[m] = p64[m * 64];
                }
            } else if (out_ids) {
                memcpy(out_ids, [b_out_tok contents], size_t(M) * sizeof(int32_t));
            }
        }
    }

    // 2-Level Prefix State Reuse:
    //   Level 1: Live multi-turn `cached_ids` (0ms restore!)
    //   Level 2: Baked in-RAM / disk `ram_snap` (0.4ms restore for stateless multi-user API sharing a system prompt!)
    //   Auto-captures `ram_snap` at `prefix_len` whenever a new shared system prompt is seen!
    void prefill_range(const std::vector<int32_t>& pids, size_t start, size_t end, int& pos,
                       bool run_last_head, bool want_top64, int32_t* last_out_id,
                       int32_t* last_top64_ids, float* last_top64_vals) {
        SparseRepParams no_rp{MAX_M, vocab, 0, 1.0f, {}};
        int32_t out_ids[MAX_M] = {0};
        int32_t t64_i[MAX_M * 64];
        float   t64_v[MAX_M * 64];
        for (size_t i = start; i < end; i += MAX_M) {
            int chunk = int(std::min<size_t>(MAX_M, end - i));
            bool is_last = run_last_head && (i + size_t(chunk) == end);
            no_rp.M = chunk;
            run_step_m(pids.data() + i, chunk, pos, out_ids, is_last, no_rp, nullptr,
                       is_last && want_top64, t64_i, t64_v);
            if (chunk > 1) pending_dn_commit = chunk;
            for (int c = 0; c < chunk; ++c) cached_ids.push_back(pids[i + size_t(c)]);
            pos += chunk;
            if (is_last) {
                int last_m = chunk - 1;
                if (last_out_id) *last_out_id = out_ids[last_m];
                if (want_top64 && last_top64_ids && last_top64_vals) {
                    memcpy(last_top64_ids,  t64_i + last_m * 64, 64 * sizeof(int32_t));
                    memcpy(last_top64_vals, t64_v + last_m * 64, 64 * sizeof(float));
                }
            }
        }
    }

    int32_t prefill_incremental(const std::vector<int32_t>& pids, int prefix_len, int& pos_out, int& reused_out,
                                bool want_top64 = false, int32_t* last_top64_ids = nullptr, float* last_top64_vals = nullptr) {
        // Check Level 1: exact match against live cached_ids
        size_t lcp_live = 0;
        while (lcp_live < cached_ids.size() && lcp_live < pids.size() && cached_ids[lcp_live] == pids[lcp_live]) lcp_live++;
        bool hit_live = (lcp_live == cached_ids.size() && lcp_live > 0 && lcp_live < pids.size());

        // Check Level 2: exact match against baked ram_snap
        size_t lcp_snap = 0;
        if (ram_snap.valid()) {
            while (lcp_snap < ram_snap.ids.size() && lcp_snap < pids.size() && ram_snap.ids[lcp_snap] == pids[lcp_snap]) lcp_snap++;
        }
        bool hit_snap = (ram_snap.valid() && lcp_snap == ram_snap.ids.size() && lcp_snap > 0 && lcp_snap < pids.size());

        size_t start = 0;
        if (hit_live && (!hit_snap || lcp_live >= lcp_snap)) {
            start = lcp_live;
            reused_out = int(lcp_live);
        } else if (hit_snap) {
            restore_snapshot(ram_snap);
            start = lcp_snap;
            reused_out = int(lcp_snap);
        } else {
            reset_states();
            start = 0;
            reused_out = 0;
        }

        int pos = int(start);
        int32_t out_tok = 0;

        // If caller marked a reusable system prefix (`prefix_len`) that isn't snapshotted yet, prefill to `prefix_len` and snapshot!
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
        SparseRepParams no_rp{8, vocab, 0, 1.0f, {}};
        for (int w = 0; w < 2; ++w) { run_step_m(d_in, 4, 32, d_out, true, no_rp); pending_dn_commit = 0; }
        for (int m : {1, 2, 4, 6, 8}) {
            std::vector<double> ts;
            for (int it = 0; it < 5; ++it) {
                auto a = std::chrono::high_resolution_clock::now();
                run_step_m(d_in, m, 32, d_out, true, no_rp);
                pending_dn_commit = 0;
                ts.push_back(std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - a).count());
            }
            std::sort(ts.begin(), ts.end());
            cost[m] = ts[ts.size() / 2];
        }
        for (int m = 3; m <= 7; m += 2) cost[m] = 0.5 * (cost[m-1] + cost[m+1]);
        reset_states();
    }

    void profile_xray(const double cost[9]) {
        SparseRepParams no_rp{1, vocab, 0, 1.0f, {}};
        int32_t d_in[8] = {16, 11, 220, 17, 11, 220, 18, 11}, o_out[8];
        for (int w = 0; w < 2; ++w) { run_step_m(d_in, 8, 32, o_out, true, no_rp); pending_dn_commit = 0; }

        auto tm = [&](auto fn) {
            for (int w = 0; w < 2; ++w) { auto cb = [queue commandBuffer]; auto e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; fn(e); [e endEncoding]; [cb commit]; [cb waitUntilCompleted]; }
            std::vector<double> ts;
            for (int it = 0; it < 9; ++it) {
                auto cb = [queue commandBuffer]; auto e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                fn(e); [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                ts.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
            }
            std::sort(ts.begin(), ts.end());
            return ts[ts.size() / 2];
        };

        double t_dn = tm([&](id<MTLComputeCommandEncoder> e){ for (int l=0;l<n_layers;++l) if (layers[l].type==0) enc_layer_attn(e, 1, l, 128); });
        double t_fa = tm([&](id<MTLComputeCommandEncoder> e){ for (int l=0;l<n_layers;++l) if (layers[l].type==1) enc_layer_attn(e, 1, l, 128); });
        double t_ff = tm([&](id<MTLComputeCommandEncoder> e){ for (int l=0;l<n_layers;++l) enc_layer_ffn(e, 1, l); });
        double t_hd = tm([&](id<MTLComputeCommandEncoder> e){ enc_qmv(e, 1, b_xn, lm_head, b_logits, false); });

        double t_dn8 = tm([&](id<MTLComputeCommandEncoder> e){ for (int l=0;l<n_layers;++l) if (layers[l].type==0) enc_layer_attn(e, 8, l, 128); });
        double t_fa8 = tm([&](id<MTLComputeCommandEncoder> e){ for (int l=0;l<n_layers;++l) if (layers[l].type==1) enc_layer_attn(e, 8, l, 128); });
        double t_ff8 = tm([&](id<MTLComputeCommandEncoder> e){ for (int l=0;l<n_layers;++l) enc_layer_ffn(e, 8, l); });
        double t_hd8 = tm([&](id<MTLComputeCommandEncoder> e){ enc_qmv(e, 8, b_xn, lm_head, b_logits, false); });

        printf("\n▸ bihaba-serve GPU X-Ray Profile (ALL %dL @ M=1 & M=8 3D-MoE+MMA, pos=128, topK=%d):\n", n_layers, top_k);
        printf("   M=1 (Burst)  : DeltaNet=%.2fms | FullAttn-256=%.2fms | %s=%.2fms | Head(%dK)=%.2fms  (sum=%.2fms)\n",
               t_dn, t_fa, is_moe ? "MoE(256-Exp)" : "Dense-FFN", t_ff, vocab / 1000, t_hd, t_dn + t_fa + t_ff + t_hd);
        printf("   M=8 (3D-MMA) : DeltaNet=%.2fms | FullAttn-256=%.2fms | %s=%.2fms | Head(%dK)=%.2fms  (sum=%.2fms)\n",
               t_dn8, t_fa8, is_moe ? "MoE(256-Exp)" : "Dense-FFN", t_ff8, vocab / 1000, t_hd8, t_dn8 + t_fa8 + t_ff8 + t_hd8);
        printf("▸ Zero-Copy Speculative Chunked Step Latency (M=1..8):\n   ");
        for (int m : {1, 2, 4, 6, 8}) {
            printf("n%d=%.2fms(%.2f×, ceil %.0f tok/s)  ", m, cost[m], cost[m] / cost[1], m * 1000.0 / cost[m]);
        }
        printf("\n\n");
        reset_states();
    }
};

struct SpecStats {
    std::vector<int32_t> tokens;
    double prefill_ms = 0.0, decode_ms = 0.0;
    int reused = 0, steps = 0, proposed = 0, accepted = 0;
};

static int choose_best_k(double alpha, const double cost[9], int max_k) {
    int best_k = 0;
    double best_tps = 1.0 / cost[1];
    for (int k = 1; k <= max_k; ++k) {
        double et = (alpha >= 0.999) ? (k + 1.0) : (1.0 - std::pow(alpha, k + 1)) / (1.0 - alpha);
        double tps = et / cost[k + 1];
        if (tps > best_tps) { best_tps = tps; best_k = k; }
    }
    return best_k;
}

// Compute filtered target distribution over sorted Top-64 logits (temperature, top_k, min_p, top_p)
static void build_target_probs(const float vals[64], const SamplingConfig& sc, float probs[64]) {
    if (sc.temperature <= 1e-4f) {
        probs[0] = 1.0f;
        for (int i = 1; i < 64; ++i) probs[i] = 0.0f;
        return;
    }
    const float inv_t = 1.0f / sc.temperature;
    const float mx = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < 64; ++i) {
        float p = (vals[i] > -1e20f) ? std::exp((vals[i] - mx) * inv_t) : 0.0f;
        probs[i] = p;
        sum += p;
    }
    const float inv_s = 1.0f / std::max(sum, 1e-12f);
    for (int i = 0; i < 64; ++i) probs[i] *= inv_s;

    int k_lim = (sc.top_k > 0 && sc.top_k < 64) ? sc.top_k : 64;
    for (int i = k_lim; i < 64; ++i) probs[i] = 0.0f;

    if (sc.min_p > 0.0f) {
        float thresh = probs[0] * sc.min_p;
        for (int i = 1; i < k_lim; ++i) if (probs[i] < thresh) probs[i] = 0.0f;
    }
    if (sc.top_p > 0.0f && sc.top_p < 1.0f) {
        float csum = 0.0f;
        bool cut = false;
        for (int i = 0; i < k_lim; ++i) {
            if (cut) { probs[i] = 0.0f; continue; }
            csum += probs[i];
            if (csum >= sc.top_p) cut = true;
        }
    }
    float fsum = 0.0f;
    for (int i = 0; i < 64; ++i) fsum += probs[i];
    if (fsum <= 1e-12f) {
        probs[0] = 1.0f;
        for (int i = 1; i < 64; ++i) probs[i] = 0.0f;
    } else {
        float inv_f = 1.0f / fsum;
        for (int i = 0; i < 64; ++i) probs[i] *= inv_f;
    }
}

static int32_t sample_categorical(const int32_t ids[64], const float probs[64], std::mt19937& rng, int exclude_idx = -1) {
    float total = 0.0f;
    for (int i = 0; i < 64; ++i) if (i != exclude_idx) total += probs[i];
    if (total <= 1e-12f) return ids[0];
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng), acc = 0.0f;
    for (int i = 0; i < 64; ++i) {
        if (i == exclude_idx) continue;
        acc += probs[i];
        if (r <= acc) return ids[i];
    }
    return ids[0];
}

static SpecStats generate_adaptive(
    Qwen35Engine& eng, const std::vector<int32_t>& pids, int prefix_len, int max_tokens,
    int max_spec, const SamplingConfig& sc, const double cost[9], bool stream_stdout, bool ignore_eos = false)
{
    SpecStats st;
    const bool do_sample = (sc.temperature > 1e-4f);
    std::mt19937 rng(sc.seed ? sc.seed : uint32_t(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    int pos = 0;
    int32_t t64_ids[8 * 64];
    float   t64_vals[8 * 64];

    auto tp0 = std::chrono::high_resolution_clock::now();
    int32_t nxt = eng.prefill_incremental(pids, prefix_len, pos, st.reused, do_sample, t64_ids, t64_vals);
    st.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tp0).count();

    if (do_sample) {
        float probs[64];
        build_target_probs(t64_vals, sc, probs);
        nxt = sample_categorical(t64_ids, probs, rng);
    }

    FastSuffixSpeculator sfx;
    for (int32_t t : pids) sfx.push(t);
    sfx.push(nxt);

    st.tokens.push_back(nxt);
    if (stream_stdout) { printf("{\"tok\":%d}\n", nxt); fflush(stdout); }

    double alpha = 0.75;
    int32_t batch[8], out_ids[8];
    auto td0 = std::chrono::high_resolution_clock::now();

    while ((ignore_eos || !eng.is_eos(nxt)) && int(st.tokens.size()) < max_tokens && pos < eng.max_seq - 9) {
        int rem = max_tokens - int(st.tokens.size());
        int kk = (max_spec > 0) ? std::min(choose_best_k(alpha, cost, max_spec), rem) : 0;
        auto draft = sfx.propose(kk);
        int M = 1 + int(draft.size());
        batch[0] = nxt;
        for (int i = 0; i < int(draft.size()); ++i) batch[1 + i] = draft[i];
        st.proposed += (M - 1);

        SparseRepParams rp = build_rep(st.tokens, M, eng.vocab, sc.rep_pen);
        eng.run_step_m(batch, M, pos, out_ids, true, rp, nullptr, do_sample, t64_ids, t64_vals);
        st.steps++;

        std::vector<int32_t> emitted;
        int acc = 0;

        if (!do_sample) {
            while (acc < M - 1 && batch[1 + acc] == out_ids[acc]) {
                emitted.push_back(out_ids[acc]);
                acc++;
            }
            emitted.push_back(out_ids[acc]);
        } else {
            // 100% Lossless Speculative Rejection Sampling for Deterministic Draft (q(d) = 1)!
            bool rejected = false;
            for (int i = 0; i < M - 1; ++i) {
                float probs[64];
                build_target_probs(t64_vals + i * 64, sc, probs);
                int32_t d_tok = batch[1 + i];
                int d_idx = -1;
                for (int j = 0; j < 64; ++j) if (t64_ids[i * 64 + j] == d_tok) { d_idx = j; break; }
                float p_d = (d_idx >= 0) ? probs[d_idx] : 0.0f;
                if (u01(rng) < p_d) {
                    emitted.push_back(d_tok);
                    acc++;
                } else {
                    int32_t corr = sample_categorical(t64_ids + i * 64, probs, rng, d_idx);
                    emitted.push_back(corr);
                    rejected = true;
                    break;
                }
            }
            if (!rejected) {
                // Bonus token from position M-1!
                float probs[64];
                build_target_probs(t64_vals + (M - 1) * 64, sc, probs);
                emitted.push_back(sample_categorical(t64_ids + (M - 1) * 64, probs, rng));
            }
        }

        st.accepted += acc;
        int n_emit = int(emitted.size());

        if (M > 1) {
            eng.pending_dn_commit = n_emit;
            alpha = std::min(0.98, std::max(0.05, 0.70 * alpha + 0.30 * (double(acc) / (M - 1))));
        } else {
            alpha += 0.05 * (0.75 - alpha);
        }

        for (int i = 0; i < n_emit; ++i) {
            int32_t tok = emitted[i];
            eng.cached_ids.push_back(batch[i]);
            st.tokens.push_back(tok);
            sfx.push(tok);
            nxt = tok;
            if (stream_stdout) { printf("{\"tok\":%d}\n", tok); fflush(stdout); }
            if ((!ignore_eos && eng.is_eos(tok)) || int(st.tokens.size()) >= max_tokens) break;
        }
        pos += n_emit;
    }

    st.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - td0).count();
    return st;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc < 2) {
            printf("Usage (Direct Zero-Copy ~/models/<dir> OR .taalas2 binary):\n");
            printf("  Profile: %s <model_dir> --profile [--bench 128] [--top-k 4] [--spec 7] [--max-seq 4096]\n", argv[0]);
            printf("  Daemon:  %s <model_dir> --daemon [--state <baked.tstate>] [--top-k 4] [--spec 7] [--rep-penalty 1.1]\n", argv[0]);
            return 1;
        }
        std::string model_path = argv[1], state_path = "";
        bool do_profile = false, daemon_mode = false, willneed = false;
        int bench_toks = 0, override_topk = 0, max_spec = 7, req_max_seq = 4096;
        SamplingConfig def_sc;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--profile") == 0) do_profile = true;
            else if (strcmp(argv[i], "--daemon") == 0) daemon_mode = true;
            else if (strcmp(argv[i], "--willneed") == 0) willneed = true;
            else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) bench_toks = atoi(argv[++i]);
            else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) override_topk = atoi(argv[++i]);
            else if (strcmp(argv[i], "--spec") == 0 && i + 1 < argc) max_spec = std::min(7, std::max(0, atoi(argv[++i])));
            else if (strcmp(argv[i], "--max-seq") == 0 && i + 1 < argc) req_max_seq = atoi(argv[++i]);
            else if (strcmp(argv[i], "--state") == 0 && i + 1 < argc) state_path = argv[++i];
            else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) def_sc.rep_pen = atof(argv[++i]);
            else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) def_sc.temperature = atof(argv[++i]);
        }

        Qwen35Engine eng;
        if (!eng.load(model_path, override_topk, willneed, req_max_seq)) return 1;

        if (!state_path.empty()) {
            auto ts0 = std::chrono::high_resolution_clock::now();
            if (eng.load_snapshot_file(eng.ram_snap, state_path)) {
                double sms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - ts0).count();
                fprintf(stderr, "▸ Loaded Baked State Snapshot: %s (%d prefix toks, %.1f MB in %.1f ms)\n",
                        state_path.c_str(), eng.ram_snap.pos, eng.ram_snap.bytes() / 1e6, sms);
            } else {
                fprintf(stderr, "WARNING: Failed to load state file %s\n", state_path.c_str());
            }
        }

        double cost[9] = {0};
        eng.measure_step_costs(cost);

        if (do_profile && !daemon_mode) {
            eng.profile_xray(cost);
        }

        if (bench_toks > 0 && !daemon_mode) {
            eng.reset_states();
            SparseRepParams rp1{1, eng.vocab, 0, 1.0f, {}};
            int32_t tok = 9707;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < bench_toks; ++i) eng.run_step_m(&tok, 1, i, &tok, true, rp1);
            double ms1 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            printf("▸ Live Decode Bench (Greedy M=1):         %d toks in %.1f ms => %.1f tok/s\n", bench_toks, ms1, bench_toks * 1000.0 / ms1);

            // Speculative Greedy vs Sampling + Auto System-Prompt Snapshot TTFT Benchmark!
            std::vector<int32_t> pattern = {16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220, 22, 11, 220, 23, 198};
            std::vector<int32_t> prompt;
            for (int r = 0; r < 16; ++r) prompt.insert(prompt.end(), pattern.begin(), pattern.end()); // 368-token prompt
            int sys_len = 320;

            SamplingConfig sc_greedy{};
            eng.reset_states(); eng.ram_snap = StateSnapshot{};
            SpecStats g0 = generate_adaptive(eng, prompt, 0, bench_toks, 0, sc_greedy, cost, false, true);

            eng.reset_states(); eng.ram_snap = StateSnapshot{};
            SpecStats gs = generate_adaptive(eng, prompt, sys_len, bench_toks, max_spec, sc_greedy, cost, false, true);
            int match = 0, lim = int(std::min(g0.tokens.size(), gs.tokens.size()));
            while (match < lim && g0.tokens[match] == gs.tokens[match]) match++;
            int n_dec = std::max(0, int(gs.tokens.size()) - 1);
            printf("▸ Live Speculative Bench (Greedy k=%d):    %d toks in %.1f ms => %.1f tok/s (%.2f tok/step, match=%d/%d)\n",
                   max_spec, n_dec, gs.decode_ms, n_dec * 1000.0 / std::max(gs.decode_ms, 0.01),
                   double(n_dec) / std::max(gs.steps, 1), match, lim);

            // Simulate a NEW stateless user arriving with the SAME 320-token system prompt + different user suffix!
            std::vector<int32_t> prompt_user2(prompt.begin(), prompt.begin() + sys_len);
            for (int i = 0; i < 16; ++i) prompt_user2.push_back(9707 + i);
            SamplingConfig sc_samp{}; sc_samp.temperature = 0.7f; sc_samp.top_p = 0.9f; sc_samp.top_k = 40; sc_samp.min_p = 0.05f; sc_samp.seed = 42;
            SpecStats gu2 = generate_adaptive(eng, prompt_user2, sys_len, bench_toks, max_spec, sc_samp, cost, false, true);
            int n_u2 = std::max(0, int(gu2.tokens.size()) - 1);
            printf("▸ Multi-User Baked-Prefix + Sampling:     %d toks in %.1f ms => %.1f tok/s (prefill=%.1f ms vs cold %.1f ms, reused=%d/%zu toks, snap=%.1f MB)\n",
                   n_u2, gu2.decode_ms, n_u2 * 1000.0 / std::max(gu2.decode_ms, 0.01),
                   gu2.prefill_ms, gs.prefill_ms, gu2.reused, prompt_user2.size(), eng.ram_snap.bytes() / 1e6);
            return 0;
        }

        if (daemon_mode) {
            printf("{\"ready\":true,\"version\":\"v14\",\"arch\":\"%s\",\"model_dir\":\"%s\",\"top_k\":%d,\"max_spec\":%d,\"baked_toks\":%d}\n",
                   eng.arch_name.c_str(), eng.model_dir.c_str(), eng.top_k, max_spec, eng.ram_snap.pos);
            fflush(stdout);
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                NSData* reqData = [NSData dataWithBytes:line.data() length:line.size()];
                NSDictionary* req = [NSJSONSerialization JSONObjectWithData:reqData options:0 error:nil];
                if (!req) continue;

                NSString* cmd = req[@"cmd"];
                if (cmd && [cmd isEqualToString:@"bake"]) {
                    std::vector<int32_t> pids;
                    for (NSNumber* num in req[@"prompt_ids"]) pids.push_back([num intValue]);
                    std::string out_p = req[@"path"] ? [req[@"path"] UTF8String] : "";
                    auto tb0 = std::chrono::high_resolution_clock::now();
                    eng.reset_states();
                    int pos = 0;
                    eng.prefill_range(pids, 0, pids.size(), pos, false, false, nullptr, nullptr, nullptr);
                    eng.capture_snapshot(eng.ram_snap, pids, pos);
                    bool saved = out_p.empty() ? true : eng.save_snapshot_file(eng.ram_snap, out_p);
                    double bms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tb0).count();
                    printf("{\"baked\":%s,\"tokens\":%d,\"mb\":%.1f,\"ms\":%.1f}\n",
                           saved ? "true" : "false", pos, eng.ram_snap.bytes() / 1e6, bms);
                    fflush(stdout);
                    continue;
                }
                if (cmd && [cmd isEqualToString:@"load_state"]) {
                    std::string in_p = req[@"path"] ? [req[@"path"] UTF8String] : "";
                    auto tb0 = std::chrono::high_resolution_clock::now();
                    bool ok = eng.load_snapshot_file(eng.ram_snap, in_p);
                    double bms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - tb0).count();
                    printf("{\"loaded\":%s,\"tokens\":%d,\"mb\":%.1f,\"ms\":%.1f}\n",
                           ok ? "true" : "false", eng.ram_snap.pos, eng.ram_snap.bytes() / 1e6, bms);
                    fflush(stdout);
                    continue;
                }

                std::vector<int32_t> pids;
                for (NSNumber* num in req[@"prompt_ids"]) pids.push_back([num intValue]);
                int mt         = req[@"max_tokens"]  ? [req[@"max_tokens"] intValue]  : 512;
                int prefix_len = req[@"prefix_len"]  ? [req[@"prefix_len"] intValue]  : 0;
                int sp_k       = req[@"max_spec"]    ? std::min(7, std::max(0, [req[@"max_spec"] intValue])) : max_spec;

                SamplingConfig sc = def_sc;
                if (req[@"temperature"]) sc.temperature = [req[@"temperature"] floatValue];
                if (req[@"top_p"])       sc.top_p       = [req[@"top_p"] floatValue];
                if (req[@"top_k"])       sc.top_k       = [req[@"top_k"] intValue];
                if (req[@"min_p"])       sc.min_p       = [req[@"min_p"] floatValue];
                if (req[@"rep_penalty"]) sc.rep_pen     = [req[@"rep_penalty"] floatValue];
                if (req[@"seed"])        sc.seed        = uint32_t([req[@"seed"] unsignedIntValue]);

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
