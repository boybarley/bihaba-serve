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
#include <mach-o/dyld.h>
#include <libgen.h>

#include <algorithm>
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

struct NormParams      { int32_t dim; float eps; };
struct QMVParams       { int32_t K, N, gs_shift; };
struct QMVMultiParams  { int32_t M, K, N, gs_shift; };
struct TrRopeParams    { int32_t M, n_heads, n_kv_heads, head_dim, pos, max_seq, has_qkv_bias, has_qk_norm; float eps; };
struct TrGQAParams     { int32_t M, n_heads, n_kv_heads, pos, max_seq; };
struct SparseRepParams { int32_t M, K_len, n_slots; float penalty; int32_t slots[64]; };

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

struct TransformerEngine {
    static constexpr int MAX_M = 8;

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
    BufSlice cos_s, sin_s, fnorm_s, zero_dummy;
    std::vector<LayerMeta> layers;

    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> pso_embed=nil, pso_rms=nil, pso_rep=nil, pso_argmax=nil, pso_topk64=nil;
    id<MTLComputePipelineState> pso_qmv1[9], pso_qmv1_add[9];
    id<MTLComputePipelineState> pso_qmvm[9], pso_qmvm_add[9], pso_sw[9];
    id<MTLComputePipelineState> pso_qmm4_mma8=nil, pso_qmm4_mma8_add=nil, pso_qmm2_mma8=nil, pso_qmm2_mma8_add=nil;
    id<MTLComputePipelineState> pso_sw4_mma8=nil, pso_sw2_mma8=nil;
    id<MTLComputePipelineState> pso_qk_rope=nil, pso_gqa=nil;

    id<MTLBuffer> b_tok=nil, b_x=nil, b_xn=nil;
    id<MTLBuffer> b_q=nil, b_k=nil, b_v=nil, b_att=nil, b_dense_h=nil;
    id<MTLBuffer> b_logits=nil, b_out_tok=nil, b_top64_ids=nil, b_top64_vals=nil;

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
            int gs = K_expect / kg;
            int bits = (kw * 32) / K_expect;
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

        const bool bf = scales_bf16;
        pso_embed         = make_pso(dev, lib, "embed_lookup_q",          false, bf);
        pso_rms           = make_pso(dev, lib, "rms_norm",                false, bf);
        pso_rep           = make_pso(dev, lib, "apply_rep_penalty_sparse",false, bf);
        pso_argmax        = make_pso(dev, lib, "argmax_full",             false, bf);
        pso_topk64        = make_pso(dev, lib, "topk_logits_64",          false, bf);
        pso_qmm4_mma8     = make_pso(dev, lib, "qmm_4b_mma8",             false, bf);
        pso_qmm4_mma8_add = make_pso(dev, lib, "qmm_4b_mma8",             true,  bf);
        pso_qmm2_mma8     = make_pso(dev, lib, "qmm_2b_mma8",             false, bf);
        pso_qmm2_mma8_add = make_pso(dev, lib, "qmm_2b_mma8",             true,  bf);
        pso_sw4_mma8      = make_pso(dev, lib, "dense_swiglu_4b_mma8",    false, bf);
        pso_sw2_mma8      = make_pso(dev, lib, "dense_swiglu_2b_mma8",    false, bf);
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
        b_tok        = mkbuf(MAX_M * sizeof(int32_t));
        b_x          = mkbuf(MAX_M * dim * 2);
        b_xn         = mkbuf(MAX_M * dim * 2);
        b_q          = mkbuf(MAX_M * q_full_dim * 2);
        b_k          = mkbuf(MAX_M * kv_full_dim * 2);
        b_v          = mkbuf(MAX_M * kv_full_dim * 2);
        b_att        = mkbuf(MAX_M * q_full_dim * 2);
        b_dense_h    = mkbuf(MAX_M * dense_inter * 2);
        b_logits     = mkbuf(MAX_M * vocab * 2);
        b_out_tok    = mkbuf(MAX_M * sizeof(int32_t));
        b_top64_ids  = mkbuf(MAX_M * 64 * sizeof(int32_t));
        b_top64_vals = mkbuf(MAX_M * 64 * sizeof(float));

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
    inline void bar(id<MTLComputeCommandEncoder> e) { [e memoryBarrierWithScope:MTLBarrierScopeBuffers]; }

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

    bool save_snapshot_file(const StateSnapshot& snap, const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        const char magic[8] = {'T','S','T','A','T','E','T','1'};
        int32_t hdr[6] = {n_layers, dim, n_kv_heads, head_dim, snap.pos, int32_t(snap.ids.size())};
        fwrite(magic, 1, 8, f); fwrite(hdr, sizeof(int32_t), 6, f);
        fwrite(snap.ids.data(), sizeof(int32_t), snap.ids.size(), f);
        for (int l = 0; l < n_layers; ++l) {
            uint64_t sz[2] = {snap.k_data[l].size(), snap.v_data[l].size()};
            fwrite(sz, sizeof(uint64_t), 2, f);
            fwrite(snap.k_data[l].data(), 1, sz[0], f); fwrite(snap.v_data[l].data(), 1, sz[1], f);
        }
        fclose(f); return true;
    }

    bool load_snapshot_file(StateSnapshot& snap, const std::string& path) const {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        char magic[8] = {0}; int32_t hdr[6] = {0};
        if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "TSTATET1", 8) != 0) { fclose(f); return false; }
        if (fread(hdr, sizeof(int32_t), 6, f) != 6) { fclose(f); return false; }
        if (hdr[0] != n_layers || hdr[1] != dim || hdr[2] != n_kv_heads || hdr[3] != head_dim || hdr[4] >= max_seq) {
            fclose(f); return false;
        }
        snap.pos = hdr[4]; snap.ids.resize(size_t(hdr[5]));
        if (fread(snap.ids.data(), sizeof(int32_t), snap.ids.size(), f) != snap.ids.size()) { fclose(f); return false; }
        snap.k_data.assign(n_layers, {}); snap.v_data.assign(n_layers, {});
        for (int l = 0; l < n_layers; ++l) {
            uint64_t sz[2] = {0, 0};
            if (fread(sz, sizeof(uint64_t), 2, f) != 2) { fclose(f); return false; }
            snap.k_data[l].resize(sz[0]); snap.v_data[l].resize(sz[1]);
            if (fread(snap.k_data[l].data(), 1, sz[0], f) != sz[0]) { fclose(f); return false; }
            if (fread(snap.v_data[l].data(), 1, sz[1], f) != sz[1]) { fclose(f); return false; }
        }
        fclose(f); return true;
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

    void enc_layer(id<MTLComputeCommandEncoder> e, int M, int l, int pos) {
        const auto& L = layers[l];
        NormParams np{dim, rms_eps};
        [e setComputePipelineState:pso_rms];
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
        [e setComputePipelineState:pso_qk_rope];
        [e setBuffer:b_q offset:0 atIndex:0]; [e setBuffer:b_k offset:0 atIndex:1]; [e setBuffer:b_v offset:0 atIndex:2];
        [e setBuffer:k_caches[l] offset:0 atIndex:3]; [e setBuffer:v_caches[l] offset:0 atIndex:4];
        [e setBuffer:cos_s.buf offset:cos_s.off atIndex:5]; [e setBuffer:sin_s.buf offset:sin_s.off atIndex:6];
        [e setBuffer:L.q_norm.buf offset:L.q_norm.off atIndex:7]; [e setBuffer:L.k_norm.buf offset:L.k_norm.off atIndex:8];
        [e setBuffer:L.q_bias.buf offset:L.q_bias.off atIndex:9]; [e setBuffer:L.k_bias.buf offset:L.k_bias.off atIndex:10];
        [e setBuffer:L.v_bias.buf offset:L.v_bias.off atIndex:11];
        [e setBytes:&rp length:sizeof(rp) atIndex:12];
        [e dispatchThreadgroups:MTLSizeMake(n_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        bar(e);

        TrGQAParams gp{M, n_heads, n_kv_heads, pos, max_seq};
        [e setComputePipelineState:pso_gqa];
        [e setBuffer:b_q offset:0 atIndex:0]; [e setBuffer:k_caches[l] offset:0 atIndex:1];
        [e setBuffer:v_caches[l] offset:0 atIndex:2]; [e setBuffer:b_att offset:0 atIndex:3];
        [e setBytes:&gp length:sizeof(gp) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(n_heads, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        bar(e);

        enc_qmv(e, M, b_att, L.o, b_x, true);
        bar(e);

        [e setComputePipelineState:pso_rms];
        [e setBuffer:b_x offset:0 atIndex:0]; [e setBuffer:L.post_norm.buf offset:L.post_norm.off atIndex:1];
        [e setBuffer:b_xn offset:0 atIndex:2]; [e setBytes:&np length:sizeof(np) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        bar(e);

        enc_swiglu(e, M, b_xn, L.gate, L.up, b_dense_h);
        bar(e);
        enc_qmv(e, M, b_dense_h, L.down, b_x, true);
        bar(e);
    }

    void run_step_m(const int32_t* ids, int M, int pos, int32_t* out_ids,
                    bool run_head, const SparseRepParams& rp, double* gpu_ms_out = nullptr,
                    bool want_top64 = false, int32_t* top64_ids_out = nullptr, float* top64_vals_out = nullptr) {
        memcpy([b_tok contents], ids, size_t(M) * sizeof(int32_t));
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

        QMVParams ep{dim, vocab, embed.gs_shift};
        int ebits = embed.bits;
        [e setComputePipelineState:pso_embed];
        [e setBuffer:b_tok offset:0 atIndex:0];
        [e setBuffer:embed.w_buf offset:embed.w atIndex:1]; [e setBuffer:embed.s_buf offset:embed.s atIndex:2]; [e setBuffer:embed.b_buf offset:embed.b atIndex:3];
        [e setBuffer:b_x offset:0 atIndex:4]; [e setBytes:&ep length:sizeof(ep) atIndex:5]; [e setBytes:&ebits length:sizeof(ebits) atIndex:6];
        [e dispatchThreads:MTLSizeMake(dim, M, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        bar(e);

        for (int l = 0; l < n_layers; ++l) enc_layer(e, M, l, pos);

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
                [e setBuffer:b_logits offset:0 atIndex:0]; [e setBuffer:b_top64_ids offset:0 atIndex:1];
                [e setBuffer:b_top64_vals offset:0 atIndex:2]; [e setBytes:&vocab length:sizeof(vocab) atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [e setComputePipelineState:pso_argmax];
                [e setBuffer:b_logits offset:0 atIndex:0]; [e setBuffer:b_out_tok offset:0 atIndex:1];
                [e setBytes:&vocab length:sizeof(vocab) atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(M, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (gpu_ms_out) *gpu_ms_out = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        if (run_head) {
            if (want_top64) {
                if (top64_ids_out)  memcpy(top64_ids_out,  [b_top64_ids contents],  size_t(M) * 64 * sizeof(int32_t));
                if (top64_vals_out) memcpy(top64_vals_out, [b_top64_vals contents], size_t(M) * 64 * sizeof(float));
                if (out_ids) { const int32_t* p64 = (const int32_t*)[b_top64_ids contents]; for (int m = 0; m < M; ++m) out_ids[m] = p64[m * 64]; }
            } else if (out_ids) {
                memcpy(out_ids, [b_out_tok contents], size_t(M) * sizeof(int32_t));
            }
        }
    }

    void prefill_range(const std::vector<int32_t>& pids, size_t start, size_t end, int& pos,
                       bool run_last_head, bool want_top64, int32_t* last_out_id,
                       int32_t* last_top64_ids, float* last_top64_vals) {
        SparseRepParams no_rp{MAX_M, vocab, 0, 1.0f, {}};
        int32_t out_ids[MAX_M] = {0}, t64_i[MAX_M * 64]; float t64_v[MAX_M * 64];
        for (size_t i = start; i < end; i += MAX_M) {
            int chunk = int(std::min<size_t>(MAX_M, end - i));
            bool is_last = run_last_head && (i + size_t(chunk) == end);
            no_rp.M = chunk;
            run_step_m(pids.data() + i, chunk, pos, out_ids, is_last, no_rp, nullptr, is_last && want_top64, t64_i, t64_v);
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
        SparseRepParams no_rp{8, vocab, 0, 1.0f, {}};
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

    double alpha = 0.75; int32_t batch[8], out_ids[8];
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

        std::vector<int32_t> emitted; int acc = 0;
        if (!do_sample) {
            while (acc < M - 1 && batch[1 + acc] == out_ids[acc]) { emitted.push_back(out_ids[acc]); acc++; }
            emitted.push_back(out_ids[acc]);
        } else {
            bool rejected = false;
            for (int i = 0; i < M - 1; ++i) {
                float probs[64]; build_target_probs(t64_vals + i * 64, sc, probs);
                int32_t d_tok = batch[1 + i]; int d_idx = -1;
                for (int j = 0; j < 64; ++j) if (t64_ids[i * 64 + j] == d_tok) { d_idx = j; break; }
                float p_d = (d_idx >= 0) ? probs[d_idx] : 0.0f;
                if (u01(rng) < p_d) { emitted.push_back(d_tok); acc++; }
                else { emitted.push_back(sample_categorical(t64_ids + i * 64, probs, rng, d_idx)); rejected = true; break; }
            }
            if (!rejected) {
                float probs[64]; build_target_probs(t64_vals + (M - 1) * 64, sc, probs);
                emitted.push_back(sample_categorical(t64_ids + (M - 1) * 64, probs, rng));
            }
        }
        st.accepted += acc;
        int n_emit = int(emitted.size());
        if (M > 1) alpha = std::min(0.98, std::max(0.05, 0.70 * alpha + 0.30 * (double(acc) / (M - 1))));
        else       alpha += 0.05 * (0.75 - alpha);

        for (int i = 0; i < n_emit; ++i) {
            int32_t tok = emitted[i];
            eng.cached_ids.push_back(batch[i]);
            st.tokens.push_back(tok); sfx.push(tok); nxt = tok;
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
            printf("Usage: %s <model_dir> [--bench 128] [--daemon] [--spec 7] [--max-seq 4096]\n", argv[0]);
            return 1;
        }
        std::string model_path = argv[1], state_path = "";
        bool daemon_mode = false, willneed = false;
        int bench_toks = 0, max_spec = 7, req_max_seq = 4096;
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
        }

        TransformerEngine eng;
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
            SparseRepParams rp1{1, eng.vocab, 0, 1.0f, {}};
            int32_t tok = 9707;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < bench_toks; ++i) eng.run_step_m(&tok, 1, i, &tok, true, rp1);
            double ms1 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            printf("▸ Live Decode Bench (Greedy M=1):      %d toks in %.1f ms => %.1f tok/s\n", bench_toks, ms1, bench_toks * 1000.0 / ms1);

            std::vector<int32_t> pattern = {16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220, 22, 11, 220, 23, 198};
            std::vector<int32_t> prompt;
            for (int r = 0; r < 8; ++r) prompt.insert(prompt.end(), pattern.begin(), pattern.end());
            SamplingConfig sc_g{};
            eng.reset_states(); SpecStats g0 = generate_adaptive(eng, prompt, 0, bench_toks, 0, sc_g, cost, false, true);
            eng.reset_states(); SpecStats gs = generate_adaptive(eng, prompt, 0, bench_toks, max_spec, sc_g, cost, false, true);
            int match = 0, lim = int(std::min(g0.tokens.size(), gs.tokens.size()));
            while (match < lim && g0.tokens[match] == gs.tokens[match]) match++;
            int n_dec = std::max(0, int(gs.tokens.size()) - 1);
            printf("▸ Live Speculative Bench (Greedy k=%d): %d toks in %.1f ms => %.1f tok/s (%.2f tok/step, match=%d/%d)\n",
                   max_spec, n_dec, gs.decode_ms, n_dec * 1000.0 / std::max(gs.decode_ms, 0.01),
                   double(n_dec) / std::max(gs.steps, 1), match, lim);
            return 0;
        }

        if (daemon_mode) {
            printf("{\"ready\":true,\"version\":\"1.0\",\"arch\":\"%s\",\"model_dir\":\"%s\",\"top_k\":0,\"max_spec\":%d,\"baked_toks\":%d}\n",
                   eng.arch_name.c_str(), eng.model_dir.c_str(), max_spec, eng.ram_snap.pos);
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
