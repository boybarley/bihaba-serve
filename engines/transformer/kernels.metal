// ============================================================================
// bihaba-serve — Native Apple Metal Compute Kernels (Standard GQA Transformers)
// Powered by Taalas-Metal Engine | Created by Boy Barley from Bihaba (Team Pasai Aceh)
//
// Supports: Qwen2.5 (0.5B-7B), Qwen3 (0.6B-8B, Bihaba-1.7B), Llama-3.2 (1B-3B), MNN-Pocket (0.3B)
//   • Native FP16 & BF16 affine quantized matvec/matmul (2b / 3b / 4b / 6b / 8b)
//   • BM=8, BN=32, BK=64 Compact-Half Hardware MMA (simdgroup_half8x8)
//   • Unified QK-Norm + Optional QKV Bias + Full Split-Half RoPE (hd = 64 / 128 / 256)
//   • Multi-Token Speculative GQA FlashAttention (M=1..8) + Exact Top-64 Logit Extractor
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant bool ADD_RES_C [[function_constant(0)]];
constant bool IS_BF16_C [[function_constant(1)]];

constant float LOG2E = 1.4426950408889634f;

struct NormParams      { int dim; float eps; };
struct QMVParams       { int K, N, gs_shift; };
struct QMVMultiParams  { int M, K, N, gs_shift; };
struct TrRopeParams    { int M, n_heads, n_kv_heads, head_dim, pos, max_seq, has_qkv_bias, has_qk_norm; float eps; };
struct TrGQAParams     { int M, n_heads, n_kv_heads, pos, max_seq; };

// ─────────────────────── 0. Native FP16 / BF16 Scale-Bias Loader ───────────────────────
inline half load_sb_h(device const ushort* ptr, ulong idx) {
    const ushort u = ptr[idx];
    return IS_BF16_C ? half(as_type<float>(uint(u) << 16)) : as_type<half>(u);
}

inline float load_sb_f(device const ushort* ptr, ulong idx) {
    const ushort u = ptr[idx];
    return IS_BF16_C ? as_type<float>(uint(u) << 16) : float(as_type<half>(u));
}

// ─────────────────────── 1. Multi-Token Embedding Lookup (2b/3b/4b/6b/8b) ───────────────────────
kernel void embed_lookup_q(
    device const int*      tok_ids [[buffer(0)]],
    device const uint32_t* w       [[buffer(1)]],
    device const ushort*   scales  [[buffer(2)]],
    device const ushort*   biases  [[buffer(3)]],
    device half*           out     [[buffer(4)]],
    constant QMVParams&    p       [[buffer(5)]],
    constant int&          bits    [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int d = int(gid.x), m = int(gid.y);
    if (d >= p.K) return;
    const ulong tid = ulong(tok_ids[m]);
    const ulong kg  = ulong(p.K >> p.gs_shift);
    const ulong g   = ulong(d >> p.gs_shift);
    const half s = load_sb_h(scales, tid * kg + g);
    const half b = load_sb_h(biases, tid * kg + g);
    uint32_t q = 0u;

    if (bits == 4) {
        const uint32_t word = w[tid * ulong(p.K >> 3) + ulong(d >> 3)];
        q = (word >> ((d & 7) << 2)) & 0xFu;
    } else if (bits == 8) {
        const uint32_t word = w[tid * ulong(p.K >> 2) + ulong(d >> 2)];
        q = (word >> ((d & 3) << 3)) & 0xFFu;
    } else if (bits == 2) {
        const uint32_t word = w[tid * ulong(p.K >> 4) + ulong(d >> 4)];
        q = (word >> ((d & 15) << 1)) & 0x3u;
    } else if (bits == 6) {
        const ulong kw = ulong((p.K * 6) >> 5);
        const ulong blk = ulong(d >> 4) * 3UL;
        const int idx = d & 15;
        const uint32_t w0 = w[tid * kw + blk + 0];
        const uint32_t w1 = w[tid * kw + blk + 1];
        const uint32_t w2 = w[tid * kw + blk + 2];
        if (idx < 5)        q = (w0 >> (6 * idx)) & 63u;
        else if (idx == 5)  q = ((w0 >> 30) & 3u) | ((w1 & 15u) << 2);
        else if (idx < 10)  q = (w1 >> (4 + 6 * (idx - 6))) & 63u;
        else if (idx == 10) q = ((w1 >> 28) & 15u) | ((w2 & 3u) << 4);
        else                q = (w2 >> (2 + 6 * (idx - 11))) & 63u;
    } else {
        const ulong kw = ulong((p.K * 3) >> 5);
        const ulong blk = ulong(d >> 5) * 3UL;
        const int idx = d & 31;
        const uint32_t w0 = w[tid * kw + blk + 0];
        const uint32_t w1 = w[tid * kw + blk + 1];
        const uint32_t w2 = w[tid * kw + blk + 2];
        if (idx < 10)       q = (w0 >> (3 * idx)) & 7u;
        else if (idx == 10) q = ((w0 >> 30) & 3u) | ((w1 & 1u) << 2);
        else if (idx < 21)  q = (w1 >> (1 + 3 * (idx - 11))) & 7u;
        else if (idx == 21) q = ((w1 >> 31) & 1u) | ((w2 & 3u) << 1);
        else                q = (w2 >> (2 + 3 * (idx - 22))) & 7u;
    }
    out[m * p.K + d] = fma(s, half(q), b);
}

// ─────────────────────── 2. Multi-Token Parallel RMSNorm ───────────────────────
kernel void rms_norm(
    device const half*   x      [[buffer(0)]],
    device const half*   weight [[buffer(1)]],
    device half*         out    [[buffer(2)]],
    constant NormParams& p      [[buffer(3)]],
    uint tg_x [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int m = int(tg_x);
    threadgroup float partial[8];
    const int d4 = p.dim >> 2;
    const device half4* xm = (const device half4*)(x + m * p.dim);
    float ss = 0.0f;
    for (int i = int(tid); i < d4; i += 256) { float4 v = float4(xm[i]); ss += dot(v, v); }
    ss = simd_sum(ss);
    if (lane == 0) partial[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float total = simd_sum((lane < 8) ? partial[lane] : 0.0f);
        if (lane == 0) partial[0] = rsqrt(total / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    const device half4* wm = (const device half4*)weight;
    device half4* om = (device half4*)(out + m * p.dim);
    for (int i = int(tid); i < d4; i += 256) om[i] = half4(float4(xm[i]) * inv * float4(wm[i]));
}

// ─────────────────────── 3. Quantized Word Dot Helpers (2b, 3b, 4b, 6b, 8b) ───────────────────────
inline float dot_4bit_word(const uint32_t wv, const half2 hs, const half2 hb, const device half4* xp) {
    const half2 c1024 = half2(1024.0h);
    const half4 lo = xp[0], hi = xp[1];
    half2 acc = half2(lo.x, hi.x) * fma(hs, as_type<half2>(( wv        & 0x000F000Fu) | 0x64006400u) - c1024, hb);
    acc = fma(half2(lo.y, hi.y),    fma(hs, as_type<half2>(((wv >> 4)  & 0x000F000Fu) | 0x64006400u) - c1024, hb), acc);
    acc = fma(half2(lo.z, hi.z),    fma(hs, as_type<half2>(((wv >> 8)  & 0x000F000Fu) | 0x64006400u) - c1024, hb), acc);
    acc = fma(half2(lo.w, hi.w),    fma(hs, as_type<half2>(((wv >> 12) & 0x000F000Fu) | 0x64006400u) - c1024, hb), acc);
    return float(acc.x) + float(acc.y);
}

inline float dot_2bit_word(const uint32_t wv, const half2 hs, const half2 hb, const device half4* xp) {
    const half2 c1024 = half2(1024.0h);
    const half4 a = xp[0], b = xp[1], c = xp[2], d = xp[3];
    half2 acc = half2(a.x, c.x) * fma(hs, as_type<half2>(( wv        & 0x00030003u) | 0x64006400u) - c1024, hb);
    acc = fma(half2(a.y, c.y),    fma(hs, as_type<half2>(((wv >> 2)  & 0x00030003u) | 0x64006400u) - c1024, hb), acc);
    acc = fma(half2(a.z, c.z),    fma(hs, as_type<half2>(((wv >> 4)  & 0x00030003u) | 0x64006400u) - c1024, hb), acc);
    acc = fma(half2(a.w, c.w),    fma(hs, as_type<half2>(((wv >> 6)  & 0x00030003u) | 0x64006400u) - c1024, hb), acc);
    acc = fma(half2(b.x, d.x),    fma(hs, as_type<half2>(((wv >> 8)  & 0x00030003u) | 0x64006400u) - c1024, hb), acc);
    acc = fma(half2(b.y, d.y),    fma(hs, as_type<half2>(((wv >> 10) & 0x00030003u) | 0x64006400u) - c1024, hb), acc);
    acc = fma(half2(b.z, d.z),    fma(hs, as_type<half2>(((wv >> 12) & 0x00030003u) | 0x64006400u) - c1024, hb), acc);
    acc = fma(half2(b.w, d.w),    fma(hs, as_type<half2>(((wv >> 14) & 0x00030003u) | 0x64006400u) - c1024, hb), acc);
    return float(acc.x) + float(acc.y);
}

inline float dot_8bit_word(const uint32_t wv, const float s, const float b, const half4 xv) {
    const float4 q = float4(float(wv & 0xFFu), float((wv >> 8) & 0xFFu), float((wv >> 16) & 0xFFu), float((wv >> 24) & 0xFFu));
    return dot(float4(xv), fma(q, float4(s), float4(b)));
}

inline float dot_6bit_block16(const uint32_t w0, const uint32_t w1, const uint32_t w2,
                              const float s, const float b, const device half4* xp) {
    const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
    float sum_x  = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w)
                 + (x2.x + x2.y + x2.z + x2.w) + (x3.x + x3.y + x3.z + x3.w);
    float sum_qx = dot(x0, float4(float(w0 & 63u), float((w0 >> 6) & 63u), float((w0 >> 12) & 63u), float((w0 >> 18) & 63u)))
                 + dot(x1, float4(float((w0 >> 24) & 63u), float(((w0 >> 30) & 3u) | ((w1 & 15u) << 2)), float((w1 >> 4) & 63u), float((w1 >> 10) & 63u)))
                 + dot(x2, float4(float((w1 >> 16) & 63u), float((w1 >> 22) & 63u), float(((w1 >> 28) & 15u) | ((w2 & 3u) << 4)), float((w2 >> 2) & 63u)))
                 + dot(x3, float4(float((w2 >> 8) & 63u), float((w2 >> 14) & 63u), float((w2 >> 20) & 63u), float((w2 >> 26) & 63u)));
    return fma(s, sum_qx, b * sum_x);
}

inline float dot_3bit_block32(const uint32_t w0, const uint32_t w1, const uint32_t w2,
                              const float s, const float b, const device half4* xp) {
    const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
    float sum_x  = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w) + (x2.x + x2.y + x2.z + x2.w) + (x3.x + x3.y + x3.z + x3.w);
    float sum_qx = dot(x0, float4(float(w0 & 7u), float((w0 >> 3) & 7u), float((w0 >> 6) & 7u), float((w0 >> 9) & 7u)))
                 + dot(x1, float4(float((w0 >> 12) & 7u), float((w0 >> 15) & 7u), float((w0 >> 18) & 7u), float((w0 >> 21) & 7u)))
                 + dot(x2, float4(float((w0 >> 24) & 7u), float((w0 >> 27) & 7u), float(((w0 >> 30) & 3u) | ((w1 & 1u) << 2)), float((w1 >> 1) & 7u)))
                 + dot(x3, float4(float((w1 >> 4) & 7u), float((w1 >> 7) & 7u), float((w1 >> 10) & 7u), float((w1 >> 13) & 7u)));
    const float4 x4 = float4(xp[4]), x5 = float4(xp[5]), x6 = float4(xp[6]), x7 = float4(xp[7]);
    sum_x += (x4.x + x4.y + x4.z + x4.w) + (x5.x + x5.y + x5.z + x5.w) + (x6.x + x6.y + x6.z + x6.w) + (x7.x + x7.y + x7.z + x7.w);
    sum_qx += dot(x4, float4(float((w1 >> 16) & 7u), float((w1 >> 19) & 7u), float((w1 >> 22) & 7u), float((w1 >> 25) & 7u)))
            + dot(x5, float4(float((w1 >> 28) & 7u), float(((w1 >> 31) & 1u) | ((w2 & 3u) << 1)), float((w2 >> 2) & 7u), float((w2 >> 5) & 7u)))
            + dot(x6, float4(float((w2 >> 8) & 7u), float((w2 >> 11) & 7u), float((w2 >> 14) & 7u), float((w2 >> 17) & 7u)))
            + dot(x7, float4(float((w2 >> 20) & 7u), float((w2 >> 23) & 7u), float((w2 >> 26) & 7u), float((w2 >> 29) & 7u)));
    return fma(s, sum_qx, b * sum_x);
}

// ─────────────────────── 4A. Fast M=1 Dual-Row QMV (2b / 3b / 4b / 6b / 8b, 4-Byte Safe!) ───────────────────────
kernel void qmv_4b(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int K = p.K, N = p.N, gsh = p.gs_shift;
    const int row0 = (int(tg_x) * 4 + int(sg)) * 2;
    if (row0 >= N) return;
    const int r0 = row0, r1 = min(row0 + 1, N - 1);
    const ulong kw = ulong(K >> 3), kg = ulong(K >> gsh);
    const ulong wo0 = ulong(r0) * kw, wo1 = ulong(r1) * kw;
    const ulong so0 = ulong(r0) * kg, so1 = ulong(r1) * kg;
    float a0 = 0.0f, a1 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const int wi = gk >> 3, g = gk >> gsh;
        const half2 hs0 = half2(load_sb_h(scales, so0 + g)), hb0 = half2(load_sb_h(biases, so0 + g));
        const half2 hs1 = half2(load_sb_h(scales, so1 + g)), hb1 = half2(load_sb_h(biases, so1 + g));
        const device half4* xp = (const device half4*)(x + gk);
        a0 += dot_4bit_word(w[wo0+wi], hs0, hb0, xp) + dot_4bit_word(w[wo0+wi+1], hs0, hb0, xp + 2);
        a1 += dot_4bit_word(w[wo1+wi], hs1, hb1, xp) + dot_4bit_word(w[wo1+wi+1], hs1, hb1, xp + 2);
    }
    const float v0 = simd_sum(a0), v1 = simd_sum(a1);
    if (lane == 0) {
        if (ADD_RES_C) { out[r0] += half(v0); if (row0 + 1 < N) out[r1] += half(v1); }
        else           { out[r0]  = half(v0); if (row0 + 1 < N) out[r1]  = half(v1); }
    }
}

kernel void qmv_2b(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int K = p.K, N = p.N, gsh = p.gs_shift;
    const int row0 = (int(tg_x) * 4 + int(sg)) * 2;
    if (row0 >= N) return;
    const int r0 = row0, r1 = min(row0 + 1, N - 1);
    const ulong kw = ulong(K >> 4), kg = ulong(K >> gsh);
    const ulong wo0 = ulong(r0) * kw, wo1 = ulong(r1) * kw;
    const ulong so0 = ulong(r0) * kg, so1 = ulong(r1) * kg;
    float a0 = 0.0f, a1 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const int kwi = gk >> 4, g = gk >> gsh;
        const device half4* xp = (const device half4*)(x + gk);
        a0 += dot_2bit_word(w[wo0 + kwi], half2(load_sb_h(scales, so0 + g)), half2(load_sb_h(biases, so0 + g)), xp);
        a1 += dot_2bit_word(w[wo1 + kwi], half2(load_sb_h(scales, so1 + g)), half2(load_sb_h(biases, so1 + g)), xp);
    }
    const float v0 = simd_sum(a0), v1 = simd_sum(a1);
    if (lane == 0) {
        if (ADD_RES_C) { out[r0] += half(v0); if (row0 + 1 < N) out[r1] += half(v1); }
        else           { out[r0]  = half(v0); if (row0 + 1 < N) out[r1]  = half(v1); }
    }
}

kernel void qmv_3b(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int K = p.K, N = p.N, gsh = p.gs_shift;
    const int row0 = (int(tg_x) * 4 + int(sg)) * 2;
    if (row0 >= N) return;
    const int r0 = row0, r1 = min(row0 + 1, N - 1);
    const ulong kw = ulong((K * 3) >> 5), kg = ulong(K >> gsh);
    const ulong wo0 = ulong(r0) * kw, wo1 = ulong(r1) * kw;
    const ulong so0 = ulong(r0) * kg, so1 = ulong(r1) * kg;
    float a0 = 0.0f, a1 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 1024) {
        const int gk = k0 + int(lane) * 32;
        if (gk >= K) break;
        const int blk = (gk >> 5) * 3, g = gk >> gsh;
        const device half4* xp = (const device half4*)(x + gk);
        a0 += dot_3bit_block32(w[wo0+blk], w[wo0+blk+1], w[wo0+blk+2], load_sb_f(scales, so0+g), load_sb_f(biases, so0+g), xp);
        a1 += dot_3bit_block32(w[wo1+blk], w[wo1+blk+1], w[wo1+blk+2], load_sb_f(scales, so1+g), load_sb_f(biases, so1+g), xp);
    }
    const float v0 = simd_sum(a0), v1 = simd_sum(a1);
    if (lane == 0) {
        if (ADD_RES_C) { out[r0] += half(v0); if (row0 + 1 < N) out[r1] += half(v1); }
        else           { out[r0]  = half(v0); if (row0 + 1 < N) out[r1]  = half(v1); }
    }
}

kernel void qmv_6b(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int K = p.K, N = p.N, gsh = p.gs_shift;
    const int row0 = (int(tg_x) * 4 + int(sg)) * 2;
    if (row0 >= N) return;
    const int r0 = row0, r1 = min(row0 + 1, N - 1);
    const ulong kw = ulong((K * 6) >> 5), kg = ulong(K >> gsh);
    const ulong wo0 = ulong(r0) * kw, wo1 = ulong(r1) * kw;
    const ulong so0 = ulong(r0) * kg, so1 = ulong(r1) * kg;
    float a0 = 0.0f, a1 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const int blk = (gk >> 4) * 3, g = gk >> gsh;
        const device half4* xp = (const device half4*)(x + gk);
        a0 += dot_6bit_block16(w[wo0+blk], w[wo0+blk+1], w[wo0+blk+2], load_sb_f(scales, so0+g), load_sb_f(biases, so0+g), xp);
        a1 += dot_6bit_block16(w[wo1+blk], w[wo1+blk+1], w[wo1+blk+2], load_sb_f(scales, so1+g), load_sb_f(biases, so1+g), xp);
    }
    const float v0 = simd_sum(a0), v1 = simd_sum(a1);
    if (lane == 0) {
        if (ADD_RES_C) { out[r0] += half(v0); if (row0 + 1 < N) out[r1] += half(v1); }
        else           { out[r0]  = half(v0); if (row0 + 1 < N) out[r1]  = half(v1); }
    }
}

kernel void qmv_8b(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int K = p.K, N = p.N, gsh = p.gs_shift;
    const int row0 = (int(tg_x) * 4 + int(sg)) * 2;
    if (row0 >= N) return;
    const int r0 = row0, r1 = min(row0 + 1, N - 1);
    const ulong kw = ulong(K >> 2), kg = ulong(K >> gsh);
    const ulong wo0 = ulong(r0) * kw, wo1 = ulong(r1) * kw;
    const ulong so0 = ulong(r0) * kg, so1 = ulong(r1) * kg;
    float a0 = 0.0f, a1 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const int wi = gk >> 2, g = gk >> gsh;
        const float s0 = load_sb_f(scales, so0 + g), b0 = load_sb_f(biases, so0 + g);
        const float s1 = load_sb_f(scales, so1 + g), b1 = load_sb_f(biases, so1 + g);
        const device half4* xp = (const device half4*)(x + gk);
        const half4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
        a0 += dot_8bit_word(w[wo0+wi], s0, b0, x0) + dot_8bit_word(w[wo0+wi+1], s0, b0, x1) + dot_8bit_word(w[wo0+wi+2], s0, b0, x2) + dot_8bit_word(w[wo0+wi+3], s0, b0, x3);
        a1 += dot_8bit_word(w[wo1+wi], s1, b1, x0) + dot_8bit_word(w[wo1+wi+1], s1, b1, x1) + dot_8bit_word(w[wo1+wi+2], s1, b1, x2) + dot_8bit_word(w[wo1+wi+3], s1, b1, x3);
    }
    const float v0 = simd_sum(a0), v1 = simd_sum(a1);
    if (lane == 0) {
        if (ADD_RES_C) { out[r0] += half(v0); if (row0 + 1 < N) out[r1] += half(v1); }
        else           { out[r0]  = half(v0); if (row0 + 1 < N) out[r1]  = half(v1); }
    }
}

// ─────────────────────── 4B. Multi-Token Weight-Reuse QMV (M=2..8) ───────────────────────
template <int BITS>
inline void qmv_m_impl(
    device const half* x, device const uint32_t* w, device const ushort* scales, device const ushort* biases,
    device half* out, constant QMVMultiParams& p, uint tg_x, uint sg, uint lane)
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift;
    const int row = int(tg_x) * 4 + int(sg);
    if (row >= N) return;
    const ulong kg = ulong(K >> gsh), so = ulong(row) * kg;
    float acc[8] = {0,0,0,0,0,0,0,0};

    if (BITS == 4) {
        const ulong wo = ulong(row) * ulong(K >> 3);
        const half2 c1024 = half2(1024.0h);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const uint32_t wvx = w[wo + wi], wvy = w[wo + wi + 1];
            const half2 hs = half2(load_sb_h(scales, so + g)), hb = half2(load_sb_h(biases, so + g));
            const half2 w0 = fma(hs, as_type<half2>(( wvx        & 0x000F000Fu) | 0x64006400u) - c1024, hb);
            const half2 w1 = fma(hs, as_type<half2>(((wvx >> 4)  & 0x000F000Fu) | 0x64006400u) - c1024, hb);
            const half2 w2_ = fma(hs, as_type<half2>(((wvx >> 8)  & 0x000F000Fu) | 0x64006400u) - c1024, hb);
            const half2 w3 = fma(hs, as_type<half2>(((wvx >> 12) & 0x000F000Fu) | 0x64006400u) - c1024, hb);
            const half2 w4 = fma(hs, as_type<half2>(( wvy        & 0x000F000Fu) | 0x64006400u) - c1024, hb);
            const half2 w5 = fma(hs, as_type<half2>(((wvy >> 4)  & 0x000F000Fu) | 0x64006400u) - c1024, hb);
            const half2 w6 = fma(hs, as_type<half2>(((wvy >> 8)  & 0x000F000Fu) | 0x64006400u) - c1024, hb);
            const half2 w7 = fma(hs, as_type<half2>(((wvy >> 12) & 0x000F000Fu) | 0x64006400u) - c1024, hb);
            const half4 wa = half4(w0.x, w1.x, w2_.x, w3.x), wb = half4(w0.y, w1.y, w2_.y, w3.y);
            const half4 wc = half4(w4.x, w5.x, w6.x,  w7.x), wd = half4(w4.y, w5.y, w6.y,  w7.y);
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                half4 s = fma(wd, xp[3], fma(wc, xp[2], fma(wb, xp[1], wa * xp[0])));
                acc[m] += float(s.x + s.y) + float(s.z + s.w);
            }
        }
    } else if (BITS == 6) {
        const ulong wo = ulong(row) * ulong((K * 6) >> 5);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int blk = (gk >> 4) * 3, g = gk >> gsh;
            const uint32_t w0 = w[wo+blk], w1 = w[wo+blk+1], w2 = w[wo+blk+2];
            const float s = load_sb_f(scales, so+g), b = load_sb_f(biases, so+g);
            for (int m = 0; m < M; ++m)
                acc[m] += dot_6bit_block16(w0, w1, w2, s, b, (const device half4*)(x + m * K + gk));
        }
    } else if (BITS == 3) {
        const ulong wo = ulong(row) * ulong((K * 3) >> 5);
        for (int k0 = 0; k0 < K; k0 += 1024) {
            const int gk = k0 + int(lane) * 32;
            if (gk >= K) break;
            const int blk = (gk >> 5) * 3, g = gk >> gsh;
            const uint32_t w0 = w[wo+blk], w1 = w[wo+blk+1], w2 = w[wo+blk+2];
            const float s = load_sb_f(scales, so+g), b = load_sb_f(biases, so+g);
            for (int m = 0; m < M; ++m)
                acc[m] += dot_3bit_block32(w0, w1, w2, s, b, (const device half4*)(x + m * K + gk));
        }
    } else if (BITS == 2) {
        const ulong wo = ulong(row) * ulong(K >> 4);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const uint32_t wv = w[wo + (gk >> 4)];
            const int g = gk >> gsh;
            const half2 hs = half2(load_sb_h(scales, so+g)), hb = half2(load_sb_h(biases, so+g));
            for (int m = 0; m < M; ++m)
                acc[m] += dot_2bit_word(wv, hs, hb, (const device half4*)(x + m * K + gk));
        }
    } else {
        const ulong wo = ulong(row) * ulong(K >> 2);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 2, g = gk >> gsh;
            const uint32_t w0 = w[wo+wi], w1 = w[wo+wi+1], w2 = w[wo+wi+2], w3 = w[wo+wi+3];
            const float s = load_sb_f(scales, so+g), b = load_sb_f(biases, so+g);
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                acc[m] += dot_8bit_word(w0, s, b, xp[0]) + dot_8bit_word(w1, s, b, xp[1])
                        + dot_8bit_word(w2, s, b, xp[2]) + dot_8bit_word(w3, s, b, xp[3]);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) {
            if (ADD_RES_C) out[m * N + row] += half(v);
            else           out[m * N + row]  = half(v);
        }
    }
}

#define DECL_QMVM(NAME, BITS) \
kernel void NAME(device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]], \
    device const ushort* s [[buffer(2)]], device const ushort* b [[buffer(3)]], device half* out [[buffer(4)]], \
    constant QMVMultiParams& p [[buffer(5)]], \
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ qmv_m_impl<BITS>(x, w, s, b, out, p, tg_x, sg, lane); }

DECL_QMVM(qmv_2b_m, 2)
DECL_QMVM(qmv_3b_m, 3)
DECL_QMVM(qmv_4b_m, 4)
DECL_QMVM(qmv_6b_m, 6)
DECL_QMVM(qmv_8b_m, 8)

// ─────────────────────── 4C. Fused Single-Pass Gate+Up SwiGLU (2b/3b/4b/6b/8b) ───────────────────────
template <int BITS>
inline void dense_swiglu_m_impl(
    device const half* x,
    device const uint32_t* wg, device const ushort* sg_a, device const ushort* bg_a,
    device const uint32_t* wu, device const ushort* su_a, device const ushort* bu_a,
    device half* out, constant QMVMultiParams& p, uint tg_x, uint sg, uint lane)
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift;
    const int row = int(tg_x) * 4 + int(sg);
    if (row >= N) return;
    const ulong kg = ulong(K >> gsh), so = ulong(row) * kg;
    float ag[8] = {0,0,0,0,0,0,0,0}, au[8] = {0,0,0,0,0,0,0,0};

    if (BITS == 4) {
        const ulong wo = ulong(row) * ulong(K >> 3);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const uint32_t ga = wg[wo+wi], gb = wg[wo+wi+1], ua = wu[wo+wi], ub = wu[wo+wi+1];
            const half2 hsg = half2(load_sb_h(sg_a, so+g)), hbg = half2(load_sb_h(bg_a, so+g));
            const half2 hsu = half2(load_sb_h(su_a, so+g)), hbu = half2(load_sb_h(bu_a, so+g));
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                ag[m] += dot_4bit_word(ga, hsg, hbg, xp) + dot_4bit_word(gb, hsg, hbg, xp + 2);
                au[m] += dot_4bit_word(ua, hsu, hbu, xp) + dot_4bit_word(ub, hsu, hbu, xp + 2);
            }
        }
    } else if (BITS == 6) {
        const ulong wo = ulong(row) * ulong((K * 6) >> 5);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int blk = (gk >> 4) * 3, g = gk >> gsh;
            const uint32_t g0 = wg[wo+blk], g1 = wg[wo+blk+1], g2 = wg[wo+blk+2];
            const uint32_t u0 = wu[wo+blk], u1 = wu[wo+blk+1], u2 = wu[wo+blk+2];
            const float sgv = load_sb_f(sg_a, so+g), bgv = load_sb_f(bg_a, so+g);
            const float suv = load_sb_f(su_a, so+g), buv = load_sb_f(bu_a, so+g);
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                ag[m] += dot_6bit_block16(g0, g1, g2, sgv, bgv, xp);
                au[m] += dot_6bit_block16(u0, u1, u2, suv, buv, xp);
            }
        }
    } else if (BITS == 3) {
        const ulong wo = ulong(row) * ulong((K * 3) >> 5);
        for (int k0 = 0; k0 < K; k0 += 1024) {
            const int gk = k0 + int(lane) * 32;
            if (gk >= K) break;
            const int blk = (gk >> 5) * 3, g = gk >> gsh;
            const uint32_t g0 = wg[wo+blk], g1 = wg[wo+blk+1], g2 = wg[wo+blk+2];
            const uint32_t u0 = wu[wo+blk], u1 = wu[wo+blk+1], u2 = wu[wo+blk+2];
            const float sgv = load_sb_f(sg_a, so+g), bgv = load_sb_f(bg_a, so+g);
            const float suv = load_sb_f(su_a, so+g), buv = load_sb_f(bu_a, so+g);
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                ag[m] += dot_3bit_block32(g0, g1, g2, sgv, bgv, xp);
                au[m] += dot_3bit_block32(u0, u1, u2, suv, buv, xp);
            }
        }
    } else if (BITS == 2) {
        const ulong wo = ulong(row) * ulong(K >> 4);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int kwi = gk >> 4, g = gk >> gsh;
            const uint32_t wgv = wg[wo + kwi], wuv = wu[wo + kwi];
            const half2 hsg = half2(load_sb_h(sg_a, so+g)), hbg = half2(load_sb_h(bg_a, so+g));
            const half2 hsu = half2(load_sb_h(su_a, so+g)), hbu = half2(load_sb_h(bu_a, so+g));
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                ag[m] += dot_2bit_word(wgv, hsg, hbg, xp);
                au[m] += dot_2bit_word(wuv, hsu, hbu, xp);
            }
        }
    } else {
        const ulong wo = ulong(row) * ulong(K >> 2);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 2, g = gk >> gsh;
            const uint32_t g0 = wg[wo+wi], g1 = wg[wo+wi+1], g2 = wg[wo+wi+2], g3 = wg[wo+wi+3];
            const uint32_t u0 = wu[wo+wi], u1 = wu[wo+wi+1], u2 = wu[wo+wi+2], u3 = wu[wo+wi+3];
            const float sgv = load_sb_f(sg_a, so+g), bgv = load_sb_f(bg_a, so+g);
            const float suv = load_sb_f(su_a, so+g), buv = load_sb_f(bu_a, so+g);
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                ag[m] += dot_8bit_word(g0, sgv, bgv, xp[0]) + dot_8bit_word(g1, sgv, bgv, xp[1])
                       + dot_8bit_word(g2, sgv, bgv, xp[2]) + dot_8bit_word(g3, sgv, bgv, xp[3]);
                au[m] += dot_8bit_word(u0, suv, buv, xp[0]) + dot_8bit_word(u1, suv, buv, xp[1])
                       + dot_8bit_word(u2, suv, buv, xp[2]) + dot_8bit_word(u3, suv, buv, xp[3]);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        const float gv = simd_sum(ag[m]), uv = simd_sum(au[m]);
        if (lane == 0) out[m * N + row] = half((gv / (1.0f + exp(-gv))) * uv);
    }
}

#define DECL_SWIG_M(NAME, BITS) \
kernel void NAME(device const half* x [[buffer(0)]], \
    device const uint32_t* wg [[buffer(1)]], device const ushort* sg_a [[buffer(2)]], device const ushort* bg_a [[buffer(3)]], \
    device const uint32_t* wu [[buffer(4)]], device const ushort* su_a [[buffer(5)]], device const ushort* bu_a [[buffer(6)]], \
    device half* out [[buffer(7)]], constant QMVMultiParams& p [[buffer(8)]], \
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ dense_swiglu_m_impl<BITS>(x, wg, sg_a, bg_a, wu, su_a, bu_a, out, p, tg_x, sg, lane); }

DECL_SWIG_M(dense_swiglu_2b, 2)
DECL_SWIG_M(dense_swiglu_3b, 3)
DECL_SWIG_M(dense_swiglu_4b, 4)
DECL_SWIG_M(dense_swiglu_6b, 6)
DECL_SWIG_M(dense_swiglu_8b, 8)

// ─────────────────────── 4D. BM=8·MT, BN=32, BK=64 Compact-Half Hardware MMA ───────────────────────
// MT = jumlah tile 8-token. MT=1: verifikasi spekulatif M=4..8. MT=4: prefill M=9..32.
// Tile bobot (32 baris × 64 K) di-dekuantisasi SEKALI ke threadgroup memory lalu dipakai untuk semua
// MT tile token — inilah yang membuat prefill 32 token hanya sedikit lebih mahal dari 8 token.
inline void dequant_4bit_to_tg(const uint32_t wv, const half2 hs, const half2 hb, threadgroup half* dst) {
    const half2 c1024 = half2(1024.0h);
    const half2 p0 = fma(hs, as_type<half2>(( wv        & 0x000F000Fu) | 0x64006400u) - c1024, hb);
    const half2 p1 = fma(hs, as_type<half2>(((wv >> 4)  & 0x000F000Fu) | 0x64006400u) - c1024, hb);
    const half2 p2 = fma(hs, as_type<half2>(((wv >> 8)  & 0x000F000Fu) | 0x64006400u) - c1024, hb);
    const half2 p3 = fma(hs, as_type<half2>(((wv >> 12) & 0x000F000Fu) | 0x64006400u) - c1024, hb);
    *((threadgroup half4*)(dst))     = half4(p0.x, p1.x, p2.x, p3.x);
    *((threadgroup half4*)(dst + 4)) = half4(p0.y, p1.y, p2.y, p3.y);
}

inline void dequant_2bit_to_tg(const uint32_t wv, const half2 hs, const half2 hb, threadgroup half* dst) {
    const half2 c1024 = half2(1024.0h);
    const half2 p0 = fma(hs, as_type<half2>(( wv        & 0x00030003u) | 0x64006400u) - c1024, hb);
    const half2 p1 = fma(hs, as_type<half2>(((wv >> 2)  & 0x00030003u) | 0x64006400u) - c1024, hb);
    const half2 p2 = fma(hs, as_type<half2>(((wv >> 4)  & 0x00030003u) | 0x64006400u) - c1024, hb);
    const half2 p3 = fma(hs, as_type<half2>(((wv >> 6)  & 0x00030003u) | 0x64006400u) - c1024, hb);
    *((threadgroup half4*)(dst))      = half4(p0.x, p1.x, p2.x, p3.x);
    *((threadgroup half4*)(dst + 8))  = half4(p0.y, p1.y, p2.y, p3.y);
    const half2 p4 = fma(hs, as_type<half2>(((wv >> 8)  & 0x00030003u) | 0x64006400u) - c1024, hb);
    const half2 p5 = fma(hs, as_type<half2>(((wv >> 10) & 0x00030003u) | 0x64006400u) - c1024, hb);
    const half2 p6 = fma(hs, as_type<half2>(((wv >> 12) & 0x00030003u) | 0x64006400u) - c1024, hb);
    const half2 p7 = fma(hs, as_type<half2>(((wv >> 14) & 0x00030003u) | 0x64006400u) - c1024, hb);
    *((threadgroup half4*)(dst + 4))  = half4(p4.x, p5.x, p6.x, p7.x);
    *((threadgroup half4*)(dst + 12)) = half4(p4.y, p5.y, p6.y, p7.y);
}

// Muat tile token X[0..8·MT, k0..k0+64] ke Xs (baris di luar M diisi nol)
template <int MT>
inline void load_x_tile(device const half* x, int M, int K, int k0, threadgroup half* Xs, uint tid) {
    const int xm = int(tid) >> 4, xk = (int(tid) & 15) * 4;
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        const int m = xm + 8 * t;
        *((threadgroup half4*)(Xs + m * 64 + xk)) = (m < M)
            ? *((const device half4*)(x + m * K + k0 + xk)) : half4(0.0h);
    }
}

// Dekuantisasi generik 16 nilai berurutan (bit-stream LSB-first ala MLX) untuk 3/6/8-bit.
// k kelipatan 16 → offset bit awal: 8-bit & 6-bit selalu 0, 3-bit 0 atau 16. Tidak membaca melewati blok.
template <int BITS>
inline void dequant_gen16_to_tg(device const uint32_t* wrow, int k, float s, float b, threadgroup half* dst) {
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr int NW = (16 * BITS + 31) / 32 + 1;
    const uint bit0 = uint(k) * uint(BITS);
    const device uint32_t* p = wrow + (bit0 >> 5);
    const uint sh = bit0 & 31u;
    uint a[NW];
    #pragma unroll
    for (int i = 0; i < NW; ++i) a[i] = (uint(i) * 32u < sh + 16u * uint(BITS)) ? p[i] : 0u;
    half v[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint pos = sh + uint(j) * uint(BITS), wi = pos >> 5, o = pos & 31u;
        uint q = a[wi] >> o;
        if (o + uint(BITS) > 32u) q |= a[wi + 1] << (32u - o);
        v[j] = half(fma(float(q & MASK), s, b));
    }
    #pragma unroll
    for (int j = 0; j < 16; j += 4) *((threadgroup half4*)(dst + j)) = half4(v[j], v[j + 1], v[j + 2], v[j + 3]);
}

// Dekuantisasi tile bobot 32 baris × 64 K (4 thread per baris, 16 nilai per thread)
template <int BITS>
inline void load_w_tile(device const uint32_t* w, device const ushort* scales, device const ushort* biases,
                        int r, int K, int KG, int gsh, int k0, threadgroup half* Ws, uint tid) {
    constexpr int WS = 72;
    const int r_off = int(tid) >> 2, wi = int(tid) & 3;
    const ulong s_off = ulong(r) * ulong(KG) + ulong((k0 + wi * 16) >> gsh);
    const half2 hs = half2(load_sb_h(scales, s_off));
    const half2 hb = half2(load_sb_h(biases, s_off));
    threadgroup half* w_dst = Ws + r_off * WS + wi * 16;
    if (BITS == 4) {
        const ulong w_idx = ulong(r) * ulong(K >> 3) + ulong((k0 >> 3) + wi * 2);
        dequant_4bit_to_tg(w[w_idx],     hs, hb, w_dst);
        dequant_4bit_to_tg(w[w_idx + 1], hs, hb, w_dst + 8);
    } else if (BITS == 2) {
        dequant_2bit_to_tg(w[ulong(r) * ulong(K >> 4) + ulong((k0 >> 4) + wi)], hs, hb, w_dst);
    } else {
        const ulong row_words = ulong((K * BITS) >> 5);
        dequant_gen16_to_tg<BITS>(w + ulong(r) * row_words, k0 + wi * 16, float(hs.x), float(hb.x), w_dst);
    }
}

template <int BITS, int MT>
inline void qmm_bk64_impl(
    device const half* x, device const uint32_t* w, device const ushort* scales, device const ushort* biases,
    device half* out, QMVMultiParams p,
    threadgroup half* Xs, threadgroup half* Ws, threadgroup float* Cs,
    uint tg_x, uint tid, uint sg, uint lane)
{
    constexpr int BK = 64, WS = 72;
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift;
    const int KG = K >> gsh;
    const int row_base = int(tg_x) * 32;
    const int r = min(row_base + (int(tid) >> 2), N - 1);

    simdgroup_float8x8 c[MT];
    #pragma unroll
    for (int t = 0; t < MT; ++t) c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const threadgroup half* w_sg = Ws + (int(sg) * 8) * WS;
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_x_tile<MT>(x, M, K, k0, Xs, tid);
        load_w_tile<BITS>(w, scales, biases, r, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 bt;
            simdgroup_load(bt, w_sg + kk, WS, ulong2(0, 0), true);
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (t * 8 < M) {
                    simdgroup_half8x8 a;
                    simdgroup_load(a, Xs + t * 8 * BK + kk, BK);
                    simdgroup_multiply_accumulate(c[t], a, bt, c[t]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* cbuf = Cs + int(sg) * 64;
    const int n_base = row_base + int(sg) * 8;
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t * 8 >= M) break;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(c[t], cbuf, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int q = 0; q < 2; ++q) {
            const int idx = int(lane) * 2 + q, m = t * 8 + (idx >> 3), col = n_base + (idx & 7);
            if (m < M && col < N) {
                if (ADD_RES_C) out[m * N + col] += half(cbuf[idx]);
                else           out[m * N + col]  = half(cbuf[idx]);
            }
        }
    }
}

#define DECL_QMM(NAME, BITS, MT) \
kernel void NAME(device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]], \
    device const ushort* s [[buffer(2)]], device const ushort* b [[buffer(3)]], device half* out [[buffer(4)]], \
    constant QMVMultiParams& p [[buffer(5)]], \
    uint tg_x [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ \
    threadgroup half Xs[8 * MT * 64], Ws[32 * 72]; threadgroup float Cs[4 * 64]; \
    qmm_bk64_impl<BITS, MT>(x, w, s, b, out, p, Xs, Ws, Cs, tg_x, tid, sg, lane); \
}
DECL_QMM(qmm_4b_mma8, 4, 1)
DECL_QMM(qmm_2b_mma8, 2, 1)
DECL_QMM(qmm_4b_mmaP, 4, 4)
DECL_QMM(qmm_2b_mmaP, 2, 4)
DECL_QMM(qmm_3b_mma8, 3, 1)
DECL_QMM(qmm_3b_mmaP, 3, 4)
DECL_QMM(qmm_6b_mma8, 6, 1)
DECL_QMM(qmm_6b_mmaP, 6, 4)
DECL_QMM(qmm_8b_mma8, 8, 1)
DECL_QMM(qmm_8b_mmaP, 8, 4)

template <int BITS, int MT>
inline void swiglu_bk64_impl(
    device const half* x,
    device const uint32_t* wg, device const ushort* sg_a, device const ushort* bg_a,
    device const uint32_t* wu, device const ushort* su_a, device const ushort* bu_a,
    device half* out, QMVMultiParams p,
    threadgroup half* Xs, threadgroup half* Ws, threadgroup float* Cg_s, threadgroup float* Cu_s,
    uint tg_x, uint tid, uint sg, uint lane)
{
    constexpr int BK = 64, WS = 72;
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift, KG = K >> gsh;
    const int row_base = int(tg_x) * 32;
    const int r = min(row_base + (int(tid) >> 2), N - 1);

    simdgroup_float8x8 cg[MT], cu[MT];
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        cg[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        cu[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const threadgroup half* w_sg = Ws + (int(sg) * 8) * WS;

    for (int k0 = 0; k0 < K; k0 += BK) {
        load_x_tile<MT>(x, M, K, k0, Xs, tid);
        load_w_tile<BITS>(wg, sg_a, bg_a, r, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 bt;
            simdgroup_load(bt, w_sg + kk, WS, ulong2(0, 0), true);
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (t * 8 < M) {
                    simdgroup_half8x8 a;
                    simdgroup_load(a, Xs + t * 8 * BK + kk, BK);
                    simdgroup_multiply_accumulate(cg[t], a, bt, cg[t]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_w_tile<BITS>(wu, su_a, bu_a, r, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 bt;
            simdgroup_load(bt, w_sg + kk, WS, ulong2(0, 0), true);
            #pragma unroll
            for (int t = 0; t < MT; ++t) {
                if (t * 8 < M) {
                    simdgroup_half8x8 a;
                    simdgroup_load(a, Xs + t * 8 * BK + kk, BK);
                    simdgroup_multiply_accumulate(cu[t], a, bt, cu[t]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* gbuf = Cg_s + int(sg) * 64;
    threadgroup float* ubuf = Cu_s + int(sg) * 64;
    const int n_base = row_base + int(sg) * 8;
    #pragma unroll
    for (int t = 0; t < MT; ++t) {
        if (t * 8 >= M) break;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_store(cg[t], gbuf, 8);
        simdgroup_store(cu[t], ubuf, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int q = 0; q < 2; ++q) {
            const int idx = int(lane) * 2 + q, m = t * 8 + (idx >> 3), col = n_base + (idx & 7);
            if (m < M && col < N) {
                const float gv = gbuf[idx];
                out[m * N + col] = half((gv / (1.0f + exp(-gv))) * ubuf[idx]);
            }
        }
    }
}

#define DECL_SWIG_MMA(NAME, BITS, MT) \
kernel void NAME(device const half* x [[buffer(0)]], \
    device const uint32_t* wg [[buffer(1)]], device const ushort* sg_a [[buffer(2)]], device const ushort* bg_a [[buffer(3)]], \
    device const uint32_t* wu [[buffer(4)]], device const ushort* su_a [[buffer(5)]], device const ushort* bu_a [[buffer(6)]], \
    device half* out [[buffer(7)]], constant QMVMultiParams& p [[buffer(8)]], \
    uint tg_x [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ \
    threadgroup half Xs[8 * MT * 64], Ws[32 * 72]; threadgroup float Cg_s[4 * 64], Cu_s[4 * 64]; \
    swiglu_bk64_impl<BITS, MT>(x, wg, sg_a, bg_a, wu, su_a, bu_a, out, p, Xs, Ws, Cg_s, Cu_s, tg_x, tid, sg, lane); \
}
DECL_SWIG_MMA(dense_swiglu_4b_mma8, 4, 1)
DECL_SWIG_MMA(dense_swiglu_2b_mma8, 2, 1)
DECL_SWIG_MMA(dense_swiglu_4b_mmaP, 4, 4)
DECL_SWIG_MMA(dense_swiglu_2b_mmaP, 2, 4)
DECL_SWIG_MMA(dense_swiglu_3b_mma8, 3, 1)
DECL_SWIG_MMA(dense_swiglu_3b_mmaP, 3, 4)
DECL_SWIG_MMA(dense_swiglu_6b_mma8, 6, 1)
DECL_SWIG_MMA(dense_swiglu_6b_mmaP, 6, 4)
DECL_SWIG_MMA(dense_swiglu_8b_mma8, 8, 1)
DECL_SWIG_MMA(dense_swiglu_8b_mmaP, 8, 4)

// ─────────────────────── 5. Unified QK-Norm + Optional QKV Bias + Split-Half RoPE (hd=64/128/256) ───────────────────────
kernel void tr_qknorm_rope_kv(
    device half*       q_buf   [[buffer(0)]],
    device const half* k_in    [[buffer(1)]],
    device const half* v_in    [[buffer(2)]],
    device half*       k_cache [[buffer(3)]],
    device half*       v_cache [[buffer(4)]],
    device const half* cos_t   [[buffer(5)]],
    device const half* sin_t   [[buffer(6)]],
    device const half* q_norm  [[buffer(7)]],
    device const half* k_norm  [[buffer(8)]],
    device const half* q_bias  [[buffer(9)]],
    device const half* k_bias  [[buffer(10)]],
    device const half* v_bias  [[buffer(11)]],
    constant TrRopeParams& p   [[buffer(12)]],
    uint2 tg_id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    const int h = int(tg_id.x), m = int(tg_id.y);
    if (h >= p.n_heads || m >= p.M) return;

    const int hd = p.head_dim;
    const int iters = hd >> 5;       // 2 (hd=64), 4 (hd=128), 8 (hd=256)
    const int half_it = iters >> 1;  // 1, 2, 4
    const int half_r = hd >> 1;
    const int cur_pos = p.pos + m;
    const int q_stride = p.n_heads * hd;

    device half* q_ptr = q_buf + m * q_stride + h * hd;
    float qv[8], q_ss = 0.0f;
    for (int k = 0; k < iters; ++k) {
        const int idx = k * 32 + int(lane);
        float val = float(q_ptr[idx]);
        if (p.has_qkv_bias) val += float(q_bias[h * hd + idx]);
        qv[k] = val;
        q_ss += val * val;
    }
    if (p.has_qk_norm) {
        const float q_inv = rsqrt(simd_sum(q_ss) / float(hd) + p.eps);
        for (int k = 0; k < iters; ++k) qv[k] *= q_inv * float(q_norm[k * 32 + int(lane)]);
    }
    for (int k = 0; k < half_it; ++k) {
        const int ridx = k * 32 + int(lane);
        const float c = float(cos_t[cur_pos * half_r + ridx]);
        const float s = float(sin_t[cur_pos * half_r + ridx]);
        const float a = qv[k], b = qv[k + half_it];
        qv[k]           = a * c - b * s;
        qv[k + half_it] = b * c + a * s;
    }
    for (int k = 0; k < iters; ++k) q_ptr[k * 32 + int(lane)] = half(qv[k]);

    if (h < p.n_kv_heads) {
        const int kv_stride = p.n_kv_heads * hd;
        const device half* k_src = k_in + m * kv_stride + h * hd;
        const device half* v_src = v_in + m * kv_stride + h * hd;
        const ulong dst = (ulong(h) * ulong(p.max_seq) + ulong(cur_pos)) * ulong(hd);

        float kv[8], k_ss = 0.0f;
        for (int k = 0; k < iters; ++k) {
            const int idx = k * 32 + int(lane);
            float kval = float(k_src[idx]);
            float vval = float(v_src[idx]);
            if (p.has_qkv_bias) {
                kval += float(k_bias[h * hd + idx]);
                vval += float(v_bias[h * hd + idx]);
            }
            kv[k] = kval;
            k_ss += kval * kval;
            v_cache[dst + idx] = half(vval);
        }
        if (p.has_qk_norm) {
            const float k_inv = rsqrt(simd_sum(k_ss) / float(hd) + p.eps);
            for (int k = 0; k < iters; ++k) kv[k] *= k_inv * float(k_norm[k * 32 + int(lane)]);
        }
        for (int k = 0; k < half_it; ++k) {
            const int ridx = k * 32 + int(lane);
            const float c = float(cos_t[cur_pos * half_r + ridx]);
            const float s = float(sin_t[cur_pos * half_r + ridx]);
            const float a = kv[k], b = kv[k + half_it];
            kv[k]           = a * c - b * s;
            kv[k + half_it] = b * c + a * s;
        }
        for (int k = 0; k < iters; ++k) k_cache[dst + k * 32 + int(lane)] = half(kv[k]);
    }
}

// ─────────────────────── 6. Multi-Token GQA FlashAttention (hd = 64 / 128 / 256, M=1..8) ───────────────────────
kernel void gqa_attention_64(
    device const half* q_buf [[buffer(0)]], device const half* k_cache [[buffer(1)]], device const half* v_cache [[buffer(2)]],
    device half* attn_out [[buffer(3)]], constant TrGQAParams& p [[buffer(4)]],
    uint2 tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int h = int(tg_id.x), m = int(tg_id.y);
    if (h >= p.n_heads || m >= p.M) return;
    const int kv_h = h / (p.n_heads / p.n_kv_heads), seq_len = p.pos + m + 1;
    const float scale_log2e = (1.0f / 8.0f) * LOG2E;
    const int q_base = (m * p.n_heads + h) * 64 + int(lane) * 2;
    const ulong kv_head_base = ulong(kv_h) * ulong(p.max_seq) * 64UL;
    const int blk = (seq_len + 7) >> 3, t0 = int(sg) * blk, t1 = min(t0 + blk, seq_len);

    const float2 q0 = float2(*((const device half2*)(q_buf + q_base))) * scale_log2e;
    float2 o0 = float2(0.0f); float ms = -1e30f, se = 0.0f;
    for (int t = t0; t < t1; ++t) {
        const ulong ko = kv_head_base + ulong(t) * 64UL + ulong(lane * 2);
        const float s0 = simd_sum(dot(q0, float2(*((const device half2*)(k_cache + ko)))));
        const float nm = max(ms, s0), al = (se > 0.0f) ? exp2(ms - nm) : 0.0f, w0 = exp2(s0 - nm);
        se = se * al + w0;
        o0 = o0 * al + w0 * float2(*((const device half2*)(v_cache + ko)));
        ms = nm;
    }
    threadgroup float sm_max[8], sm_sum[8]; threadgroup float2 sm_o0[8][32];
    if (lane == 0) { sm_max[sg] = ms; sm_sum[sg] = se; }
    sm_o0[sg][lane] = o0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float gm = sm_max[0]; for (int s = 1; s < 8; ++s) gm = max(gm, sm_max[s]);
        float gs = 0.0f; float2 go0 = float2(0.0f);
        for (int s = 0; s < 8; ++s) if (sm_sum[s] > 0.0f) {
            const float ws = exp2(sm_max[s] - gm); gs += sm_sum[s] * ws; go0 += sm_o0[s][lane] * ws;
        }
        *((device half2*)(attn_out + q_base)) = half2(go0 / max(gs, 1e-12f));
    }
}

kernel void gqa_attention_128(
    device const half* q_buf [[buffer(0)]], device const half* k_cache [[buffer(1)]], device const half* v_cache [[buffer(2)]],
    device half* attn_out [[buffer(3)]], constant TrGQAParams& p [[buffer(4)]],
    uint2 tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int h = int(tg_id.x), m = int(tg_id.y);
    if (h >= p.n_heads || m >= p.M) return;
    const int kv_h = h / (p.n_heads / p.n_kv_heads), seq_len = p.pos + m + 1;
    const float scale_log2e = rsqrt(128.0f) * LOG2E;
    const int q_base = (m * p.n_heads + h) * 128 + int(lane) * 4;
    const ulong kv_head_base = ulong(kv_h) * ulong(p.max_seq) * 128UL;
    const int blk = (seq_len + 7) >> 3, t0 = int(sg) * blk, t1 = min(t0 + blk, seq_len);

    const float4 q0 = float4(*((const device half4*)(q_buf + q_base))) * scale_log2e;
    float4 o0 = float4(0.0f); float ms = -1e30f, se = 0.0f;
    for (int t = t0; t < t1; ++t) {
        const ulong ko = kv_head_base + ulong(t) * 128UL + ulong(lane * 4);
        const float s0 = simd_sum(dot(q0, float4(*((const device half4*)(k_cache + ko)))));
        const float nm = max(ms, s0), al = (se > 0.0f) ? exp2(ms - nm) : 0.0f, w0 = exp2(s0 - nm);
        se = se * al + w0;
        o0 = o0 * al + w0 * float4(*((const device half4*)(v_cache + ko)));
        ms = nm;
    }
    threadgroup float sm_max[8], sm_sum[8]; threadgroup float4 sm_o0[8][32];
    if (lane == 0) { sm_max[sg] = ms; sm_sum[sg] = se; }
    sm_o0[sg][lane] = o0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float gm = sm_max[0]; for (int s = 1; s < 8; ++s) gm = max(gm, sm_max[s]);
        float gs = 0.0f; float4 go0 = float4(0.0f);
        for (int s = 0; s < 8; ++s) if (sm_sum[s] > 0.0f) {
            const float ws = exp2(sm_max[s] - gm); gs += sm_sum[s] * ws; go0 += sm_o0[s][lane] * ws;
        }
        *((device half4*)(attn_out + q_base)) = half4(go0 / max(gs, 1e-12f));
    }
}

kernel void gqa_attention_256(
    device const half* q_buf [[buffer(0)]], device const half* k_cache [[buffer(1)]], device const half* v_cache [[buffer(2)]],
    device half* attn_out [[buffer(3)]], constant TrGQAParams& p [[buffer(4)]],
    uint2 tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int h = int(tg_id.x), m = int(tg_id.y);
    if (h >= p.n_heads || m >= p.M) return;
    const int kv_h = h / (p.n_heads / p.n_kv_heads), seq_len = p.pos + m + 1;
    const float scale_log2e = (1.0f / 16.0f) * LOG2E;
    const int q_base = (m * p.n_heads + h) * 256 + int(lane) * 8;
    const ulong kv_head_base = ulong(kv_h) * ulong(p.max_seq) * 256UL;
    const int blk = (seq_len + 7) >> 3, t0 = int(sg) * blk, t1 = min(t0 + blk, seq_len);

    const float4 q0 = float4(*((const device half4*)(q_buf + q_base)))     * scale_log2e;
    const float4 q1 = float4(*((const device half4*)(q_buf + q_base + 4))) * scale_log2e;
    float4 o0 = float4(0.0f), o1 = float4(0.0f); float ms = -1e30f, se = 0.0f;
    for (int t = t0; t < t1; ++t) {
        const ulong ko = kv_head_base + ulong(t) * 256UL + ulong(lane * 8);
        const float s0 = simd_sum(dot(q0, float4(*((const device half4*)(k_cache + ko))))
                                + dot(q1, float4(*((const device half4*)(k_cache + ko + 4)))));
        const float nm = max(ms, s0), al = (se > 0.0f) ? exp2(ms - nm) : 0.0f, w0 = exp2(s0 - nm);
        se = se * al + w0;
        o0 = o0 * al + w0 * float4(*((const device half4*)(v_cache + ko)));
        o1 = o1 * al + w0 * float4(*((const device half4*)(v_cache + ko + 4)));
        ms = nm;
    }
    threadgroup float sm_max[8], sm_sum[8]; threadgroup float4 sm_o0[8][32], sm_o1[8][32];
    if (lane == 0) { sm_max[sg] = ms; sm_sum[sg] = se; }
    sm_o0[sg][lane] = o0; sm_o1[sg][lane] = o1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float gm = sm_max[0]; for (int s = 1; s < 8; ++s) gm = max(gm, sm_max[s]);
        float gs = 0.0f; float4 go0 = float4(0.0f), go1 = float4(0.0f);
        for (int s = 0; s < 8; ++s) if (sm_sum[s] > 0.0f) {
            const float ws = exp2(sm_max[s] - gm); gs += sm_sum[s] * ws; go0 += sm_o0[s][lane] * ws; go1 += sm_o1[s][lane] * ws;
        }
        const float inv = 1.0f / max(gs, 1e-12f);
        *((device half4*)(attn_out + q_base))     = half4(go0 * inv);
        *((device half4*)(attn_out + q_base + 4)) = half4(go1 * inv);
    }
}

// ─────────────────────── 7. Repetition penalty PER BARIS + Argmax + Top-64 exact ───────────────────────
// Baris m memverifikasi token SETELAH (riwayat + draft 1..m), jadi himpunan penalty-nya harus ikut
// memuat draft itu — persis seperti decode berurutan. rows[m][0] = jumlah slot, rows[m][1..64] = token.
struct RepRowsParams { int M, K_len, stride; float penalty; int rows[8 * 65]; };

kernel void apply_rep_penalty_rows(
    device half* logits [[buffer(0)]], constant RepRowsParams& rp [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int i = int(gid.x), m = int(gid.y);
    if (m >= rp.M) return;
    const int base = m * rp.stride;
    if (i >= rp.rows[base]) return;
    const ulong idx = ulong(m) * ulong(rp.K_len) + ulong(rp.rows[base + 1 + i]);
    const float v = float(logits[idx]);
    logits[idx] = half((v > 0.0f) ? (v / rp.penalty) : (v * rp.penalty));
}

kernel void argmax_full(
    device const half* logits [[buffer(0)]], device int* out_tok [[buffer(1)]], constant int& V [[buffer(2)]],
    uint tg_x [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int m = int(tg_x);
    const device half* row = logits + ulong(m) * ulong(V);
    float b1 = -1e30f; int idx1 = 0;
    for (int i = int(tid); i < V; i += 256) {
        float v = float(row[i]);
        if (v > b1) { b1 = v; idx1 = i; }
    }
    threadgroup float sm_b1[8]; threadgroup int sm_id[8];
    float sg_max = simd_max(b1);
    int win_lane = simd_min((b1 == sg_max) ? int(lane) : 32);
    int win_id = simd_broadcast(idx1, ushort(win_lane));
    if (lane == 0) { sm_b1[sg] = sg_max; sm_id[sg] = win_id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float fb = -1e30f; int fid = 0;
        for (int s = 0; s < 8; ++s) if (sm_b1[s] > fb) { fb = sm_b1[s]; fid = sm_id[s]; }
        out_tok[m] = fid;
    }
}

// Top-64 tersortir yang DIJAMIN exact:
//  1) tiap thread menyimpan top-8 dari bagiannya + nilai terbesar yang ia buang (ev_max);
//  2) tiap simdgroup memilih top-64 dari 256 kandidatnya; 3) simdgroup 0 memilih top-64 dari 512;
//  4) bila ADA thread yang membuang nilai > nilai ke-64 hasil akhir, hasilnya mungkin tidak lengkap →
//     out_flag[m] = 1 dan host menghitung ulang top-64 dari logits di CPU. (Versi lama hanya top-2 per
//     thread dan diam-diam kehilangan token ±40% langkah.)
inline void sg_select_top64(thread float* lv, thread int* li, int nreg,
                            threadgroup float* dst_v, threadgroup int* dst_i, uint lane)
{
    for (int k = 0; k < 64; ++k) {
        float best_v = lv[0]; int best_s = 0;
        for (int j = 1; j < nreg; ++j) if (lv[j] > best_v) { best_v = lv[j]; best_s = j; }
        const float gv = simd_max(best_v);
        const int win_lane = simd_min((best_v == gv) ? int(lane) : 32);
        const int win_id   = simd_broadcast(li[best_s], ushort(win_lane));
        if (int(lane) == win_lane) {
            for (int j = 0; j < nreg; ++j) if (j == best_s) lv[j] = -1e30f;
        }
        if (lane == 0) { dst_v[k] = gv; dst_i[k] = win_id; }
    }
}

kernel void topk_logits_64(
    device const half* logits   [[buffer(0)]],
    device int*        out_ids  [[buffer(1)]],   // [M, 64]
    device float*      out_vals [[buffer(2)]],   // [M, 64]
    constant int&      V        [[buffer(3)]],
    device int*        out_flag [[buffer(4)]],   // [M]: 1 = hitung ulang di CPU
    uint tg_x [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int m = int(tg_x);
    const device half* row = logits + ulong(m) * ulong(V);

    float bv[8]; int bi[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) { bv[j] = -1e30f; bi[j] = 0; }
    float ev_max = -1e30f;
    for (int i = int(tid); i < V; i += 256) {
        const float v = float(row[i]);
        if (v > bv[7]) {
            ev_max = max(ev_max, bv[7]);
            int j = 7;
            while (j > 0 && bv[j - 1] < v) { bv[j] = bv[j - 1]; bi[j] = bi[j - 1]; --j; }
            bv[j] = v; bi[j] = i;
        } else {
            ev_max = max(ev_max, v);
        }
    }

    threadgroup float s_v[512]; threadgroup int s_i[512];
    threadgroup float s_ev[8];
    threadgroup float f_v[64]; threadgroup int f_i[64];   // threadgroup hanya boleh di scope kernel
    sg_select_top64(bv, bi, 8, s_v + int(sg) * 64, s_i + int(sg) * 64, lane);
    const float sg_ev = simd_max(ev_max);
    if (lane == 0) s_ev[sg] = sg_ev;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float lv[16]; int li[16];
        #pragma unroll
        for (int j = 0; j < 16; ++j) { lv[j] = s_v[int(lane) + j * 32]; li[j] = s_i[int(lane) + j * 32]; }
        sg_select_top64(lv, li, 16, f_v, f_i, lane);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float ev_all = -1e30f;
        for (int s = 0; s < 8; ++s) ev_all = max(ev_all, s_ev[s]);
        if (lane == 0) out_flag[m] = (ev_all > f_v[63]) ? 1 : 0;
        for (int k = int(lane); k < 64; k += 32) {
            out_ids[m * 64 + k]  = f_i[k];
            out_vals[m * 64 + k] = f_v[k];
        }
    }
}
