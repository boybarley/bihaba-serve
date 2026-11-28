// ============================================================================
// bihaba-serve — Native Apple Metal Compute Kernels (Qwen3.5 Dense & MoE)
// Powered by Taalas-Metal Engine | Created by Boy Barley from Bihaba (Team Pasai Aceh)
//
// Highlights:
//   • Native FP16 & BF16 affine quantized matvec/matmul (2b / 3b / 4b / 8b)
//   • BM=8, BN=32, BK=64 Compact-Half Hardware MMA (simdgroup_half8x8)
//   • Zero-Copy Speculative Chunked Gated DeltaNet (M=1..8) + Delayed Commit
//   • Gated Full Attention (hd=256) with Partial RoPE (Standard & YaRN)
//   • Register-Only 32-Lane SIMD Top-K + 3D Contiguous-Expert 256-Expert MoE
//   • Exact 256-Thread Top-2 -> 512 -> 1-SG SIMD Top-64 Logit Extractor
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant bool ADD_RES_C [[function_constant(0)]];
constant bool IS_BF16_C [[function_constant(1)]];   // true for Direct Zero-Copy .safetensors (BF16 scales/biases), false for .taalas2 (FP16)

constant float LOG2E = 1.4426950408889634f;

struct NormParams       { int dim; float eps; };
struct QMVParams        { int K, N, gs_shift; };
struct QMVMultiParams   { int M, K, N, gs_shift; };
struct Conv1dParams     { int M, C, k_size; };
struct DeltaNetParams   { int M, n_kh, n_vh, k_hd, v_hd; float eps; };
struct QKNormRopeParams { int M, n_heads, n_kv_heads, head_dim, rope_dim, pos, max_seq; float eps; };
struct GQA256Params     { int M, n_heads, n_kv_heads, pos, max_seq; };
struct TopKParams       { int M, E, top_k; };
struct MoEProjParams    { int M, K, N, top_k, gs_shift; ulong stride_w, stride_s; };
struct DeltaNetInParams { int M, K, gs_shift, n_qkv, n_z, n_a, n_b; };
struct FullQKVParams    { int M, K, gs_shift, n_q, n_k, n_v; };

// ─────────────────────── 0. Zero-Overhead Native FP16 / BF16 Scale-Bias Loader ───────────────────────
inline half load_sb_h(device const ushort* ptr, ulong idx) {
    const ushort u = ptr[idx];
    return IS_BF16_C ? half(as_type<float>(uint(u) << 16)) : as_type<half>(u);
}

inline float load_sb_f(device const ushort* ptr, ulong idx) {
    const ushort u = ptr[idx];
    return IS_BF16_C ? as_type<float>(uint(u) << 16) : float(as_type<half>(u));
}

// ─────────────────────── 1. Multi-Token Embedding Lookup (M=1..8, 2b/3b/4b/8b) ───────────────────────
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
    const int d = int(gid.x);
    const int m = int(gid.y);
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

// ─────────────────────── 2. Multi-Token Parallel RMSNorm (grid=(M,1,1)) ───────────────────────
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

// ─────────────────────── 3. Quantized Word Dot Helpers (2b, 3b, 4b, 8b) ───────────────────────
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

// ─────────────────────── 4A. Fast M=1 Dual-Row QMV (4-Byte Aligned Zero-Copy Safe!) ───────────────────────
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
        const uint32_t w0a = w[wo0 + wi], w0b = w[wo0 + wi + 1];
        const uint32_t w1a = w[wo1 + wi], w1b = w[wo1 + wi + 1];
        const half2 hs0 = half2(load_sb_h(scales, so0 + g)), hb0 = half2(load_sb_h(biases, so0 + g));
        const half2 hs1 = half2(load_sb_h(scales, so1 + g)), hb1 = half2(load_sb_h(biases, so1 + g));
        const device half4* xp = (const device half4*)(x + gk);
        a0 += dot_4bit_word(w0a, hs0, hb0, xp) + dot_4bit_word(w0b, hs0, hb0, xp + 2);
        a1 += dot_4bit_word(w1a, hs1, hb1, xp) + dot_4bit_word(w1b, hs1, hb1, xp + 2);
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

// ─────────────────────── 4B. Multi-Token Weight-Reuse QMV (M=2..8, 4-Byte Safe!) ───────────────────────
kernel void qmv_4b_m(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVMultiParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift;
    const int row = int(tg_x) * 4 + int(sg);
    if (row >= N) return;
    const ulong kw = ulong(K >> 3), kg = ulong(K >> gsh);
    const ulong wo = ulong(row) * kw, so = ulong(row) * kg;
    const half2 c1024 = half2(1024.0h);

    float acc[8] = {0,0,0,0,0,0,0,0};
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
    for (int m = 0; m < M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) {
            if (ADD_RES_C) out[m * N + row] += half(v);
            else           out[m * N + row]  = half(v);
        }
    }
}

kernel void qmv_2b_m(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVMultiParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift;
    const int row = int(tg_x) * 4 + int(sg);
    if (row >= N) return;
    const ulong kw = ulong(K >> 4), kg = ulong(K >> gsh);
    const ulong wo = ulong(row) * kw, so = ulong(row) * kg;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const uint32_t wv = w[wo + (gk >> 4)];
        const int g = gk >> gsh;
        const half2 hs = half2(load_sb_h(scales, so + g)), hb = half2(load_sb_h(biases, so + g));
        for (int m = 0; m < M; ++m)
            acc[m] += dot_2bit_word(wv, hs, hb, (const device half4*)(x + m * K + gk));
    }
    for (int m = 0; m < M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) {
            if (ADD_RES_C) out[m * N + row] += half(v);
            else           out[m * N + row]  = half(v);
        }
    }
}

kernel void qmv_3b_m(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVMultiParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift;
    const int row = int(tg_x) * 4 + int(sg);
    if (row >= N) return;
    const ulong kw = ulong((K * 3) >> 5), kg = ulong(K >> gsh);
    const ulong wo = ulong(row) * kw, so = ulong(row) * kg;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 1024) {
        const int gk = k0 + int(lane) * 32;
        if (gk >= K) break;
        const int blk = (gk >> 5) * 3, g = gk >> gsh;
        const uint32_t w0 = w[wo+blk], w1 = w[wo+blk+1], w2 = w[wo+blk+2];
        const float s = load_sb_f(scales, so+g), b = load_sb_f(biases, so+g);
        for (int m = 0; m < M; ++m)
            acc[m] += dot_3bit_block32(w0, w1, w2, s, b, (const device half4*)(x + m * K + gk));
    }
    for (int m = 0; m < M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) {
            if (ADD_RES_C) out[m * N + row] += half(v);
            else           out[m * N + row]  = half(v);
        }
    }
}

kernel void qmv_8b_m(
    device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]],
    device const ushort* scales [[buffer(2)]], device const ushort* biases [[buffer(3)]],
    device half* out [[buffer(4)]], constant QMVMultiParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift;
    const int row = int(tg_x) * 4 + int(sg);
    if (row >= N) return;
    const ulong kw = ulong(K >> 2), kg = ulong(K >> gsh);
    const ulong wo = ulong(row) * kw, so = ulong(row) * kg;
    float acc[8] = {0,0,0,0,0,0,0,0};
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
    for (int m = 0; m < M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) {
            if (ADD_RES_C) out[m * N + row] += half(v);
            else           out[m * N + row]  = half(v);
        }
    }
}

// ─────────────────────── 4C. Fused DeltaNet QKVZAB & FullAttn QKV (M=2..3) ───────────────────────
kernel void deltanet_in_proj_4b(
    device const half* x [[buffer(0)]],
    device const uint32_t* w_qkv [[buffer(1)]], device const ushort* s_qkv [[buffer(2)]], device const ushort* b_qkv [[buffer(3)]],
    device const uint32_t* w_z   [[buffer(4)]], device const ushort* s_z   [[buffer(5)]], device const ushort* b_z   [[buffer(6)]],
    device const uint32_t* w_a   [[buffer(7)]], device const ushort* s_a   [[buffer(8)]], device const ushort* b_a   [[buffer(9)]],
    device const uint32_t* w_b   [[buffer(10)]],device const ushort* s_b   [[buffer(11)]],device const ushort* b_b   [[buffer(12)]],
    device half* o_qkv [[buffer(13)]], device half* o_z [[buffer(14)]], device half* o_a [[buffer(15)]], device half* o_b [[buffer(16)]],
    constant DeltaNetInParams& p [[buffer(17)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int M = p.M, c0 = p.n_qkv, c1 = c0 + p.n_z, c2 = c1 + p.n_a, c3 = c2 + p.n_b;
    const int g_row = int(tg_x) * 4 + int(sg);
    if (g_row >= c3) return;
    device const uint32_t* wp; device const ushort *sp, *bp; device half* op;
    int lr, stride;
    if (g_row < c0)      { wp = w_qkv; sp = s_qkv; bp = b_qkv; op = o_qkv; lr = g_row;      stride = p.n_qkv; }
    else if (g_row < c1) { wp = w_z;   sp = s_z;   bp = b_z;   op = o_z;   lr = g_row - c0; stride = p.n_z;   }
    else if (g_row < c2) { wp = w_a;   sp = s_a;   bp = b_a;   op = o_a;   lr = g_row - c1; stride = p.n_a;   }
    else                 { wp = w_b;   sp = s_b;   bp = b_b;   op = o_b;   lr = g_row - c2; stride = p.n_b;   }

    const int K = p.K, gsh = p.gs_shift;
    const ulong kw = ulong(K >> 3), kg = ulong(K >> gsh);
    const ulong wo = ulong(lr) * kw, so = ulong(lr) * kg;
    const half2 c1024 = half2(1024.0h);

    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const int wi = gk >> 3, g = gk >> gsh;
        const uint32_t wvx = wp[wo + wi], wvy = wp[wo + wi + 1];
        const half2 hs = half2(load_sb_h(sp, so + g)), hb = half2(load_sb_h(bp, so + g));
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
    for (int m = 0; m < M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) op[m * stride + lr] = half(v);
    }
}

kernel void q35_qkv_proj_4b(
    device const half* x [[buffer(0)]],
    device const uint32_t* w_q [[buffer(1)]], device const ushort* s_q [[buffer(2)]], device const ushort* b_q [[buffer(3)]],
    device const uint32_t* w_k [[buffer(4)]], device const ushort* s_k [[buffer(5)]], device const ushort* b_k [[buffer(6)]],
    device const uint32_t* w_v [[buffer(7)]], device const ushort* s_v [[buffer(8)]], device const ushort* b_v [[buffer(9)]],
    device half* o_q [[buffer(10)]], device half* o_k [[buffer(11)]], device half* o_v [[buffer(12)]],
    constant FullQKVParams& p [[buffer(13)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int M = p.M, c0 = p.n_q, c1 = c0 + p.n_k, c2 = c1 + p.n_v;
    const int g_row = int(tg_x) * 4 + int(sg);
    if (g_row >= c2) return;
    device const uint32_t* wp; device const ushort *sp, *bp; device half* op;
    int lr, stride;
    if (g_row < c0)      { wp = w_q; sp = s_q; bp = b_q; op = o_q; lr = g_row;      stride = p.n_q; }
    else if (g_row < c1) { wp = w_k; sp = s_k; bp = b_k; op = o_k; lr = g_row - c0; stride = p.n_k; }
    else                 { wp = w_v; sp = s_v; bp = b_v; op = o_v; lr = g_row - c1; stride = p.n_v; }

    const int K = p.K, gsh = p.gs_shift;
    const ulong kw = ulong(K >> 3), kg = ulong(K >> gsh);
    const ulong wo = ulong(lr) * kw, so = ulong(lr) * kg;
    const half2 c1024 = half2(1024.0h);

    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const int wi = gk >> 3, g = gk >> gsh;
        const uint32_t wvx = wp[wo + wi], wvy = wp[wo + wi + 1];
        const half2 hs = half2(load_sb_h(sp, so + g)), hb = half2(load_sb_h(bp, so + g));
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
    for (int m = 0; m < M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) op[m * stride + lr] = half(v);
    }
}

// ─────────────────────── 4D. Multi-Token Fused Single-Pass Gate+Up SwiGLU (4-Byte Safe!) ───────────────────────
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
        const ulong kw = ulong(K >> 3), wo = ulong(row) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const uint32_t ga = wg[wo+wi], gb = wg[wo+wi+1], ua = wu[wo+wi], ub = wu[wo+wi+1];
            const half2 hsg = half2(load_sb_h(sg_a, so + g)), hbg = half2(load_sb_h(bg_a, so + g));
            const half2 hsu = half2(load_sb_h(su_a, so + g)), hbu = half2(load_sb_h(bu_a, so + g));
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                ag[m] += dot_4bit_word(ga, hsg, hbg, xp) + dot_4bit_word(gb, hsg, hbg, xp + 2);
                au[m] += dot_4bit_word(ua, hsu, hbu, xp) + dot_4bit_word(ub, hsu, hbu, xp + 2);
            }
        }
    } else if (BITS == 2) {
        const ulong kw = ulong(K >> 4), wo = ulong(row) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int kwi = gk >> 4, g = gk >> gsh;
            const uint32_t wgv = wg[wo + kwi], wuv = wu[wo + kwi];
            const half2 hsg = half2(load_sb_h(sg_a, so + g)), hbg = half2(load_sb_h(bg_a, so + g));
            const half2 hsu = half2(load_sb_h(su_a, so + g)), hbu = half2(load_sb_h(bu_a, so + g));
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                ag[m] += dot_2bit_word(wgv, hsg, hbg, xp);
                au[m] += dot_2bit_word(wuv, hsu, hbu, xp);
            }
        }
    } else if (BITS == 3) {
        const ulong kw = ulong((K * 3) >> 5), wo = ulong(row) * kw;
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
    } else {
        const ulong kw = ulong(K >> 2), wo = ulong(row) * kw;
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
DECL_SWIG_M(dense_swiglu_8b, 8)

// ─── 1.7.0.dev6: SwiGLU decode M=1 dengan akumulator skalar (array ag[8]/au[8] ber-indeks runtime
// di kernel generik membuat gate+up 9B hanya ~204 GB/s vs MLX ~270 GB/s). Loop & ekspresi sama. ───
template <int BITS>
inline void dense_swiglu_m1_impl(
    device const half* x,
    device const uint32_t* wg, device const ushort* sg_a, device const ushort* bg_a,
    device const uint32_t* wu, device const ushort* su_a, device const ushort* bu_a,
    device half* out, constant QMVMultiParams& p, uint tg_x, uint sg, uint lane)
{
    const int K = p.K, N = p.N, gsh = p.gs_shift;
    const int row = int(tg_x) * 4 + int(sg);
    if (row >= N) return;
    const ulong kg = ulong(K >> gsh), so = ulong(row) * kg;
    float ag0 = 0.0f, au0 = 0.0f;

    if (BITS == 4) {
        const ulong kw = ulong(K >> 3), wo = ulong(row) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const uint32_t ga = wg[wo+wi], gb = wg[wo+wi+1], ua = wu[wo+wi], ub = wu[wo+wi+1];
            const half2 hsg = half2(load_sb_h(sg_a, so + g)), hbg = half2(load_sb_h(bg_a, so + g));
            const half2 hsu = half2(load_sb_h(su_a, so + g)), hbu = half2(load_sb_h(bu_a, so + g));
            {
                const device half4* xp = (const device half4*)(x + gk);
                ag0 += dot_4bit_word(ga, hsg, hbg, xp) + dot_4bit_word(gb, hsg, hbg, xp + 2);
                au0 += dot_4bit_word(ua, hsu, hbu, xp) + dot_4bit_word(ub, hsu, hbu, xp + 2);
            }
        }
    } else if (BITS == 2) {
        const ulong kw = ulong(K >> 4), wo = ulong(row) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int kwi = gk >> 4, g = gk >> gsh;
            const uint32_t wgv = wg[wo + kwi], wuv = wu[wo + kwi];
            const half2 hsg = half2(load_sb_h(sg_a, so + g)), hbg = half2(load_sb_h(bg_a, so + g));
            const half2 hsu = half2(load_sb_h(su_a, so + g)), hbu = half2(load_sb_h(bu_a, so + g));
            {
                const device half4* xp = (const device half4*)(x + gk);
                ag0 += dot_2bit_word(wgv, hsg, hbg, xp);
                au0 += dot_2bit_word(wuv, hsu, hbu, xp);
            }
        }
    } else if (BITS == 3) {
        const ulong kw = ulong((K * 3) >> 5), wo = ulong(row) * kw;
        for (int k0 = 0; k0 < K; k0 += 1024) {
            const int gk = k0 + int(lane) * 32;
            if (gk >= K) break;
            const int blk = (gk >> 5) * 3, g = gk >> gsh;
            const uint32_t g0 = wg[wo+blk], g1 = wg[wo+blk+1], g2 = wg[wo+blk+2];
            const uint32_t u0 = wu[wo+blk], u1 = wu[wo+blk+1], u2 = wu[wo+blk+2];
            const float sgv = load_sb_f(sg_a, so+g), bgv = load_sb_f(bg_a, so+g);
            const float suv = load_sb_f(su_a, so+g), buv = load_sb_f(bu_a, so+g);
            {
                const device half4* xp = (const device half4*)(x + gk);
                ag0 += dot_3bit_block32(g0, g1, g2, sgv, bgv, xp);
                au0 += dot_3bit_block32(u0, u1, u2, suv, buv, xp);
            }
        }
    } else {
        const ulong kw = ulong(K >> 2), wo = ulong(row) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 2, g = gk >> gsh;
            const uint32_t g0 = wg[wo+wi], g1 = wg[wo+wi+1], g2 = wg[wo+wi+2], g3 = wg[wo+wi+3];
            const uint32_t u0 = wu[wo+wi], u1 = wu[wo+wi+1], u2 = wu[wo+wi+2], u3 = wu[wo+wi+3];
            const float sgv = load_sb_f(sg_a, so+g), bgv = load_sb_f(bg_a, so+g);
            const float suv = load_sb_f(su_a, so+g), buv = load_sb_f(bu_a, so+g);
            {
                const device half4* xp = (const device half4*)(x + gk);
                ag0 += dot_8bit_word(g0, sgv, bgv, xp[0]) + dot_8bit_word(g1, sgv, bgv, xp[1])
                       + dot_8bit_word(g2, sgv, bgv, xp[2]) + dot_8bit_word(g3, sgv, bgv, xp[3]);
                au0 += dot_8bit_word(u0, suv, buv, xp[0]) + dot_8bit_word(u1, suv, buv, xp[1])
                       + dot_8bit_word(u2, suv, buv, xp[2]) + dot_8bit_word(u3, suv, buv, xp[3]);
            }
        }
    }
    {
        const float gv = simd_sum(ag0), uv = simd_sum(au0);
        if (lane == 0) out[row] = half((gv / (1.0f + exp(-gv))) * uv);
    }
}

#define DECL_SWIG_M1(NAME, BITS) \
kernel void NAME(device const half* x [[buffer(0)]], \
    device const uint32_t* wg [[buffer(1)]], device const ushort* sg_a [[buffer(2)]], device const ushort* bg_a [[buffer(3)]], \
    device const uint32_t* wu [[buffer(4)]], device const ushort* su_a [[buffer(5)]], device const ushort* bu_a [[buffer(6)]], \
    device half* out [[buffer(7)]], constant QMVMultiParams& p [[buffer(8)]], \
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ dense_swiglu_m1_impl<BITS>(x, wg, sg_a, bg_a, wu, su_a, bu_a, out, p, tg_x, sg, lane); }

DECL_SWIG_M1(dense_swiglu_2b_m1, 2)
DECL_SWIG_M1(dense_swiglu_3b_m1, 3)
DECL_SWIG_M1(dense_swiglu_4b_m1, 4)
DECL_SWIG_M1(dense_swiglu_8b_m1, 8)

// ─────────────────────── 4E. Fast M=1 Dual-Row Fused DeltaNet-In & Full-QKV (4-Byte Safe!) ───────────────────────
kernel void deltanet_in_proj_4b_m1(
    device const half*     x      [[buffer(0)]],
    device const uint32_t* w_qkv  [[buffer(1)]],  device const ushort* s_qkv [[buffer(2)]],  device const ushort* b_qkv [[buffer(3)]],
    device const uint32_t* w_z    [[buffer(4)]],  device const ushort* s_z   [[buffer(5)]],  device const ushort* b_z   [[buffer(6)]],
    device const uint32_t* w_a    [[buffer(7)]],  device const ushort* s_a   [[buffer(8)]],  device const ushort* b_a   [[buffer(9)]],
    device const uint32_t* w_b    [[buffer(10)]], device const ushort* s_b   [[buffer(11)]], device const ushort* b_b   [[buffer(12)]],
    device half*           o_qkv  [[buffer(13)]],
    device half*           o_z    [[buffer(14)]],
    device half*           o_a    [[buffer(15)]],
    device half*           o_b    [[buffer(16)]],
    constant DeltaNetInParams& p  [[buffer(17)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int g_row = (int(tg_x) * 4 + int(sg)) * 2;
    const int c0 = p.n_qkv, c1 = c0 + p.n_z, c2 = c1 + p.n_a, c3 = c2 + p.n_b;
    if (g_row >= c3) return;
    device const uint32_t* wp; device const ushort *sp, *bp; device half* op;
    int lr0, stride;
    if (g_row < c0)      { wp = w_qkv; sp = s_qkv; bp = b_qkv; op = o_qkv; lr0 = g_row;      stride = p.n_qkv; }
    else if (g_row < c1) { wp = w_z;   sp = s_z;   bp = b_z;   op = o_z;   lr0 = g_row - c0; stride = p.n_z;   }
    else if (g_row < c2) { wp = w_a;   sp = s_a;   bp = b_a;   op = o_a;   lr0 = g_row - c1; stride = p.n_a;   }
    else                 { wp = w_b;   sp = s_b;   bp = b_b;   op = o_b;   lr0 = g_row - c2; stride = p.n_b;   }

    const int K = p.K, gsh = p.gs_shift;
    const int r0 = lr0, r1 = min(lr0 + 1, stride - 1);
    const ulong kw = ulong(K >> 3), kg = ulong(K >> gsh);
    const ulong wo0 = ulong(r0)*kw, wo1 = ulong(r1)*kw;
    const ulong so0 = ulong(r0)*kg, so1 = ulong(r1)*kg;

    float a0 = 0.0f, a1 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const int wi = gk >> 3, g = gk >> gsh;
        const uint32_t w0a = wp[wo0+wi], w0b = wp[wo0+wi+1];
        const uint32_t w1a = wp[wo1+wi], w1b = wp[wo1+wi+1];
        const half2 hs0 = half2(load_sb_h(sp, so0 + g)), hb0 = half2(load_sb_h(bp, so0 + g));
        const half2 hs1 = half2(load_sb_h(sp, so1 + g)), hb1 = half2(load_sb_h(bp, so1 + g));
        const device half4* xp = (const device half4*)(x + gk);
        a0 += dot_4bit_word(w0a, hs0, hb0, xp) + dot_4bit_word(w0b, hs0, hb0, xp + 2);
        a1 += dot_4bit_word(w1a, hs1, hb1, xp) + dot_4bit_word(w1b, hs1, hb1, xp + 2);
    }
    const float v0 = simd_sum(a0), v1 = simd_sum(a1);
    if (lane == 0) { op[r0] = half(v0); if (lr0 + 1 < stride) op[r1] = half(v1); }
}

kernel void q35_qkv_proj_4b_m1(
    device const half*     x   [[buffer(0)]],
    device const uint32_t* w_q [[buffer(1)]], device const ushort* s_q [[buffer(2)]], device const ushort* b_q [[buffer(3)]],
    device const uint32_t* w_k [[buffer(4)]], device const ushort* s_k [[buffer(5)]], device const ushort* b_k [[buffer(6)]],
    device const uint32_t* w_v [[buffer(7)]], device const ushort* s_v [[buffer(8)]], device const ushort* b_v [[buffer(9)]],
    device half* o_q [[buffer(10)]], device half* o_k [[buffer(11)]], device half* o_v [[buffer(12)]],
    constant FullQKVParams& p [[buffer(13)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int g_row = (int(tg_x) * 4 + int(sg)) * 2;
    const int c0 = p.n_q, c1 = c0 + p.n_k, c2 = c1 + p.n_v;
    if (g_row >= c2) return;
    device const uint32_t* wp; device const ushort *sp, *bp; device half* op;
    int lr0, stride;
    if (g_row < c0)      { wp = w_q; sp = s_q; bp = b_q; op = o_q; lr0 = g_row;      stride = p.n_q; }
    else if (g_row < c1) { wp = w_k; sp = s_k; bp = b_k; op = o_k; lr0 = g_row - c0; stride = p.n_k; }
    else                 { wp = w_v; sp = s_v; bp = b_v; op = o_v; lr0 = g_row - c1; stride = p.n_v; }

    const int K = p.K, gsh = p.gs_shift;
    const int r0 = lr0, r1 = min(lr0 + 1, stride - 1);
    const ulong kw = ulong(K >> 3), kg = ulong(K >> gsh);
    const ulong wo0 = ulong(r0)*kw, wo1 = ulong(r1)*kw;
    const ulong so0 = ulong(r0)*kg, so1 = ulong(r1)*kg;

    float a0 = 0.0f, a1 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 512) {
        const int gk = k0 + int(lane) * 16;
        if (gk >= K) break;
        const int wi = gk >> 3, g = gk >> gsh;
        const uint32_t w0a = wp[wo0+wi], w0b = wp[wo0+wi+1];
        const uint32_t w1a = wp[wo1+wi], w1b = wp[wo1+wi+1];
        const half2 hs0 = half2(load_sb_h(sp, so0 + g)), hb0 = half2(load_sb_h(bp, so0 + g));
        const half2 hs1 = half2(load_sb_h(sp, so1 + g)), hb1 = half2(load_sb_h(bp, so1 + g));
        const device half4* xp = (const device half4*)(x + gk);
        a0 += dot_4bit_word(w0a, hs0, hb0, xp) + dot_4bit_word(w0b, hs0, hb0, xp + 2);
        a1 += dot_4bit_word(w1a, hs1, hb1, xp) + dot_4bit_word(w1b, hs1, hb1, xp + 2);
    }
    const float v0 = simd_sum(a0), v1 = simd_sum(a1);
    if (lane == 0) { op[r0] = half(v0); if (lr0 + 1 < stride) op[r1] = half(v1); }
}

// ─────────────────────── 4F. BM=8·MT, BN=32, BK=64 Compact-Half Hardware MMA ───────────────────────
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

// Akumulator MMA sebagai variabel bernama (c0..c3), BUKAN array simdgroup_matrix.
// Versi array (c[MT]) + loop dengan break dinamis membuat compiler Metal menaruh akumulator di memori:
// terukur 1.5–1.6× lebih lambat di M=8 (Qwythos-9B FFN 32→50 ms). MT adalah konstanta template, jadi
// cabang `MT > 1` hilang total pada varian mma8.
#define MMA_ACC_TILE(XS, WSG, C0, C1, C2, C3) \
    _Pragma("unroll") \
    for (int kk = 0; kk < 64; kk += 8) { \
        simdgroup_half8x8 bt, a; \
        simdgroup_load(bt, (WSG) + kk, 72, ulong2(0, 0), true); \
        simdgroup_load(a, (XS) + kk, 64); \
        simdgroup_multiply_accumulate(C0, a, bt, C0); \
        if (MT > 1) { \
            if (M > 8)  { simdgroup_load(a, (XS) + 8 * 64 + kk, 64);  simdgroup_multiply_accumulate(C1, a, bt, C1); } \
            if (M > 16) { simdgroup_load(a, (XS) + 16 * 64 + kk, 64); simdgroup_multiply_accumulate(C2, a, bt, C2); } \
            if (M > 24) { simdgroup_load(a, (XS) + 24 * 64 + kk, 64); simdgroup_multiply_accumulate(C3, a, bt, C3); } \
        } \
    }

inline void qmm_store_tile(simdgroup_float8x8 c, int m0, threadgroup float* cbuf, device half* out,
                           int M, int N, int n_base, uint lane) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(c, cbuf, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = 0; q < 2; ++q) {
        const int idx = int(lane) * 2 + q, m = m0 + (idx >> 3), col = n_base + (idx & 7);
        if (m < M && col < N) {
            if (ADD_RES_C) out[m * N + col] += half(cbuf[idx]);
            else           out[m * N + col]  = half(cbuf[idx]);
        }
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

    simdgroup_float8x8 c0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), c1 = c0, c2 = c0, c3 = c0;
    const threadgroup half* w_sg = Ws + (int(sg) * 8) * WS;
    for (int k0 = 0; k0 < K; k0 += BK) {
        load_x_tile<MT>(x, M, K, k0, Xs, tid);
        load_w_tile<BITS>(w, scales, biases, r, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        MMA_ACC_TILE(Xs, w_sg, c0, c1, c2, c3)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* cbuf = Cs + int(sg) * 64;
    const int n_base = row_base + int(sg) * 8;
    qmm_store_tile(c0, 0, cbuf, out, M, N, n_base, lane);
    if (MT > 1) {
        if (M > 8)  qmm_store_tile(c1, 8,  cbuf, out, M, N, n_base, lane);
        if (M > 16) qmm_store_tile(c2, 16, cbuf, out, M, N, n_base, lane);
        if (M > 24) qmm_store_tile(c3, 24, cbuf, out, M, N, n_base, lane);
    }
}

#define DECL_QMM(NAME, BITS, MT) \
[[max_total_threads_per_threadgroup(128)]] \
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

#define DECL_DN_IN_MMA(NAME, MT) \
[[max_total_threads_per_threadgroup(128)]] \
kernel void NAME( \
    device const half* x [[buffer(0)]], \
    device const uint32_t* w_qkv [[buffer(1)]], device const ushort* s_qkv [[buffer(2)]], device const ushort* b_qkv [[buffer(3)]], \
    device const uint32_t* w_z   [[buffer(4)]], device const ushort* s_z   [[buffer(5)]], device const ushort* b_z   [[buffer(6)]], \
    device const uint32_t* w_a   [[buffer(7)]], device const ushort* s_a   [[buffer(8)]], device const ushort* b_a   [[buffer(9)]], \
    device const uint32_t* w_b   [[buffer(10)]],device const ushort* s_b   [[buffer(11)]],device const ushort* b_b   [[buffer(12)]], \
    device half* o_qkv [[buffer(13)]], device half* o_z [[buffer(14)]], device half* o_a [[buffer(15)]], device half* o_b [[buffer(16)]], \
    constant DeltaNetInParams& p [[buffer(17)]], \
    uint tg_x [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ \
    const int t0 = (p.n_qkv + 31) >> 5, t1 = t0 + ((p.n_z + 31) >> 5), \
              t2 = t1 + ((p.n_a + 31) >> 5), t3 = t2 + ((p.n_b + 31) >> 5); \
    const int t = int(tg_x); \
    if (t >= t3) return; \
    device const uint32_t* wp; device const ushort *sp, *bp; device half* op; \
    int tile_idx, N; \
    if (t < t0)      { wp = w_qkv; sp = s_qkv; bp = b_qkv; op = o_qkv; tile_idx = t;      N = p.n_qkv; } \
    else if (t < t1) { wp = w_z;   sp = s_z;   bp = b_z;   op = o_z;   tile_idx = t - t0; N = p.n_z;   } \
    else if (t < t2) { wp = w_a;   sp = s_a;   bp = b_a;   op = o_a;   tile_idx = t - t1; N = p.n_a;   } \
    else             { wp = w_b;   sp = s_b;   bp = b_b;   op = o_b;   tile_idx = t - t2; N = p.n_b;   } \
    threadgroup half Xs[8 * MT * 64], Ws[32 * 72]; threadgroup float Cs[4 * 64]; \
    QMVMultiParams mp{p.M, p.K, N, p.gs_shift}; \
    qmm_bk64_impl<4, MT>(x, wp, sp, bp, op, mp, Xs, Ws, Cs, uint(tile_idx), tid, sg, lane); \
}
DECL_DN_IN_MMA(deltanet_in_proj_4b_mma8, 1)
DECL_DN_IN_MMA(deltanet_in_proj_4b_mmaP, 4)

#define DECL_QKV_MMA(NAME, MT) \
[[max_total_threads_per_threadgroup(128)]] \
kernel void NAME( \
    device const half* x [[buffer(0)]], \
    device const uint32_t* w_q [[buffer(1)]], device const ushort* s_q [[buffer(2)]], device const ushort* b_q [[buffer(3)]], \
    device const uint32_t* w_k [[buffer(4)]], device const ushort* s_k [[buffer(5)]], device const ushort* b_k [[buffer(6)]], \
    device const uint32_t* w_v [[buffer(7)]], device const ushort* s_v [[buffer(8)]], device const ushort* b_v [[buffer(9)]], \
    device half* o_q [[buffer(10)]], device half* o_k [[buffer(11)]], device half* o_v [[buffer(12)]], \
    constant FullQKVParams& p [[buffer(13)]], \
    uint tg_x [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ \
    const int t0 = (p.n_q + 31) >> 5, t1 = t0 + ((p.n_k + 31) >> 5), t2 = t1 + ((p.n_v + 31) >> 5); \
    const int t = int(tg_x); \
    if (t >= t2) return; \
    device const uint32_t* wp; device const ushort *sp, *bp; device half* op; \
    int tile_idx, N; \
    if (t < t0)      { wp = w_q; sp = s_q; bp = b_q; op = o_q; tile_idx = t;      N = p.n_q; } \
    else if (t < t1) { wp = w_k; sp = s_k; bp = b_k; op = o_k; tile_idx = t - t0; N = p.n_k; } \
    else             { wp = w_v; sp = s_v; bp = b_v; op = o_v; tile_idx = t - t1; N = p.n_v; } \
    threadgroup half Xs[8 * MT * 64], Ws[32 * 72]; threadgroup float Cs[4 * 64]; \
    QMVMultiParams mp{p.M, p.K, N, p.gs_shift}; \
    qmm_bk64_impl<4, MT>(x, wp, sp, bp, op, mp, Xs, Ws, Cs, uint(tile_idx), tid, sg, lane); \
}
DECL_QKV_MMA(q35_qkv_proj_4b_mma8, 1)
DECL_QKV_MMA(q35_qkv_proj_4b_mmaP, 4)

inline void swiglu_store_tile(simdgroup_float8x8 cg, simdgroup_float8x8 cu, int m0,
                              threadgroup float* gbuf, threadgroup float* ubuf, device half* out,
                              int M, int N, int n_base, uint lane) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(cg, gbuf, 8);
    simdgroup_store(cu, ubuf, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = 0; q < 2; ++q) {
        const int idx = int(lane) * 2 + q, m = m0 + (idx >> 3), col = n_base + (idx & 7);
        if (m < M && col < N) {
            const float gv = gbuf[idx];
            out[m * N + col] = half((gv / (1.0f + exp(-gv))) * ubuf[idx]);
        }
    }
}

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

    simdgroup_float8x8 g0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), g1 = g0, g2 = g0, g3 = g0;
    simdgroup_float8x8 u0 = g0, u1 = g0, u2 = g0, u3 = g0;
    const threadgroup half* w_sg = Ws + (int(sg) * 8) * WS;

    for (int k0 = 0; k0 < K; k0 += BK) {
        load_x_tile<MT>(x, M, K, k0, Xs, tid);
        load_w_tile<BITS>(wg, sg_a, bg_a, r, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        MMA_ACC_TILE(Xs, w_sg, g0, g1, g2, g3)
        threadgroup_barrier(mem_flags::mem_threadgroup);

        load_w_tile<BITS>(wu, su_a, bu_a, r, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        MMA_ACC_TILE(Xs, w_sg, u0, u1, u2, u3)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* gbuf = Cg_s + int(sg) * 64;
    threadgroup float* ubuf = Cu_s + int(sg) * 64;
    const int n_base = row_base + int(sg) * 8;
    swiglu_store_tile(g0, u0, 0, gbuf, ubuf, out, M, N, n_base, lane);
    if (MT > 1) {
        if (M > 8)  swiglu_store_tile(g1, u1, 8,  gbuf, ubuf, out, M, N, n_base, lane);
        if (M > 16) swiglu_store_tile(g2, u2, 16, gbuf, ubuf, out, M, N, n_base, lane);
        if (M > 24) swiglu_store_tile(g3, u3, 24, gbuf, ubuf, out, M, N, n_base, lane);
    }
}

#define DECL_SWIG_MMA(NAME, BITS, MT) \
[[max_total_threads_per_threadgroup(128)]] \
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

// ─────────────────────── 4G. Prefill GEMM (M=9..MAX_P): tile 32 token × 64 baris bobot, grid (N/64, M/32) ───────────────────────
// 4 simdgroup, masing-masing 16 token × 32 baris = 8 akumulator. Per langkah kk: 2 load A + 4 load B untuk
// 8 MMA (0.75 load/MMA) vs kernel mmaP 1.25 load/MMA. Dekuantisasi 64 baris dibagi untuk 32 token.
#define GEMM_LD 72

template <int BITS>
inline void dequant_row16(device const uint32_t* w, device const ushort* scales, device const ushort* biases,
                          int r, int K, int KG, int gsh, int kidx, threadgroup half* dst) {
    const ulong s_off = ulong(r) * ulong(KG) + ulong(kidx >> gsh);
    const half2 hs = half2(load_sb_h(scales, s_off));
    const half2 hb = half2(load_sb_h(biases, s_off));
    if (BITS == 4) {
        const ulong wi = ulong(r) * ulong(K >> 3) + ulong(kidx >> 3);
        dequant_4bit_to_tg(w[wi],     hs, hb, dst);
        dequant_4bit_to_tg(w[wi + 1], hs, hb, dst + 8);
    } else if (BITS == 2) {
        dequant_2bit_to_tg(w[ulong(r) * ulong(K >> 4) + ulong(kidx >> 4)], hs, hb, dst);
    } else {
        dequant_gen16_to_tg<BITS>(w + ulong(r) * ulong((K * BITS) >> 5), kidx, float(hs.x), float(hb.x), dst);
    }
}

inline void gemm_load_x(device const half* x, int M, int K, int k0, threadgroup half* Xs, uint tid) {
    const int m = int(tid) >> 2, kq = (int(tid) & 3) * 16;
    threadgroup half4* d = (threadgroup half4*)(Xs + m * GEMM_LD + kq);
    if (m < M) {
        const device half4* s = (const device half4*)(x + m * K + k0 + kq);
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
    } else {
        d[0] = half4(0.0h); d[1] = half4(0.0h); d[2] = half4(0.0h); d[3] = half4(0.0h);
    }
}

template <int BITS>
inline void gemm_load_w(device const uint32_t* w, device const ushort* scales, device const ushort* biases,
                        int row_base, int N, int K, int KG, int gsh, int k0, threadgroup half* Ws, uint tid) {
    const int r_off = int(tid) >> 1, r = min(row_base + r_off, N - 1);
    const int kq = (int(tid) & 1) * 32;
    dequant_row16<BITS>(w, scales, biases, r, K, KG, gsh, k0 + kq,      Ws + r_off * GEMM_LD + kq);
    dequant_row16<BITS>(w, scales, biases, r, K, KG, gsh, k0 + kq + 16, Ws + r_off * GEMM_LD + kq + 16);
}

#define GEMM_MMA(C00, C01, C02, C03, C10, C11, C12, C13) \
    _Pragma("unroll") \
    for (int kk = 0; kk < 64; kk += 8) { \
        simdgroup_half8x8 a0, a1, b0, b1, b2, b3; \
        simdgroup_load(a0, xs_sg + kk, GEMM_LD); \
        simdgroup_load(a1, xs_sg + 8 * GEMM_LD + kk, GEMM_LD); \
        simdgroup_load(b0, ws_sg + kk, GEMM_LD, ulong2(0, 0), true); \
        simdgroup_load(b1, ws_sg + 8 * GEMM_LD + kk, GEMM_LD, ulong2(0, 0), true); \
        simdgroup_load(b2, ws_sg + 16 * GEMM_LD + kk, GEMM_LD, ulong2(0, 0), true); \
        simdgroup_load(b3, ws_sg + 24 * GEMM_LD + kk, GEMM_LD, ulong2(0, 0), true); \
        simdgroup_multiply_accumulate(C00, a0, b0, C00); simdgroup_multiply_accumulate(C01, a0, b1, C01); \
        simdgroup_multiply_accumulate(C02, a0, b2, C02); simdgroup_multiply_accumulate(C03, a0, b3, C03); \
        simdgroup_multiply_accumulate(C10, a1, b0, C10); simdgroup_multiply_accumulate(C11, a1, b1, C11); \
        simdgroup_multiply_accumulate(C12, a1, b2, C12); simdgroup_multiply_accumulate(C13, a1, b3, C13); \
    }

template <int BITS>
inline void qmm_gemm_impl(
    device const half* x, device const uint32_t* w, device const ushort* scales, device const ushort* biases,
    device half* out, QMVMultiParams p, threadgroup half* Xs, threadgroup half* Ws, threadgroup float* Cs,
    uint tg_x, uint tid, uint sg, uint lane)
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift, KG = K >> gsh;
    const int row_base = int(tg_x) * 64;
    const int sm = int(sg) >> 1, sn = int(sg) & 1;
    const threadgroup half* xs_sg = Xs + (sm * 16) * GEMM_LD;
    const threadgroup half* ws_sg = Ws + (sn * 32) * GEMM_LD;

    simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00, c13 = c00;
    for (int k0 = 0; k0 < K; k0 += 64) {
        gemm_load_x(x, M, K, k0, Xs, tid);
        gemm_load_w<BITS>(w, scales, biases, row_base, N, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sm * 16 < M) { GEMM_MMA(c00, c01, c02, c03, c10, c11, c12, c13) }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* cbuf = Cs + int(sg) * 64;
    const int m0 = sm * 16, n0 = row_base + sn * 32;
    qmm_store_tile(c00, m0,     cbuf, out, M, N, n0,      lane);
    qmm_store_tile(c01, m0,     cbuf, out, M, N, n0 + 8,  lane);
    qmm_store_tile(c02, m0,     cbuf, out, M, N, n0 + 16, lane);
    qmm_store_tile(c03, m0,     cbuf, out, M, N, n0 + 24, lane);
    qmm_store_tile(c10, m0 + 8, cbuf, out, M, N, n0,      lane);
    qmm_store_tile(c11, m0 + 8, cbuf, out, M, N, n0 + 8,  lane);
    qmm_store_tile(c12, m0 + 8, cbuf, out, M, N, n0 + 16, lane);
    qmm_store_tile(c13, m0 + 8, cbuf, out, M, N, n0 + 24, lane);
}

template <int BITS>
inline void swiglu_gemm_impl(
    device const half* x,
    device const uint32_t* wg, device const ushort* sg_a, device const ushort* bg_a,
    device const uint32_t* wu, device const ushort* su_a, device const ushort* bu_a,
    device half* out, QMVMultiParams p, threadgroup half* Xs, threadgroup half* Ws,
    threadgroup float* Cg_s, threadgroup float* Cu_s, uint tg_x, uint tid, uint sg, uint lane)
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift, KG = K >> gsh;
    const int row_base = int(tg_x) * 64;
    const int sm = int(sg) >> 1, sn = int(sg) & 1;
    const threadgroup half* xs_sg = Xs + (sm * 16) * GEMM_LD;
    const threadgroup half* ws_sg = Ws + (sn * 32) * GEMM_LD;

    simdgroup_float8x8 g00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 g01 = g00, g02 = g00, g03 = g00, g10 = g00, g11 = g00, g12 = g00, g13 = g00;
    simdgroup_float8x8 u00 = g00, u01 = g00, u02 = g00, u03 = g00, u10 = g00, u11 = g00, u12 = g00, u13 = g00;
    for (int k0 = 0; k0 < K; k0 += 64) {
        gemm_load_x(x, M, K, k0, Xs, tid);
        gemm_load_w<BITS>(wg, sg_a, bg_a, row_base, N, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sm * 16 < M) { GEMM_MMA(g00, g01, g02, g03, g10, g11, g12, g13) }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_load_w<BITS>(wu, su_a, bu_a, row_base, N, K, KG, gsh, k0, Ws, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sm * 16 < M) { GEMM_MMA(u00, u01, u02, u03, u10, u11, u12, u13) }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* gbuf = Cg_s + int(sg) * 64;
    threadgroup float* ubuf = Cu_s + int(sg) * 64;
    const int m0 = sm * 16, n0 = row_base + sn * 32;
    swiglu_store_tile(g00, u00, m0,     gbuf, ubuf, out, M, N, n0,      lane);
    swiglu_store_tile(g01, u01, m0,     gbuf, ubuf, out, M, N, n0 + 8,  lane);
    swiglu_store_tile(g02, u02, m0,     gbuf, ubuf, out, M, N, n0 + 16, lane);
    swiglu_store_tile(g03, u03, m0,     gbuf, ubuf, out, M, N, n0 + 24, lane);
    swiglu_store_tile(g10, u10, m0 + 8, gbuf, ubuf, out, M, N, n0,      lane);
    swiglu_store_tile(g11, u11, m0 + 8, gbuf, ubuf, out, M, N, n0 + 8,  lane);
    swiglu_store_tile(g12, u12, m0 + 8, gbuf, ubuf, out, M, N, n0 + 16, lane);
    swiglu_store_tile(g13, u13, m0 + 8, gbuf, ubuf, out, M, N, n0 + 24, lane);
}

#define DECL_QMM_GEMM(NAME, BITS) \
[[max_total_threads_per_threadgroup(128)]] \
kernel void NAME(device const half* x [[buffer(0)]], device const uint32_t* w [[buffer(1)]], \
    device const ushort* s [[buffer(2)]], device const ushort* b [[buffer(3)]], device half* out [[buffer(4)]], \
    constant QMVMultiParams& p [[buffer(5)]], \
    uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ \
    threadgroup half Xs[32 * GEMM_LD], Ws[64 * GEMM_LD]; threadgroup float Cs[4 * 64]; \
    const int mb = int(tgp.y) * 32; QMVMultiParams q = p; q.M = min(32, p.M - mb); \
    qmm_gemm_impl<BITS>(x + ulong(mb) * ulong(p.K), w, s, b, out + ulong(mb) * ulong(p.N), q, Xs, Ws, Cs, tgp.x, tid, sg, lane); \
}
#define DECL_SWIG_GEMM(NAME, BITS) \
[[max_total_threads_per_threadgroup(128)]] \
kernel void NAME(device const half* x [[buffer(0)]], \
    device const uint32_t* wg [[buffer(1)]], device const ushort* sg_a [[buffer(2)]], device const ushort* bg_a [[buffer(3)]], \
    device const uint32_t* wu [[buffer(4)]], device const ushort* su_a [[buffer(5)]], device const ushort* bu_a [[buffer(6)]], \
    device half* out [[buffer(7)]], constant QMVMultiParams& p [[buffer(8)]], \
    uint2 tgp [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ \
    threadgroup half Xs[32 * GEMM_LD], Ws[64 * GEMM_LD]; threadgroup float Cg_s[4 * 64], Cu_s[4 * 64]; \
    const int mb = int(tgp.y) * 32; QMVMultiParams q = p; q.M = min(32, p.M - mb); \
    swiglu_gemm_impl<BITS>(x + ulong(mb) * ulong(p.K), wg, sg_a, bg_a, wu, su_a, bu_a, out + ulong(mb) * ulong(p.N), q, \
                           Xs, Ws, Cg_s, Cu_s, tgp.x, tid, sg, lane); \
}
DECL_QMM_GEMM(qmm_2b_gemm, 2)
DECL_SWIG_GEMM(dense_swiglu_2b_gemm, 2)
DECL_QMM_GEMM(qmm_3b_gemm, 3)
DECL_SWIG_GEMM(dense_swiglu_3b_gemm, 3)
DECL_QMM_GEMM(qmm_4b_gemm, 4)
DECL_SWIG_GEMM(dense_swiglu_4b_gemm, 4)
DECL_QMM_GEMM(qmm_6b_gemm, 6)
DECL_SWIG_GEMM(dense_swiglu_6b_gemm, 6)
DECL_QMM_GEMM(qmm_8b_gemm, 8)
DECL_SWIG_GEMM(dense_swiglu_8b_gemm, 8)

// ─────────────────────── 5. Zero-Copy Speculative Chunked DeltaNet (M=1 & M=2..8 + Delayed Commit) ───────────────────────
kernel void deltanet_conv1d_silu_m1(
    device half* qkv [[buffer(0)]], device half* conv_state [[buffer(1)]],
    device const half* conv_w [[buffer(2)]], constant Conv1dParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const int c = int(gid);
    if (c >= p.C) return;
    const float s0 = float(conv_state[0 * p.C + c]);
    const float s1 = float(conv_state[1 * p.C + c]);
    const float s2 = float(conv_state[2 * p.C + c]);
    const float x  = float(qkv[c]);
    const float y  = s0 * float(conv_w[c * 4 + 0]) + s1 * float(conv_w[c * 4 + 1])
                   + s2 * float(conv_w[c * 4 + 2]) + x  * float(conv_w[c * 4 + 3]);
    conv_state[0 * p.C + c] = half(s1);
    conv_state[1 * p.C + c] = half(s2);
    conv_state[2 * p.C + c] = half(x);
    qkv[c] = half(y / (1.0f + exp(-y)));
}

kernel void deltanet_conv1d_silu_spec(
    device half* qkv [[buffer(0)]], device const half* conv_state [[buffer(1)]],
    device const half* conv_w [[buffer(2)]], device half* raw_hist [[buffer(3)]],
    constant Conv1dParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]])
{
    const int c = int(gid);
    if (c >= p.C) return;
    float w0 = float(conv_state[0 * p.C + c]);
    float w1 = float(conv_state[1 * p.C + c]);
    float w2 = float(conv_state[2 * p.C + c]);
    const float cw0 = float(conv_w[c * 4 + 0]), cw1 = float(conv_w[c * 4 + 1]);
    const float cw2 = float(conv_w[c * 4 + 2]), cw3 = float(conv_w[c * 4 + 3]);
    for (int m = 0; m < p.M; ++m) {
        const int idx = m * p.C + c;
        const float x = float(qkv[idx]);
        raw_hist[idx] = half(x);
        const float y = w0 * cw0 + w1 * cw1 + w2 * cw2 + x * cw3;
        qkv[idx] = half(y / (1.0f + exp(-y)));
        w0 = w1; w1 = w2; w2 = x;
    }
}

kernel void deltanet_recurrent_m1(
    device const half* qkv [[buffer(0)]], device const half* z_buf [[buffer(1)]],
    device const half* a_buf [[buffer(2)]], device const half* b_buf [[buffer(3)]],
    device const float* A_log [[buffer(4)]], device const float* dt_bias [[buffer(5)]],
    device const half* norm_w [[buffer(6)]], device float* S_state [[buffer(7)]],
    device half* out [[buffer(8)]], constant DeltaNetParams& p [[buffer(9)]],
    uint tg_x [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int hv = int(tg_x);
    if (hv >= p.n_vh) return;
    const int rep = p.n_vh / p.n_kh, hk = hv / rep, k_dim = p.n_kh * p.k_hd;

    threadgroup float q_s[128], k_s[128], y_s[128], sm_ss[8];

    const device half* q_ptr = qkv + hk * p.k_hd;
    const device half* k_ptr = qkv + k_dim + hk * p.k_hd;
    const device half* v_ptr = qkv + 2 * k_dim + hv * p.v_hd;

    float qv_t = (int(tid) < p.k_hd) ? float(q_ptr[tid]) : 0.0f;
    float kv_t = (int(tid) < p.k_hd) ? float(k_ptr[tid]) : 0.0f;
    float q_ss = simd_sum(qv_t * qv_t), k_ss = simd_sum(kv_t * kv_t);
    if (lane == 0) sm_ss[sg] = q_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float q_inv = rsqrt(sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3] + 1e-6f) * rsqrt(float(p.k_hd));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) sm_ss[sg] = k_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float k_inv = rsqrt(sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3] + 1e-6f);
    if (int(tid) < p.k_hd) { q_s[tid] = qv_t * q_inv; k_s[tid] = kv_t * k_inv; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sp_in = float(a_buf[hv]) + dt_bias[hv];
    const float sp    = (sp_in > 20.0f) ? sp_in : log(1.0f + exp(sp_in));
    const float decay = exp(-exp(A_log[hv]) * sp);
    const float beta  = 1.0f / (1.0f + exp(-float(b_buf[hv])));

    const int col4 = int(lane) * 4;
    const float4 q_vec = float4(q_s[col4], q_s[col4+1], q_s[col4+2], q_s[col4+3]);
    const float4 k_vec = float4(k_s[col4], k_s[col4+1], k_s[col4+2], k_s[col4+3]);
    device float4* S_head = (device float4*)(S_state + ulong(hv) * ulong(p.v_hd * p.k_hd));
    const int stride4 = p.k_hd >> 2;

    for (int vi = int(sg); vi < p.v_hd; vi += 8) {
        device float4* s_row = S_head + vi * stride4;
        float4 sv = s_row[lane] * decay;
        const float v_pred = simd_sum(dot(sv, k_vec));
        const float delta  = beta * (float(v_ptr[vi]) - v_pred);
        sv = fma(float4(delta), k_vec, sv);
        s_row[lane] = sv;
        const float yv = simd_sum(dot(sv, q_vec));
        if (lane == 0) y_s[vi] = yv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float yv_t = (int(tid) < p.v_hd) ? y_s[tid] : 0.0f;
    float y_ss = simd_sum(yv_t * yv_t);
    if (lane == 0) sm_ss[sg] = y_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float y_inv = rsqrt((sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3]) / float(p.v_hd) + p.eps);
    if (int(tid) < p.v_hd) {
        const float zv = float(z_buf[hv * p.v_hd + int(tid)]);
        out[hv * p.v_hd + int(tid)] = half((yv_t * y_inv * float(norm_w[tid])) * (zv / (1.0f + exp(-zv))));
    }
}

kernel void deltanet_recurrent_spec(
    device const half* qkv [[buffer(0)]], device const half* z_buf [[buffer(1)]],
    device const half* a_buf [[buffer(2)]], device const half* b_buf [[buffer(3)]],
    device const float* A_log [[buffer(4)]], device const float* dt_bias [[buffer(5)]],
    device const half* norm_w [[buffer(6)]], device const float* S_state [[buffer(7)]],
    device half* out [[buffer(8)]],
    device float* hist_k [[buffer(9)]], device float* hist_d [[buffer(10)]], device float* hist_dec [[buffer(11)]],
    constant DeltaNetParams& p [[buffer(12)]],
    uint tg_x [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int hv = int(tg_x);
    if (hv >= p.n_vh) return;
    const int M = p.M, rep = p.n_vh / p.n_kh, hk = hv / rep;
    const int k_dim = p.n_kh * p.k_hd, v_dim = p.n_vh * p.v_hd, qkv_stride = 2 * k_dim + v_dim;

    threadgroup float q_s[8][128], k_s[8][128], y_s[8][128], dec_s[8], bet_s[8], sm_ss[8];

    for (int m = 0; m < M; ++m) {
        const device half* q_ptr = qkv + m * qkv_stride + hk * p.k_hd;
        const device half* k_ptr = qkv + m * qkv_stride + k_dim + hk * p.k_hd;
        float qv_t = (int(tid) < p.k_hd) ? float(q_ptr[tid]) : 0.0f;
        float kv_t = (int(tid) < p.k_hd) ? float(k_ptr[tid]) : 0.0f;
        float q_ss = simd_sum(qv_t * qv_t), k_ss = simd_sum(kv_t * kv_t);
        if (lane == 0) sm_ss[sg] = q_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float q_inv = rsqrt(sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3] + 1e-6f) * rsqrt(float(p.k_hd));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) sm_ss[sg] = k_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float k_inv = rsqrt(sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3] + 1e-6f);
        if (int(tid) < p.k_hd) {
            const float kn = kv_t * k_inv;
            q_s[m][tid] = qv_t * q_inv;
            k_s[m][tid] = kn;
            hist_k[(m * p.n_vh + hv) * p.k_hd + int(tid)] = kn;
        }
        if (tid == 0) {
            const float sp_in = float(a_buf[m * p.n_vh + hv]) + dt_bias[hv];
            const float sp    = (sp_in > 20.0f) ? sp_in : log(1.0f + exp(sp_in));
            const float dec   = exp(-exp(A_log[hv]) * sp);
            dec_s[m] = dec;
            bet_s[m] = 1.0f / (1.0f + exp(-float(b_buf[m * p.n_vh + hv])));
            hist_dec[m * p.n_vh + hv] = dec;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int col4 = int(lane) * 4;
    const device float4* S_head = (const device float4*)(S_state + ulong(hv) * ulong(p.v_hd * p.k_hd));
    const int stride4 = p.k_hd >> 2;

    for (int vi = int(sg); vi < p.v_hd; vi += 8) {
        float4 sv = S_head[vi * stride4 + lane];
        for (int m = 0; m < M; ++m) {
            const float4 k_vec = float4(k_s[m][col4], k_s[m][col4+1], k_s[m][col4+2], k_s[m][col4+3]);
            const float4 q_vec = float4(q_s[m][col4], q_s[m][col4+1], q_s[m][col4+2], q_s[m][col4+3]);
            sv = sv * dec_s[m];
            const float v_pred = simd_sum(dot(sv, k_vec));
            const float v_val  = float(qkv[m * qkv_stride + 2 * k_dim + hv * p.v_hd + vi]);
            const float delta  = bet_s[m] * (v_val - v_pred);
            sv = fma(float4(delta), k_vec, sv);
            const float yv = simd_sum(dot(sv, q_vec));
            if (lane == 0) {
                y_s[m][vi] = yv;
                hist_d[(m * p.n_vh + hv) * p.v_hd + vi] = delta;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int m = 0; m < M; ++m) {
        float yv_t = (int(tid) < p.v_hd) ? y_s[m][tid] : 0.0f;
        float y_ss = simd_sum(yv_t * yv_t);
        if (lane == 0) sm_ss[sg] = y_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float y_inv = rsqrt((sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3]) / float(p.v_hd) + p.eps);
        if (int(tid) < p.v_hd) {
            const int o_idx = m * v_dim + hv * p.v_hd + int(tid);
            const float zv = float(z_buf[o_idx]);
            out[o_idx] = half((yv_t * y_inv * float(norm_w[tid])) * (zv / (1.0f + exp(-zv))));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ─────────────────────── 5B. Prefill DeltaNet (M bebas, state langsung diperbarui) ───────────────────────
// Prefill tidak butuh rollback: state diperbarui token demi token di dalam SATU dispatch, dengan urutan
// operasi yang sama persis dengan jalur M=1 (deltanet_recurrent_m1) → hasil identik dengan decode satu-satu.
// Baris state S (v_hd × k_hd float) disimpan di register selama seluruh chunk: dibaca & ditulis sekali.
kernel void deltanet_conv1d_silu_prefill(
    device half* qkv [[buffer(0)]], device half* conv_state [[buffer(1)]],
    device const half* conv_w [[buffer(2)]], constant Conv1dParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const int c = int(gid);
    if (c >= p.C) return;
    float w0 = float(conv_state[0 * p.C + c]);
    float w1 = float(conv_state[1 * p.C + c]);
    float w2 = float(conv_state[2 * p.C + c]);
    const float cw0 = float(conv_w[c * 4 + 0]), cw1 = float(conv_w[c * 4 + 1]);
    const float cw2 = float(conv_w[c * 4 + 2]), cw3 = float(conv_w[c * 4 + 3]);
    for (int m = 0; m < p.M; ++m) {
        const int idx = m * p.C + c;
        const float x = float(qkv[idx]);
        const float y = w0 * cw0 + w1 * cw1 + w2 * cw2 + x * cw3;
        qkv[idx] = half(y / (1.0f + exp(-y)));
        w0 = w1; w1 = w2; w2 = x;
    }
    conv_state[0 * p.C + c] = half(w0);
    conv_state[1 * p.C + c] = half(w1);
    conv_state[2 * p.C + c] = half(w2);
}

kernel void deltanet_recurrent_prefill(
    device const half* qkv [[buffer(0)]], device const half* z_buf [[buffer(1)]],
    device const half* a_buf [[buffer(2)]], device const half* b_buf [[buffer(3)]],
    device const float* A_log [[buffer(4)]], device const float* dt_bias [[buffer(5)]],
    device const half* norm_w [[buffer(6)]], device float* S_state [[buffer(7)]],
    device half* out [[buffer(8)]], constant DeltaNetParams& p [[buffer(9)]],
    uint tg_x [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int hv = int(tg_x);
    if (hv >= p.n_vh) return;
    const int rep = p.n_vh / p.n_kh, hk = hv / rep;
    const int k_dim = p.n_kh * p.k_hd, v_dim = p.n_vh * p.v_hd, qkv_stride = 2 * k_dim + v_dim;

    threadgroup float q_s[128], k_s[128], y_s[128], sm_ss[8];

    const int col4 = int(lane) * 4;
    device float4* S_head = (device float4*)(S_state + ulong(hv) * ulong(p.v_hd * p.k_hd));
    const int stride4 = p.k_hd >> 2;
    const int nrows = (p.v_hd - int(sg) + 7) >> 3;      // baris vi = sg + 8j, j < nrows (≤ 16)

    float4 sv[16];
    #pragma unroll
    for (int j = 0; j < 16; ++j) if (j < nrows) sv[j] = S_head[(int(sg) + 8 * j) * stride4 + lane];

    for (int m = 0; m < p.M; ++m) {
        const device half* q_ptr = qkv + m * qkv_stride + hk * p.k_hd;
        const device half* k_ptr = qkv + m * qkv_stride + k_dim + hk * p.k_hd;
        const device half* v_ptr = qkv + m * qkv_stride + 2 * k_dim + hv * p.v_hd;

        float qv_t = (int(tid) < p.k_hd) ? float(q_ptr[tid]) : 0.0f;
        float kv_t = (int(tid) < p.k_hd) ? float(k_ptr[tid]) : 0.0f;
        float q_ss = simd_sum(qv_t * qv_t), k_ss = simd_sum(kv_t * kv_t);
        if (lane == 0) sm_ss[sg] = q_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float q_inv = rsqrt(sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3] + 1e-6f) * rsqrt(float(p.k_hd));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) sm_ss[sg] = k_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float k_inv = rsqrt(sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3] + 1e-6f);
        if (int(tid) < p.k_hd) { q_s[tid] = qv_t * q_inv; k_s[tid] = kv_t * k_inv; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float sp_in = float(a_buf[m * p.n_vh + hv]) + dt_bias[hv];
        const float sp    = (sp_in > 20.0f) ? sp_in : log(1.0f + exp(sp_in));
        const float decay = exp(-exp(A_log[hv]) * sp);
        const float beta  = 1.0f / (1.0f + exp(-float(b_buf[m * p.n_vh + hv])));

        const float4 q_vec = float4(q_s[col4], q_s[col4+1], q_s[col4+2], q_s[col4+3]);
        const float4 k_vec = float4(k_s[col4], k_s[col4+1], k_s[col4+2], k_s[col4+3]);
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (j < nrows) {
                const int vi = int(sg) + 8 * j;
                float4 s = sv[j] * decay;
                const float v_pred = simd_sum(dot(s, k_vec));
                const float delta  = beta * (float(v_ptr[vi]) - v_pred);
                s = fma(float4(delta), k_vec, s);
                sv[j] = s;
                const float yv = simd_sum(dot(s, q_vec));
                if (lane == 0) y_s[vi] = yv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float yv_t = (int(tid) < p.v_hd) ? y_s[tid] : 0.0f;
        float y_ss = simd_sum(yv_t * yv_t);
        if (lane == 0) sm_ss[sg] = y_ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float y_inv = rsqrt((sm_ss[0] + sm_ss[1] + sm_ss[2] + sm_ss[3]) / float(p.v_hd) + p.eps);
        if (int(tid) < p.v_hd) {
            const int o_idx = m * v_dim + hv * p.v_hd + int(tid);
            const float zv = float(z_buf[o_idx]);
            out[o_idx] = half((yv_t * y_inv * float(norm_w[tid])) * (zv / (1.0f + exp(-zv))));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // sm_ss / q_s / y_s dipakai ulang token berikutnya
    }

    #pragma unroll
    for (int j = 0; j < 16; ++j) if (j < nrows) S_head[(int(sg) + 8 * j) * stride4 + lane] = sv[j];
}

kernel void deltanet_commit_conv1d(
    device half* conv_state [[buffer(0)]], device const half* raw_hist [[buffer(1)]],
    constant Conv1dParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]])
{
    const int c = int(gid);
    if (c >= p.C) return;
    half w0 = conv_state[0 * p.C + c], w1 = conv_state[1 * p.C + c], w2 = conv_state[2 * p.C + c];
    for (int m = 0; m < p.M; ++m) { half x = raw_hist[m * p.C + c]; w0 = w1; w1 = w2; w2 = x; }
    conv_state[0 * p.C + c] = w0; conv_state[1 * p.C + c] = w1; conv_state[2 * p.C + c] = w2;
}

kernel void deltanet_commit_state(
    device float* S_state [[buffer(0)]], device const float* hist_k [[buffer(1)]],
    device const float* hist_d [[buffer(2)]], device const float* hist_dec [[buffer(3)]],
    constant DeltaNetParams& p [[buffer(4)]],
    uint tg_x [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int hv = int(tg_x);
    if (hv >= p.n_vh) return;
    threadgroup float k_s[8][128];
    for (int m = 0; m < p.M; ++m)
        if (int(tid) < p.k_hd) k_s[m][tid] = hist_k[(m * p.n_vh + hv) * p.k_hd + int(tid)];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int col4 = int(lane) * 4;
    device float4* S_head = (device float4*)(S_state + ulong(hv) * ulong(p.v_hd * p.k_hd));
    const int stride4 = p.k_hd >> 2;
    for (int vi = int(sg); vi < p.v_hd; vi += 8) {
        float4 sv = S_head[vi * stride4 + lane];
        for (int m = 0; m < p.M; ++m) {
            const float4 k_vec = float4(k_s[m][col4], k_s[m][col4+1], k_s[m][col4+2], k_s[m][col4+3]);
            sv = fma(float4(hist_d[(m * p.n_vh + hv) * p.v_hd + vi]), k_vec, sv * hist_dec[m * p.n_vh + hv]);
        }
        S_head[vi * stride4 + lane] = sv;
    }
}

// ─────────────────────── 6. Multi-Token Gated Full Attention hd=256 (M=1..8) ───────────────────────
kernel void q35_qknorm_rope_kv(
    device const half* q_and_gate [[buffer(0)]], device const half* k_in [[buffer(1)]], device const half* v_in [[buffer(2)]],
    device half* q_out [[buffer(3)]], device half* gate_out [[buffer(4)]],
    device half* k_cache [[buffer(5)]], device half* v_cache [[buffer(6)]],
    device const half* cos_t [[buffer(7)]], device const half* sin_t [[buffer(8)]],
    device const half* q_norm [[buffer(9)]], device const half* k_norm [[buffer(10)]],
    constant QKNormRopeParams& p [[buffer(11)]],
    uint2 tg_id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    const int h = int(tg_id.x), m = int(tg_id.y);
    if (h >= p.n_heads || m >= p.M) return;
    const int hd = p.head_dim, half_r = p.rope_dim >> 1, cur_pos = p.pos + m, q_stride = p.n_heads * hd;
    const device half* q_src = q_and_gate + m * (2 * q_stride) + h * (2 * hd);
    const device half* g_src = q_src + hd;

    float qv[8], q_ss = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const int idx = k * 32 + int(lane);
        qv[k] = float(q_src[idx]); q_ss += qv[k] * qv[k];
        gate_out[m * q_stride + h * hd + idx] = g_src[idx];
    }
    const float q_inv = rsqrt(simd_sum(q_ss) / float(hd) + p.eps);
    for (int k = 0; k < 8; ++k) qv[k] *= q_inv * float(q_norm[k * 32 + int(lane)]);
    const float c = float(cos_t[cur_pos * half_r + int(lane)]), s = float(sin_t[cur_pos * half_r + int(lane)]);
    const float rq0 = qv[0] * c - qv[1] * s, rq1 = qv[1] * c + qv[0] * s;
    qv[0] = rq0; qv[1] = rq1;
    for (int k = 0; k < 8; ++k) q_out[m * q_stride + h * hd + k * 32 + int(lane)] = half(qv[k]);

    if (h < p.n_kv_heads) {
        const int kv_stride = p.n_kv_heads * hd;
        const device half *k_src = k_in + m * kv_stride + h * hd, *v_src = v_in + m * kv_stride + h * hd;
        const ulong dst = (ulong(h) * ulong(p.max_seq) + ulong(cur_pos)) * ulong(hd);
        float kv[8], k_ss = 0.0f;
        for (int k = 0; k < 8; ++k) {
            const int idx = k * 32 + int(lane);
            kv[k] = float(k_src[idx]); k_ss += kv[k] * kv[k]; v_cache[dst + idx] = v_src[idx];
        }
        const float k_inv = rsqrt(simd_sum(k_ss) / float(hd) + p.eps);
        for (int k = 0; k < 8; ++k) kv[k] *= k_inv * float(k_norm[k * 32 + int(lane)]);
        const float rk0 = kv[0] * c - kv[1] * s, rk1 = kv[1] * c + kv[0] * s;
        kv[0] = rk0; kv[1] = rk1;
        for (int k = 0; k < 8; ++k) k_cache[dst + k * 32 + int(lane)] = half(kv[k]);
    }
}

kernel void gqa_attention_256_gated(
    device const half* q_buf [[buffer(0)]], device const half* gate_buf [[buffer(1)]],
    device const half* k_cache [[buffer(2)]], device const half* v_cache [[buffer(3)]],
    device half* attn_out [[buffer(4)]], constant GQA256Params& p [[buffer(5)]],
    uint2 tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int h = int(tg_id.x), m = int(tg_id.y);
    if (h >= p.n_heads || m >= p.M) return;
    const int kv_h = h / (p.n_heads / p.n_kv_heads), seq_len = p.pos + m + 1;
    const float scale_log2e = (1.0f / 16.0f) * LOG2E;
    const int q_stride = p.n_heads * 256, q_base = m * q_stride + h * 256 + int(lane) * 8;
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

    threadgroup float sm_max[8], sm_sum[8];
    threadgroup float4 sm_o0[8][32], sm_o1[8][32];
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
        const float4 g0 = float4(*((const device half4*)(gate_buf + q_base)));
        const float4 g1 = float4(*((const device half4*)(gate_buf + q_base + 4)));
        *((device half4*)(attn_out + q_base))     = half4((go0 * inv) / (1.0f + exp(-g0)));
        *((device half4*)(attn_out + q_base + 4)) = half4((go1 * inv) / (1.0f + exp(-g1)));
    }
}

// ─────────────────────── 7. v13.1 High-Occupancy Low-Register 3D Contiguous-Expert MoE (4-Byte Safe!) ───────────────────────
template <int BITS>
inline void moe_router_shgate_impl(
    device const half*     x,
    device const uint32_t* w_r, device const ushort* s_r, device const ushort* b_r,
    device const uint32_t* w_g, device const ushort* s_g, device const ushort* b_g,
    device half* out_r, device half* out_g, constant QMVMultiParams& p,
    uint tg_x, uint sg, uint lane)
{
    const int M = p.M, K = p.K, N = p.N, gsh = p.gs_shift;
    const int row = int(tg_x) * 4 + int(sg);
    const ulong kg = ulong(K >> gsh);

    device const uint32_t* wp; device const ushort *sp, *bp; device half* op;
    int lr = 0, out_stride = N;
    if (row < N)       { wp = w_r; sp = s_r; bp = b_r; op = out_r; lr = row; out_stride = N; }
    else if (row == N) { wp = w_g; sp = s_g; bp = b_g; op = out_g; lr = 0;   out_stride = 1; }
    else return;

    const ulong so = ulong(lr) * kg;
    float acc[8] = {0,0,0,0,0,0,0,0};
    if (BITS == 8) {
        const ulong wo = ulong(lr) * ulong(K >> 2);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 2, g = gk >> gsh;
            const uint32_t w0 = wp[wo+wi], w1 = wp[wo+wi+1], w2 = wp[wo+wi+2], w3 = wp[wo+wi+3];
            const float s = load_sb_f(sp, so + g), b = load_sb_f(bp, so + g);
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                acc[m] += dot_8bit_word(w0, s, b, xp[0]) + dot_8bit_word(w1, s, b, xp[1])
                        + dot_8bit_word(w2, s, b, xp[2]) + dot_8bit_word(w3, s, b, xp[3]);
            }
        }
    } else {
        const ulong wo = ulong(lr) * ulong(K >> 3);
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const uint32_t wvx = wp[wo+wi], wvy = wp[wo+wi+1];
            const half2 hs = half2(load_sb_h(sp, so + g)), hb = half2(load_sb_h(bp, so + g));
            for (int m = 0; m < M; ++m) {
                const device half4* xp = (const device half4*)(x + m * K + gk);
                acc[m] += dot_4bit_word(wvx, hs, hb, xp) + dot_4bit_word(wvy, hs, hb, xp + 2);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        const float v = simd_sum(acc[m]);
        if (lane == 0) op[m * out_stride + lr] = half(v);
    }
}

kernel void moe_router_shgate_8b(
    device const half* x [[buffer(0)]],
    device const uint32_t* wr [[buffer(1)]], device const ushort* sr [[buffer(2)]], device const ushort* br [[buffer(3)]],
    device const uint32_t* wg [[buffer(4)]], device const ushort* sg_a [[buffer(5)]], device const ushort* bg_a [[buffer(6)]],
    device half* out_r [[buffer(7)]], device half* out_g [[buffer(8)]], constant QMVMultiParams& p [[buffer(9)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{ moe_router_shgate_impl<8>(x, wr, sr, br, wg, sg_a, bg_a, out_r, out_g, p, tg_x, sg, lane); }

kernel void moe_router_shgate_4b(
    device const half* x [[buffer(0)]],
    device const uint32_t* wr [[buffer(1)]], device const ushort* sr [[buffer(2)]], device const ushort* br [[buffer(3)]],
    device const uint32_t* wg [[buffer(4)]], device const ushort* sg_a [[buffer(5)]], device const ushort* bg_a [[buffer(6)]],
    device half* out_r [[buffer(7)]], device half* out_g [[buffer(8)]], constant QMVMultiParams& p [[buffer(9)]],
    uint tg_x [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{ moe_router_shgate_impl<4>(x, wr, sr, br, wg, sg_a, bg_a, out_r, out_g, p, tg_x, sg, lane); }

// 7B. Register-Only 32-Lane Parallel SIMD Top-K across M tokens
kernel void moe_topk_and_shgate(
    device const half* router_logits [[buffer(0)]], device const half* sh_gate_logit [[buffer(1)]],
    device int* top_ids [[buffer(2)]], device float* top_w [[buffer(3)]],
    device float* out_sh_g [[buffer(4)]], constant TopKParams& p [[buffer(5)]],
    uint tg_x [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]])
{
    const int m = int(tg_x);
    if (m >= p.M) return;
    const device half* r_row = router_logits + m * p.E;
    float v[8];
    float local_mx = -1e30f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = int(lane) + i * 32;
        const float val = (e < p.E) ? float(r_row[e]) : -1e30f;
        v[i] = val; local_mx = max(local_mx, val);
    }
    const float mx = simd_max(local_mx);
    float local_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = int(lane) + i * 32;
        const float ev = (e < p.E) ? exp(v[i] - mx) : 0.0f;
        v[i] = ev; local_sum += ev;
    }
    const float inv = 1.0f / max(simd_sum(local_sum), 1e-12f);
    #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] *= inv;

    float w_sum = 0.0f;
    for (int k = 0; k < p.top_k; ++k) {
        float lv = v[0]; int lslot = 0;
        #pragma unroll
        for (int i = 1; i < 8; ++i) if (v[i] > lv) { lv = v[i]; lslot = i; }
        const float gv = simd_max(lv);
        const int win_lane = simd_min((lv == gv) ? int(lane) : 32);
        const int win_e = simd_broadcast(int(lane) + lslot * 32, ushort(win_lane));
        if (int(lane) == win_lane) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) if (i == lslot) v[i] = -2.0f;
        }
        if (lane == 0) { top_ids[m * p.top_k + k] = win_e; top_w[m * p.top_k + k] = gv; w_sum += gv; }
    }
    if (lane == 0) {
        out_sh_g[m] = 1.0f / (1.0f + exp(-float(sh_gate_logit[m])));
        const float w_inv = 1.0f / max(w_sum, 1e-12f);
        for (int k = 0; k < p.top_k; ++k) top_w[m * p.top_k + k] *= w_inv;
    }
}

// 7C. High-Occupancy 3D Dual-Row Fused Gate+Up MoE SwiGLU: grid = ((N+7)/8, top_k, M), 4-Byte Safe!
template <int BITS>
inline void moe_swiglu_3d_impl(
    device const half* x, device const int* top_ids,
    device const uint32_t* wg, device const ushort* sg_a, device const ushort* bg_a,
    device const uint32_t* wu, device const ushort* su_a, device const ushort* bu_a,
    device half* out, constant MoEProjParams& p, uint3 tg_id, uint sg, uint lane)
{
    const int row0 = (int(tg_id.x) * 4 + int(sg)) * 2;
    const int k_i  = int(tg_id.y);
    const int m    = int(tg_id.z);
    if (row0 >= p.N || k_i >= p.top_k || m >= p.M) return;

    const int r0 = row0, r1 = min(row0 + 1, p.N - 1);
    const ulong e = ulong(top_ids[m * p.top_k + k_i]);
    const int K = p.K, gsh = p.gs_shift;
    const ulong kg = ulong(K >> gsh);
    const ulong so0 = e * p.stride_s + ulong(r0) * kg, so1 = e * p.stride_s + ulong(r1) * kg;
    const device half* xm = x + m * K;

    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;

    if (BITS == 4) {
        const ulong kw = ulong(K >> 3);
        const ulong wo0 = e * p.stride_w + ulong(r0) * kw, wo1 = e * p.stride_w + ulong(r1) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const device half4* xp = (const device half4*)(xm + gk);
            const half2 hsg0 = half2(load_sb_h(sg_a, so0+g)), hbg0 = half2(load_sb_h(bg_a, so0+g));
            const half2 hsg1 = half2(load_sb_h(sg_a, so1+g)), hbg1 = half2(load_sb_h(bg_a, so1+g));
            const half2 hsu0 = half2(load_sb_h(su_a, so0+g)), hbu0 = half2(load_sb_h(bu_a, so0+g));
            const half2 hsu1 = half2(load_sb_h(su_a, so1+g)), hbu1 = half2(load_sb_h(bu_a, so1+g));
            ag0 += dot_4bit_word(wg[wo0+wi], hsg0, hbg0, xp) + dot_4bit_word(wg[wo0+wi+1], hsg0, hbg0, xp + 2);
            ag1 += dot_4bit_word(wg[wo1+wi], hsg1, hbg1, xp) + dot_4bit_word(wg[wo1+wi+1], hsg1, hbg1, xp + 2);
            au0 += dot_4bit_word(wu[wo0+wi], hsu0, hbu0, xp) + dot_4bit_word(wu[wo0+wi+1], hsu0, hbu0, xp + 2);
            au1 += dot_4bit_word(wu[wo1+wi], hsu1, hbu1, xp) + dot_4bit_word(wu[wo1+wi+1], hsu1, hbu1, xp + 2);
        }
    } else if (BITS == 3) {
        const ulong kw = ulong((K * 3) >> 5);
        const ulong wo0 = e * p.stride_w + ulong(r0) * kw, wo1 = e * p.stride_w + ulong(r1) * kw;
        for (int k0 = 0; k0 < K; k0 += 1024) {
            const int gk = k0 + int(lane) * 32;
            if (gk >= K) break;
            const int blk = (gk >> 5) * 3, g = gk >> gsh;
            const device half4* xp = (const device half4*)(xm + gk);
            ag0 += dot_3bit_block32(wg[wo0+blk], wg[wo0+blk+1], wg[wo0+blk+2], load_sb_f(sg_a, so0+g), load_sb_f(bg_a, so0+g), xp);
            ag1 += dot_3bit_block32(wg[wo1+blk], wg[wo1+blk+1], wg[wo1+blk+2], load_sb_f(sg_a, so1+g), load_sb_f(bg_a, so1+g), xp);
            au0 += dot_3bit_block32(wu[wo0+blk], wu[wo0+blk+1], wu[wo0+blk+2], load_sb_f(su_a, so0+g), load_sb_f(bu_a, so0+g), xp);
            au1 += dot_3bit_block32(wu[wo1+blk], wu[wo1+blk+1], wu[wo1+blk+2], load_sb_f(su_a, so1+g), load_sb_f(bu_a, so1+g), xp);
        }
    } else {
        const ulong kw = ulong(K >> 4);
        const ulong wo0 = e * p.stride_w + ulong(r0) * kw, wo1 = e * p.stride_w + ulong(r1) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int kwi = gk >> 4, g = gk >> gsh;
            const device half4* xp = (const device half4*)(xm + gk);
            ag0 += dot_2bit_word(wg[wo0+kwi], half2(load_sb_h(sg_a, so0+g)), half2(load_sb_h(bg_a, so0+g)), xp);
            ag1 += dot_2bit_word(wg[wo1+kwi], half2(load_sb_h(sg_a, so1+g)), half2(load_sb_h(bg_a, so1+g)), xp);
            au0 += dot_2bit_word(wu[wo0+kwi], half2(load_sb_h(su_a, so0+g)), half2(load_sb_h(bu_a, so0+g)), xp);
            au1 += dot_2bit_word(wu[wo1+kwi], half2(load_sb_h(su_a, so1+g)), half2(load_sb_h(bu_a, so1+g)), xp);
        }
    }

    const float gv0 = simd_sum(ag0), uv0 = simd_sum(au0);
    const float gv1 = simd_sum(ag1), uv1 = simd_sum(au1);
    const int o_base = (m * p.top_k + k_i) * p.N;
    if (lane == 0) {
        out[o_base + r0] = half((gv0 / (1.0f + exp(-gv0))) * uv0);
        if (row0 + 1 < p.N) out[o_base + r1] = half((gv1 / (1.0f + exp(-gv1))) * uv1);
    }
}

#define DECL_MOE_SW3D(NAME, BITS) \
kernel void NAME(device const half* x [[buffer(0)]], device const int* top_ids [[buffer(1)]], \
    device const uint32_t* wg [[buffer(2)]], device const ushort* sg_a [[buffer(3)]], device const ushort* bg_a [[buffer(4)]], \
    device const uint32_t* wu [[buffer(5)]], device const ushort* su_a [[buffer(6)]], device const ushort* bu_a [[buffer(7)]], \
    device half* out [[buffer(8)]], constant MoEProjParams& p [[buffer(9)]], \
    uint3 tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ moe_swiglu_3d_impl<BITS>(x, top_ids, wg, sg_a, bg_a, wu, su_a, bu_a, out, p, tg_id, sg, lane); }

DECL_MOE_SW3D(moe_swiglu_2b, 2)
DECL_MOE_SW3D(moe_swiglu_3b, 3)
DECL_MOE_SW3D(moe_swiglu_4b, 4)

// 7D. High-Occupancy 3D 4-Row Contiguous-Expert MoE Down: grid = ((N+15)/16, top_k, M), 4-Byte Safe!
template <int BITS>
inline void moe_down_3d_impl(
    device const half* moe_h, device const int* top_ids,
    device const uint32_t* wd, device const ushort* sd, device const ushort* bd,
    device half* moe_out, constant MoEProjParams& p, uint3 tg_id, uint sg, uint lane)
{
    const int row0 = (int(tg_id.x) * 4 + int(sg)) * 4;
    const int k_i  = int(tg_id.y);
    const int m    = int(tg_id.z);
    if (row0 >= p.N || k_i >= p.top_k || m >= p.M) return;

    const int r0 = row0, r1 = min(row0 + 1, p.N - 1), r2 = min(row0 + 2, p.N - 1), r3 = min(row0 + 3, p.N - 1);
    const ulong e = ulong(top_ids[m * p.top_k + k_i]);
    const int K = p.K, gsh = p.gs_shift;
    const ulong kg = ulong(K >> gsh);
    const ulong so0 = e * p.stride_s + ulong(r0) * kg, so1 = e * p.stride_s + ulong(r1) * kg;
    const ulong so2 = e * p.stride_s + ulong(r2) * kg, so3 = e * p.stride_s + ulong(r3) * kg;
    const device half* hk = moe_h + (m * p.top_k + k_i) * K;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (BITS == 4) {
        const ulong kw = ulong(K >> 3);
        const ulong wo0 = e * p.stride_w + ulong(r0) * kw, wo1 = e * p.stride_w + ulong(r1) * kw;
        const ulong wo2 = e * p.stride_w + ulong(r2) * kw, wo3 = e * p.stride_w + ulong(r3) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const half2 hs0 = half2(load_sb_h(sd, so0+g)), hb0 = half2(load_sb_h(bd, so0+g));
            const half2 hs1 = half2(load_sb_h(sd, so1+g)), hb1 = half2(load_sb_h(bd, so1+g));
            const half2 hs2 = half2(load_sb_h(sd, so2+g)), hb2 = half2(load_sb_h(bd, so2+g));
            const half2 hs3 = half2(load_sb_h(sd, so3+g)), hb3 = half2(load_sb_h(bd, so3+g));
            const device half4* xp = (const device half4*)(hk + gk);
            a0 += dot_4bit_word(wd[wo0+wi], hs0, hb0, xp) + dot_4bit_word(wd[wo0+wi+1], hs0, hb0, xp + 2);
            a1 += dot_4bit_word(wd[wo1+wi], hs1, hb1, xp) + dot_4bit_word(wd[wo1+wi+1], hs1, hb1, xp + 2);
            a2 += dot_4bit_word(wd[wo2+wi], hs2, hb2, xp) + dot_4bit_word(wd[wo2+wi+1], hs2, hb2, xp + 2);
            a3 += dot_4bit_word(wd[wo3+wi], hs3, hb3, xp) + dot_4bit_word(wd[wo3+wi+1], hs3, hb3, xp + 2);
        }
    } else if (BITS == 3) {
        const ulong kw = ulong((K * 3) >> 5);
        const ulong wo0 = e * p.stride_w + ulong(r0) * kw, wo1 = e * p.stride_w + ulong(r1) * kw;
        const ulong wo2 = e * p.stride_w + ulong(r2) * kw, wo3 = e * p.stride_w + ulong(r3) * kw;
        for (int k0 = 0; k0 < K; k0 += 1024) {
            const int gk = k0 + int(lane) * 32;
            if (gk >= K) break;
            const int blk = (gk >> 5) * 3, g = gk >> gsh;
            const device half4* xp = (const device half4*)(hk + gk);
            a0 += dot_3bit_block32(wd[wo0+blk], wd[wo0+blk+1], wd[wo0+blk+2], load_sb_f(sd, so0+g), load_sb_f(bd, so0+g), xp);
            a1 += dot_3bit_block32(wd[wo1+blk], wd[wo1+blk+1], wd[wo1+blk+2], load_sb_f(sd, so1+g), load_sb_f(bd, so1+g), xp);
            a2 += dot_3bit_block32(wd[wo2+blk], wd[wo2+blk+1], wd[wo2+blk+2], load_sb_f(sd, so2+g), load_sb_f(bd, so2+g), xp);
            a3 += dot_3bit_block32(wd[wo3+blk], wd[wo3+blk+1], wd[wo3+blk+2], load_sb_f(sd, so3+g), load_sb_f(bd, so3+g), xp);
        }
    } else {
        const ulong kw = ulong(K >> 4);
        const ulong wo0 = e * p.stride_w + ulong(r0) * kw, wo1 = e * p.stride_w + ulong(r1) * kw;
        const ulong wo2 = e * p.stride_w + ulong(r2) * kw, wo3 = e * p.stride_w + ulong(r3) * kw;
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int kwi = gk >> 4, g = gk >> gsh;
            const device half4* xp = (const device half4*)(hk + gk);
            a0 += dot_2bit_word(wd[wo0+kwi], half2(load_sb_h(sd, so0+g)), half2(load_sb_h(bd, so0+g)), xp);
            a1 += dot_2bit_word(wd[wo1+kwi], half2(load_sb_h(sd, so1+g)), half2(load_sb_h(bd, so1+g)), xp);
            a2 += dot_2bit_word(wd[wo2+kwi], half2(load_sb_h(sd, so2+g)), half2(load_sb_h(bd, so2+g)), xp);
            a3 += dot_2bit_word(wd[wo3+kwi], half2(load_sb_h(sd, so3+g)), half2(load_sb_h(bd, so3+g)), xp);
        }
    }
    const float v0 = simd_sum(a0), v1 = simd_sum(a1), v2 = simd_sum(a2), v3 = simd_sum(a3);
    const int o_base = (m * p.top_k + k_i) * p.N;
    if (lane == 0) {
        moe_out[o_base + r0] = half(v0);
        if (row0 + 1 < p.N) moe_out[o_base + r1] = half(v1);
        if (row0 + 2 < p.N) moe_out[o_base + r2] = half(v2);
        if (row0 + 3 < p.N) moe_out[o_base + r3] = half(v3);
    }
}

#define DECL_MOE_DN3D(NAME, BITS) \
kernel void NAME(device const half* moe_h [[buffer(0)]], device const int* top_ids [[buffer(1)]], \
    device const uint32_t* wd [[buffer(2)]], device const ushort* sd [[buffer(3)]], device const ushort* bd [[buffer(4)]], \
    device half* moe_out [[buffer(5)]], constant MoEProjParams& p [[buffer(6)]], \
    uint3 tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ moe_down_3d_impl<BITS>(moe_h, top_ids, wd, sd, bd, moe_out, p, tg_id, sg, lane); }

DECL_MOE_DN3D(moe_down_2d_2b, 2)
DECL_MOE_DN3D(moe_down_2d_3b, 3)
DECL_MOE_DN3D(moe_down_2d_4b, 4)

// 7E. 2D Vectorized half4 MoE Weighted Combine + Shared Expert + Fused Residual
kernel void moe_combine_res(
    device const half4* moe_out [[buffer(0)]], device const float* top_w [[buffer(1)]],
    device const half4* sh_out [[buffer(2)]], device const float* sh_gate [[buffer(3)]],
    device half4* x_res [[buffer(4)]], constant TopKParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int d4 = int(gid.x), m = int(gid.y);
    if (d4 >= p.E || m >= p.M) return;
    float4 acc = sh_gate[m] * float4(sh_out[m * p.E + d4]);
    for (int k = 0; k < p.top_k; ++k) {
        acc = fma(float4(top_w[m * p.top_k + k]), float4(moe_out[(m * p.top_k + k) * p.E + d4]), acc);
    }
    x_res[m * p.E + d4] = half4(float4(x_res[m * p.E + d4]) + acc);
}

// ─────────────────────── 8. Repetition penalty PER BARIS + Argmax + Top-64 exact ───────────────────────
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

// ─────────────────────── Probe bandwidth: baca buffer besar sekali, ukur GB/s nyata ───────────────────────
kernel void bw_probe(device const float4* src [[buffer(0)]], device float* out [[buffer(1)]],
                     constant uint& n4 [[buffer(2)]],
                     uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = gid; i < n4; i += gsz) acc += src[i];
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}

// ─────────────────────── Attention decode split-sequence (flash-decoding, 2 pass) ───────────────────────
// Kernel lama: 1 threadgroup per head, 8 simdgroup masing-masing memproses seq/8 token SECARA BERURUTAN
// (tiap token = simd_sum + exp yang saling bergantung). Di konteks 5k hanya 16 head × 8 SG aktif di seluruh
// GPU → decode 1.7B turun 132 → 66 tok/s. Di sini sekuens dibagi ke n_blk threadgroup per head (pass 1),
// lalu hasil parsial digabung dengan log-sum-exp (pass 2). Matematika sama; hanya urutan penjumlahan beda.
struct AttnSplitParams { int M, n_heads, n_kv_heads, pos, max_seq, n_blk; };

template <int HD>
inline void attn_split_impl(device const half* q_buf, device const half* k_cache, device const half* v_cache,
                            device float* part_ml, device float* part_o, constant AttnSplitParams& p,
                            threadgroup float* sm_m, threadgroup float* sm_s, threadgroup float* sm_o,
                            uint3 tg, uint sg, uint lane)
{
    constexpr int E = HD / 32;
    const int h = int(tg.x), b = int(tg.y), m = int(tg.z);
    if (h >= p.n_heads || b >= p.n_blk || m >= p.M) return;
    const int kv_h = h / (p.n_heads / p.n_kv_heads), seq_len = p.pos + m + 1;
    const int bl = (seq_len + p.n_blk - 1) / p.n_blk;
    const int b0 = min(b * bl, seq_len), b1 = min(b0 + bl, seq_len);
    const int sl = ((b1 - b0) + 7) >> 3, t0 = b0 + int(sg) * sl, t1 = min(t0 + sl, b1);
    const float scale = rsqrt(float(HD)) * LOG2E;
    const int qb = (m * p.n_heads + h) * HD + int(lane) * E;
    float q[E], o[E];
    #pragma unroll
    for (int e = 0; e < E; ++e) { q[e] = float(q_buf[qb + e]) * scale; o[e] = 0.0f; }
    float ms = -1e30f, se = 0.0f;
    const ulong base = ulong(kv_h) * ulong(p.max_seq) * ulong(HD) + ulong(lane) * ulong(E);
    for (int t = t0; t < t1; ++t) {
        const ulong ko = base + ulong(t) * ulong(HD);
        float d = 0.0f;
        #pragma unroll
        for (int e = 0; e < E; ++e) d += q[e] * float(k_cache[ko + e]);
        const float s = simd_sum(d);
        const float nm = max(ms, s), al = (se > 0.0f) ? exp2(ms - nm) : 0.0f, w = exp2(s - nm);
        se = se * al + w;
        #pragma unroll
        for (int e = 0; e < E; ++e) o[e] = o[e] * al + w * float(v_cache[ko + e]);
        ms = nm;
    }
    // sm_m/sm_s [8], sm_o [8*HD]: dideklarasikan di kernel (threadgroup hanya boleh di scope kernel)
    if (lane == 0) { sm_m[sg] = ms; sm_s[sg] = se; }
    #pragma unroll
    for (int e = 0; e < E; ++e) sm_o[int(sg) * HD + int(lane) * E + e] = o[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float gm = -1e30f;
        for (int s = 0; s < 8; ++s) if (sm_s[s] > 0.0f) gm = max(gm, sm_m[s]);
        float gs = 0.0f, acc[E];
        #pragma unroll
        for (int e = 0; e < E; ++e) acc[e] = 0.0f;
        for (int s = 0; s < 8; ++s) if (sm_s[s] > 0.0f) {
            const float ws = exp2(sm_m[s] - gm);
            gs += sm_s[s] * ws;
            #pragma unroll
            for (int e = 0; e < E; ++e) acc[e] += sm_o[s * HD + int(lane) * E + e] * ws;
        }
        const int pi = (m * p.n_heads + h) * p.n_blk + b;
        if (lane == 0) { part_ml[pi * 2] = gm; part_ml[pi * 2 + 1] = gs; }
        #pragma unroll
        for (int e = 0; e < E; ++e) part_o[ulong(pi) * HD + int(lane) * E + e] = acc[e];
    }
}

template <int HD, bool GATED>
inline void attn_merge_impl(device const float* part_ml, device const float* part_o, device const half* gate_buf,
                            device half* attn_out, constant AttnSplitParams& p, uint2 tg, uint lane)
{
    constexpr int E = HD / 32;
    const int h = int(tg.x), m = int(tg.y);
    if (h >= p.n_heads || m >= p.M) return;
    const int base = (m * p.n_heads + h) * p.n_blk;
    float gm = -1e30f;
    for (int b = 0; b < p.n_blk; ++b) if (part_ml[(base + b) * 2 + 1] > 0.0f) gm = max(gm, part_ml[(base + b) * 2]);
    float gs = 0.0f, acc[E];
    #pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.0f;
    for (int b = 0; b < p.n_blk; ++b) {
        const float s = part_ml[(base + b) * 2 + 1];
        if (s > 0.0f) {
            const float ws = exp2(part_ml[(base + b) * 2] - gm);
            gs += s * ws;
            #pragma unroll
            for (int e = 0; e < E; ++e) acc[e] += part_o[ulong(base + b) * HD + int(lane) * E + e] * ws;
        }
    }
    const float inv = 1.0f / max(gs, 1e-12f);
    const int qb = (m * p.n_heads + h) * HD + int(lane) * E;
    #pragma unroll
    for (int e = 0; e < E; ++e) {
        float v = acc[e] * inv;
        if (GATED) { const float g = float(gate_buf[qb + e]); v = v / (1.0f + exp(-g)); }
        attn_out[qb + e] = half(v);
    }
}

#define DECL_ATTN_SPLIT(NAME, HD) \
kernel void NAME(device const half* q [[buffer(0)]], device const half* kc [[buffer(1)]], device const half* vc [[buffer(2)]], \
    device float* pml [[buffer(3)]], device float* po [[buffer(4)]], constant AttnSplitParams& p [[buffer(5)]], \
    uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) \
{ \
    threadgroup float sm_m[8], sm_s[8], sm_o[8 * HD]; \
    attn_split_impl<HD>(q, kc, vc, pml, po, p, sm_m, sm_s, sm_o, tg, sg, lane); \
}
#define DECL_ATTN_MERGE(NAME, HD, GATED) \
kernel void NAME(device const float* pml [[buffer(0)]], device const float* po [[buffer(1)]], device const half* gate [[buffer(2)]], \
    device half* out [[buffer(3)]], constant AttnSplitParams& p [[buffer(4)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) \
{ attn_merge_impl<HD, GATED>(pml, po, gate, out, p, tg, lane); }
DECL_ATTN_SPLIT(attn_split_256, 256)
DECL_ATTN_MERGE(attn_merge_256_gated, 256, true)


// ═════════════ 1.7.0.dev10: expert MoE berkelompok (prefill, 4-bit, bit-identik) ═════════════
// moe_group: 1 threadgroup. Hitung jumlah pasangan per expert, offset (prefix), dan daftar pasangan per expert
// berurutan menurut id pasangan (deterministik, tanpa ketergantungan urutan atomik).
kernel void moe_group(
    device const int* top_ids [[buffer(0)]], device int* g_cnt [[buffer(1)]], device int* g_off [[buffer(2)]],
    device int* g_pairs [[buffer(3)]], constant TopKParams& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]])
{
    // 1.7.0.dev11: satu thread per expert memindai semua pasangan berurutan (P ≤ MAX_P·16) → deterministik,
    // tanpa salinan id di threadgroup memory (muat untuk chunk 512 × top-k 16).
    threadgroup int cnt[512];
    const int P = p.M * p.top_k, E = p.E;
    for (int ex = int(tid); ex < E; ex += int(ntg)) {
        int c = 0;
        for (int i = 0; i < P; ++i) c += (top_ids[i] == ex) ? 1 : 0;
        cnt[ex] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int s = 0;
        for (int ex = 0; ex < E; ++ex) { const int c = cnt[ex]; g_off[ex] = s; g_cnt[ex] = c; cnt[ex] = s; s += c; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int ex = int(tid); ex < E; ex += int(ntg)) {
        int w = cnt[ex];
        for (int i = 0; i < P; ++i) if (top_ids[i] == ex) g_pairs[w++] = i;
    }
}

// Gate+Up berkelompok: grid = ((N+7)/8, E). Baris sama dengan moe_swiglu_3d_impl (2 baris/simdgroup),
// token diproses per ubin 4: bobot & skala dibaca sekali per langkah K untuk 4 token.
kernel void moe_swiglu_g4(
    device const half* x [[buffer(0)]], device const int* g_cnt [[buffer(1)]], device const int* g_off [[buffer(2)]],
    device const int* g_pairs [[buffer(3)]],
    device const uint32_t* wg [[buffer(4)]], device const ushort* sg_a [[buffer(5)]], device const ushort* bg_a [[buffer(6)]],
    device const uint32_t* wu [[buffer(7)]], device const ushort* su_a [[buffer(8)]], device const ushort* bu_a [[buffer(9)]],
    device half* out [[buffer(10)]], constant MoEProjParams& p [[buffer(11)]],
    uint3 tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int row0 = (int(tg_id.x) * 4 + int(sg)) * 2;
    const int ei = int(tg_id.y);
    if (row0 >= p.N) return;
    const int c = g_cnt[ei];
    if (c == 0) return;
    const int base = g_off[ei];
    const int r0 = row0, r1 = min(row0 + 1, p.N - 1);
    const ulong e = ulong(ei);
    const int K = p.K, gsh = p.gs_shift;
    const ulong kg = ulong(K >> gsh);
    const ulong so0 = e * p.stride_s + ulong(r0) * kg, so1 = e * p.stride_s + ulong(r1) * kg;
    const ulong kw = ulong(K >> 3);
    const ulong wo0 = e * p.stride_w + ulong(r0) * kw, wo1 = e * p.stride_w + ulong(r1) * kw;

    for (int t0 = 0; t0 < c; t0 += 4) {
        int pr[4];
        const device half* xm[4];
        for (int t = 0; t < 4; ++t) {
            pr[t] = g_pairs[base + min(t0 + t, c - 1)];
            xm[t] = x + (pr[t] / p.top_k) * K;
        }
        float ag0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, ag1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float au0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, au1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const half2 hsg0 = half2(load_sb_h(sg_a, so0+g)), hbg0 = half2(load_sb_h(bg_a, so0+g));
            const half2 hsg1 = half2(load_sb_h(sg_a, so1+g)), hbg1 = half2(load_sb_h(bg_a, so1+g));
            const half2 hsu0 = half2(load_sb_h(su_a, so0+g)), hbu0 = half2(load_sb_h(bu_a, so0+g));
            const half2 hsu1 = half2(load_sb_h(su_a, so1+g)), hbu1 = half2(load_sb_h(bu_a, so1+g));
            const uint32_t g0a = wg[wo0+wi], g0b = wg[wo0+wi+1], g1a = wg[wo1+wi], g1b = wg[wo1+wi+1];
            const uint32_t u0a = wu[wo0+wi], u0b = wu[wo0+wi+1], u1a = wu[wo1+wi], u1b = wu[wo1+wi+1];
            for (int t = 0; t < 4; ++t) {
                const device half4* xp = (const device half4*)(xm[t] + gk);
                ag0[t] += dot_4bit_word(g0a, hsg0, hbg0, xp) + dot_4bit_word(g0b, hsg0, hbg0, xp + 2);
                ag1[t] += dot_4bit_word(g1a, hsg1, hbg1, xp) + dot_4bit_word(g1b, hsg1, hbg1, xp + 2);
                au0[t] += dot_4bit_word(u0a, hsu0, hbu0, xp) + dot_4bit_word(u0b, hsu0, hbu0, xp + 2);
                au1[t] += dot_4bit_word(u1a, hsu1, hbu1, xp) + dot_4bit_word(u1b, hsu1, hbu1, xp + 2);
            }
        }
        for (int t = 0; t < 4; ++t) {
            if (t0 + t >= c) break;                     // seragam dalam simdgroup
            const float gv0 = simd_sum(ag0[t]), uv0 = simd_sum(au0[t]);
            const float gv1 = simd_sum(ag1[t]), uv1 = simd_sum(au1[t]);
            const int o_base = pr[t] * p.N;
            if (lane == 0) {
                out[o_base + r0] = half((gv0 / (1.0f + exp(-gv0))) * uv0);
                if (row0 + 1 < p.N) out[o_base + r1] = half((gv1 / (1.0f + exp(-gv1))) * uv1);
            }
        }
    }
}

// Down berkelompok: grid = ((N+15)/16, E). Baris sama dengan moe_down_3d_impl (4 baris/simdgroup), ubin 4 token.
kernel void moe_down_g4(
    device const half* moe_h [[buffer(0)]], device const int* g_cnt [[buffer(1)]], device const int* g_off [[buffer(2)]],
    device const int* g_pairs [[buffer(3)]],
    device const uint32_t* wd [[buffer(4)]], device const ushort* sd [[buffer(5)]], device const ushort* bd [[buffer(6)]],
    device half* moe_out [[buffer(7)]], constant MoEProjParams& p [[buffer(8)]],
    uint3 tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]])
{
    const int row0 = (int(tg_id.x) * 4 + int(sg)) * 4;
    const int ei = int(tg_id.y);
    if (row0 >= p.N) return;
    const int c = g_cnt[ei];
    if (c == 0) return;
    const int base = g_off[ei];
    const int r0 = row0, r1 = min(row0 + 1, p.N - 1), r2 = min(row0 + 2, p.N - 1), r3 = min(row0 + 3, p.N - 1);
    const ulong e = ulong(ei);
    const int K = p.K, gsh = p.gs_shift;
    const ulong kg = ulong(K >> gsh);
    const ulong so0 = e * p.stride_s + ulong(r0) * kg, so1 = e * p.stride_s + ulong(r1) * kg;
    const ulong so2 = e * p.stride_s + ulong(r2) * kg, so3 = e * p.stride_s + ulong(r3) * kg;
    const ulong kw = ulong(K >> 3);
    const ulong wo0 = e * p.stride_w + ulong(r0) * kw, wo1 = e * p.stride_w + ulong(r1) * kw;
    const ulong wo2 = e * p.stride_w + ulong(r2) * kw, wo3 = e * p.stride_w + ulong(r3) * kw;

    for (int t0 = 0; t0 < c; t0 += 4) {
        int pr[4];
        const device half* hk[4];
        for (int t = 0; t < 4; ++t) {
            pr[t] = g_pairs[base + min(t0 + t, c - 1)];
            hk[t] = moe_h + pr[t] * K;
        }
        float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float a2[4] = {0.0f, 0.0f, 0.0f, 0.0f}, a3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k0 = 0; k0 < K; k0 += 512) {
            const int gk = k0 + int(lane) * 16;
            if (gk >= K) break;
            const int wi = gk >> 3, g = gk >> gsh;
            const half2 hs0 = half2(load_sb_h(sd, so0+g)), hb0 = half2(load_sb_h(bd, so0+g));
            const half2 hs1 = half2(load_sb_h(sd, so1+g)), hb1 = half2(load_sb_h(bd, so1+g));
            const half2 hs2 = half2(load_sb_h(sd, so2+g)), hb2 = half2(load_sb_h(bd, so2+g));
            const half2 hs3 = half2(load_sb_h(sd, so3+g)), hb3 = half2(load_sb_h(bd, so3+g));
            const uint32_t w0a = wd[wo0+wi], w0b = wd[wo0+wi+1], w1a = wd[wo1+wi], w1b = wd[wo1+wi+1];
            const uint32_t w2a = wd[wo2+wi], w2b = wd[wo2+wi+1], w3a = wd[wo3+wi], w3b = wd[wo3+wi+1];
            for (int t = 0; t < 4; ++t) {
                const device half4* xp = (const device half4*)(hk[t] + gk);
                a0[t] += dot_4bit_word(w0a, hs0, hb0, xp) + dot_4bit_word(w0b, hs0, hb0, xp + 2);
                a1[t] += dot_4bit_word(w1a, hs1, hb1, xp) + dot_4bit_word(w1b, hs1, hb1, xp + 2);
                a2[t] += dot_4bit_word(w2a, hs2, hb2, xp) + dot_4bit_word(w2b, hs2, hb2, xp + 2);
                a3[t] += dot_4bit_word(w3a, hs3, hb3, xp) + dot_4bit_word(w3b, hs3, hb3, xp + 2);
            }
        }
        for (int t = 0; t < 4; ++t) {
            if (t0 + t >= c) break;
            const float v0 = simd_sum(a0[t]), v1 = simd_sum(a1[t]), v2 = simd_sum(a2[t]), v3 = simd_sum(a3[t]);
            const int o_base = pr[t] * p.N;
            if (lane == 0) {
                moe_out[o_base + r0] = half(v0);
                if (row0 + 1 < p.N) moe_out[o_base + r1] = half(v1);
                if (row0 + 2 < p.N) moe_out[o_base + r2] = half(v2);
                if (row0 + 3 < p.N) moe_out[o_base + r3] = half(v3);
            }
        }
    }
}



// Fase 0: kernel kosong untuk mengukur biaya dispatch/barrier murni (dipakai hanya oleh bench_overhead)
kernel void null_kernel() {}
