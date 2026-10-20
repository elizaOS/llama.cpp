// tools/parakeet/src/pk-kernels-avx2.cpp - x86_64 AVX2 tier.
// Compiled with -mavx2 (/arch:AVX2 on MSVC); only called after a runtime
// CPUID + XGETBV check (cpu_has_avx2 in pk-kernels.cpp).
//
// The int8 operands are sign-extended to int16 and multiplied with VPMADDWD,
// which is exact for every int8 input (no VPMADDUBSW saturation and no
// -128 corner case), so this tier is bit-identical to the scalar reference.
#include "pk-kernels.h"

#if PK_HAVE_X86_EXT
#include <immintrin.h>

#include <cstring>

namespace pk {
namespace {

// Same nibble -> level lookup as the NEON tiers: PSHUFB with a 16-byte table
// (the nibble indices are 0..15, so the high-bit zeroing never applies).
void decode_fv5_avx2(const uint8_t * codes, size_t pair_stride, const int8_t * luts, int K, int8_t * panel) {
    const int G = K / 16;
    const __m128i m = _mm_set1_epi8(0x0F);
    for (int p = 0; p < 4; ++p) {
        const __m128i lut = _mm_loadu_si128((const __m128i *) (luts + p * 16));
        const uint8_t * src = codes + (size_t) p * pair_stride;
        int8_t * dst = panel + p * 16;
        for (int g = 0; g < G; ++g) {
            const __m128i b = _mm_loadu_si128((const __m128i *) (src + g * 16));
            const __m128i lo = _mm_and_si128(b, m);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m);
            _mm_storeu_si128((__m128i *) (dst + (size_t) (2 * g) * 64), _mm_shuffle_epi8(lut, lo));
            _mm_storeu_si128((__m128i *) (dst + (size_t) (2 * g + 1) * 64), _mm_shuffle_epi8(lut, hi));
        }
    }
}

// `fv5-b5x3t7-v1` with the NEON tier's arithmetic: q = ((x + 1) * 51) >> 8
// on 16-bit lanes, digits by subtraction, then the pair16 PSHUFB lookup.
inline __m128i fv5t7_div5(__m128i x) {
    const __m128i z = _mm_setzero_si128(), k51 = _mm_set1_epi16(51), one = _mm_set1_epi16(1);
    const __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_add_epi16(_mm_unpacklo_epi8(x, z), one), k51), 8);
    const __m128i hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_add_epi16(_mm_unpackhi_epi8(x, z), one), k51), 8);
    return _mm_packus_epi16(lo, hi);
}

inline __m128i fv5t7_times5(__m128i q) {
    const __m128i q2 = _mm_add_epi8(q, q);
    return _mm_add_epi8(_mm_add_epi8(q2, q2), q);
}

void decode_fv5t7_avx2(const uint8_t * codes, const int8_t * luts, int K, int8_t * panel) {
    const __m128i lut[4] = { _mm_loadu_si128((const __m128i *) luts), _mm_loadu_si128((const __m128i *) (luts + 16)),
                             _mm_loadu_si128((const __m128i *) (luts + 32)), _mm_loadu_si128((const __m128i *) (luts + 48)) };
    const __m128i row_hi = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8);
    const __m128i m7 = _mm_set1_epi8(0x7F);
    const size_t pbytes = (size_t) K * 8;
    const int G = (fv5t7_codes(K) + 7) / 8;
    alignas(16) int8_t stage[384];
    for (int g = 0; g < G; ++g) {
        const uint8_t * s = codes + (size_t) g * 112;
        __m128i x[8];
        __m128i c7 = _mm_setzero_si128();
        for (int m = 0; m < 7; ++m) {
            const __m128i v = _mm_loadu_si128((const __m128i *) (s + m * 16));
            x[m] = _mm_and_si128(v, m7);
            // bit 7 of every byte -> bit m (a 16-bit shift only brings in
            // bits above m from the neighbouring byte, and those are masked)
            c7 = _mm_or_si128(c7, _mm_and_si128(_mm_srli_epi16(v, 7 - m), _mm_set1_epi8((char) (1 << m))));
        }
        x[7] = c7;
        const size_t off = (size_t) g * 384;
        const bool full = off + 384 <= pbytes;
        int8_t * dst = full ? panel + off : stage;
        for (int m = 0; m < 8; ++m) {
            const __m128i q1 = fv5t7_div5(x[m]);
            const __m128i q2 = fv5t7_div5(q1);
            const __m128i d0 = _mm_sub_epi8(x[m], fv5t7_times5(q1));
            const __m128i d1 = _mm_sub_epi8(q1, fv5t7_times5(q2));
            const int v0 = 3 * m;  // panel vector index within the group; row pair = v % 4
            _mm_storeu_si128((__m128i *) (dst + (size_t) v0 * 16), _mm_shuffle_epi8(lut[v0 % 4], _mm_or_si128(d0, row_hi)));
            _mm_storeu_si128((__m128i *) (dst + (size_t) (v0 + 1) * 16), _mm_shuffle_epi8(lut[(v0 + 1) % 4], _mm_or_si128(d1, row_hi)));
            _mm_storeu_si128((__m128i *) (dst + (size_t) (v0 + 2) * 16), _mm_shuffle_epi8(lut[(v0 + 2) % 4], _mm_or_si128(q2, row_hi)));
        }
        if (!full) {
            std::memcpy(panel + off, stage, pbytes - off);
        }
    }
}

inline __m256i load_s8_as_s16(const int8_t * p) {
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *) p));
}

// Sum the 4 int32 lanes of each 128-bit half: returns {sum(lo half), sum(hi half)}.
inline void hsum_halves(__m256i v, int32_t & lo, int32_t & hi) {
    const __m128i a = _mm256_castsi256_si128(v);
    const __m128i b = _mm256_extracti128_si256(v, 1);
    // [a0+a1, a2+a3, b0+b1, b2+b3] -> [a, b, a, b]
    __m128i s = _mm_hadd_epi32(a, b);
    s = _mm_hadd_epi32(s, s);
    lo = _mm_cvtsi128_si32(s);
    hi = _mm_extract_epi32(s, 1);
}

// One token pair against the 8-row panel. Per k chunk, row pair p is one
// 256-bit int16 vector [row 2p | row 2p+1] and the token pair is
// [token 2q | token 2q+1]; VPMADDWD against it (straight) gives
// [r0 t0 | r1 t1] and against the half-swapped tokens [r0 t1 | r1 t0],
// mirroring the dotprod tier. 8 accumulators + 4 weights + 2 tokens fit in
// the 16 YMM registers.
inline void pair_tile(const int8_t * panel, const int8_t * x, int C, int32_t * out, int t0) {
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
    __m256i w0 = s0, w1 = s0, w2 = s0, w3 = s0;
    for (int c = 0; c < C; ++c) {
        const int8_t * wp = panel + (size_t) c * 64;
        const __m256i v = load_s8_as_s16(x + (size_t) c * 16);
        const __m256i u = _mm256_permute4x64_epi64(v, 0x4E);
        const __m256i p0 = load_s8_as_s16(wp);
        const __m256i p1 = load_s8_as_s16(wp + 16);
        const __m256i p2 = load_s8_as_s16(wp + 32);
        const __m256i p3 = load_s8_as_s16(wp + 48);
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(p0, v)); w0 = _mm256_add_epi32(w0, _mm256_madd_epi16(p0, u));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(p1, v)); w1 = _mm256_add_epi32(w1, _mm256_madd_epi16(p1, u));
        s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(p2, v)); w2 = _mm256_add_epi32(w2, _mm256_madd_epi16(p2, u));
        s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(p3, v)); w3 = _mm256_add_epi32(w3, _mm256_madd_epi16(p3, u));
    }
    const __m256i S[4] = { s0, s1, s2, s3 };
    const __m256i W[4] = { w0, w1, w2, w3 };
    for (int p = 0; p < 4; ++p) {
        int32_t r0t0, r1t1, r0t1, r1t0;
        hsum_halves(S[p], r0t0, r1t1);
        hsum_halves(W[p], r0t1, r1t0);
        out[t0 * 8 + 2 * p] = r0t0;
        out[(t0 + 1) * 8 + 2 * p + 1] = r1t1;
        out[(t0 + 1) * 8 + 2 * p] = r0t1;
        out[t0 * 8 + 2 * p + 1] = r1t0;
    }
}

void tile_avx2(const int8_t * panel, const int8_t * x, size_t x_stride, int K, int np, int32_t * out) {
    const int C = K / 8;
    for (int q = 0; q < np; ++q) {
        pair_tile(panel, x + (size_t) q * x_stride, C, out, 2 * q);
    }
}

inline int32_t hsum8(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

void matvec_avx2(const int8_t * w, int O, int K, const int8_t * x, int32_t * y) {
    int o = 0;
    for (; o + 4 <= O; o += 4) {
        const int8_t * r0 = w + (size_t) o * K;
        const int8_t * r1 = r0 + K;
        const int8_t * r2 = r1 + K;
        const int8_t * r3 = r2 + K;
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        for (int k = 0; k < K; k += 16) {
            const __m256i xv = load_s8_as_s16(x + k);
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(load_s8_as_s16(r0 + k), xv));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(load_s8_as_s16(r1 + k), xv));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(load_s8_as_s16(r2 + k), xv));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(load_s8_as_s16(r3 + k), xv));
        }
        y[o] = hsum8(a0);
        y[o + 1] = hsum8(a1);
        y[o + 2] = hsum8(a2);
        y[o + 3] = hsum8(a3);
    }
    for (; o < O; ++o) {
        const int8_t * r0 = w + (size_t) o * K;
        __m256i a0 = _mm256_setzero_si256();
        for (int k = 0; k < K; k += 16) {
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(load_s8_as_s16(r0 + k), load_s8_as_s16(x + k)));
        }
        y[o] = hsum8(a0);
    }
}

}  // namespace

const kernel_table & kernels_avx2() {
    static const kernel_table t = { tier::avx2, "avx2", decode_fv5_avx2, decode_fv5t7_avx2, tile_avx2, matvec_avx2 };
    return t;
}

}  // namespace pk
#endif
