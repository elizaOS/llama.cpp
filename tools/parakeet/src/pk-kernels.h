// tools/parakeet/src/pk-kernels.h - int8 GEMM kernels for the five-value
// (fv5) encoder linears and the int6 tables, plus the dispatch table.
//
// Shared layouts (K is always a multiple of 16, rows a multiple of 8):
//
//   activations  [N/2 token pairs][K/8 chunks][16 bytes]
//                16 bytes = token 2q at k 8c..8c+7, then token 2q+1 at the
//                same k. Odd N is padded with a zero token. One fp32 scale
//                per token (symmetric, round-half-even).
//
//   weight panel [K/8 chunks][4 row pairs][16 bytes] per block of 8 rows
//                16 bytes = row 2p at k 8c..8c+7, then row 2p+1. This is the
//                operand order of SMMLA (2x8 by 2x8) and of a pair of SDOTs.
//
//   fv5 codes    `fv5-pair16-v1` (see convert_phonon2_to_gguf.py): per row
//                pair, per 16 k, 16 bytes; low nibbles give chunk 2g, high
//                nibbles chunk 2g+1, each already in panel order. A 16-byte
//                table per row pair maps the nibble to the int8 level:
//                0..4 -> row 2p {-H,-L,0,L,H}, 8..12 -> row 2p+1.
//
// Every tier computes exact int32 dot products over the same int8 operands,
// so all tiers are bit-identical (tests/test_parakeet_kernels.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

namespace pk {

enum class tier : int { scalar = 1, dotprod = 2, i8mm = 3 };

struct kernel_table {
    tier t;
    const char * name;
    // decode the fv5 codes of one 8-row block (4 row pairs starting at
    // `codes`, `pair_stride` bytes apart) into a weight panel.
    void (*decode_fv5)(const uint8_t * codes, size_t pair_stride, const int8_t * luts, int K, int8_t * panel);
    // int32 tile: 8 rows x 2*np tokens (np = 1..4 token pairs). `panel` is
    // the 8-row weight panel, `x` the first token pair, `x_stride` the bytes
    // between token pairs. out[t * 8 + r] (token-major, so the epilogue
    // reads 8 consecutive output rows of one token).
    void (*tile)(const int8_t * panel, const int8_t * x, size_t x_stride, int K, int np, int32_t * out);
    // int8 matvec for the decoder: y[o] = sum_k w[o][k] * x[k] for o in
    // [0, O) over row-major int8 weights (K % 16 == 0).
    void (*matvec)(const int8_t * w, int O, int K, const int8_t * x, int32_t * y);
};

// Best tier this CPU supports, or `want` when it is available.
const kernel_table & kernels_for(int want /*0 = best*/);
bool cpu_has_dotprod();
bool cpu_has_i8mm();

// per-tier tables (defined in their own translation units)
const kernel_table & kernels_scalar();
#if defined(__aarch64__) && !defined(PK_NO_ARM_EXT)
#define PK_HAVE_ARM_EXT 1
const kernel_table & kernels_dotprod();
const kernel_table & kernels_i8mm();
#endif

}  // namespace pk
