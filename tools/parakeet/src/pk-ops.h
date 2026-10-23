// tools/parakeet/src/pk-ops.h - fp32 helpers (NEON with a scalar fallback)
// and the int8 GEMM driver used by the encoder.
#pragma once

#include "pk-kernels.h"
#include "pk-pool.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pk {

// ---- int8 activations ------------------------------------------------------

// N tokens x K in the token-pair layout of pk-kernels.h, padded to an even
// number of tokens.
struct qact {
    std::vector<int8_t> buf;
    std::vector<float> scale;
    int N = 0, K = 0, Npad = 0;
    // the fp32 source (diagnostics only: PK_DEBUG_F32_ACTS=1 runs the GEMMs
    // on it instead of the int8 copy; it must still be alive at GEMM time)
    const float * src = nullptr;
    size_t ldsrc = 0;
    size_t pair_stride() const { return (size_t) K * 2; }
};

// Symmetric per-token int8 quantisation, round half to even.
void quantize_acts(pool & P, const float * X, int N, int K, size_t ldx, qact & out);

// ---- int8 weights ----------------------------------------------------------

struct wmat {
    int O = 0, K = 0;
    // five-value: codes in `fv5-pair16-v1` or, with fv5t7, `fv5-b5x3t7-v1`;
    // one 16-byte LUT per row pair either way
    const uint8_t * fv5 = nullptr;
    bool fv5t7 = false;
    std::vector<int8_t> lut;
    // int tables: weight panels prepared at load ([O/8][K/8][4][16])
    std::vector<int8_t> panels;
    std::vector<float> ws;          // per-row dequant scale
    const float * bias = nullptr;   // optional
    // max |grid lo - lo| / hi over the rows (fv5); hi is exact
    double grid_err = 0.0;
};

void wmat_from_fv5(wmat & w, const uint8_t * codes, const float * lohi, int O, int K, bool t7 = false);
// decode the 8-row block `rb` of a five-value matrix into a weight panel
void decode_fv5_block(const kernel_table & kt, const wmat & W, int rb, int8_t * panel);
// row-major int8 [O][K] with per-row scales; `perm` (optional) permutes the
// input columns: new column j reads old column perm[j].
void wmat_from_int8(wmat & w, const int8_t * q, const float * scale, int O, int K, const int * perm = nullptr);

enum class epi : int { store, add, silu, relu };

// Y[t][o] (ldy) <- op(acc * ws[o] * xs[t] + bias[o]); for epi::add,
// Y += alpha * value.
void gemm_q8(pool & P, const kernel_table & kt, const wmat & W, const qact & X, float * Y, size_t ldy, epi op,
             float alpha = 1.0f);

// ---- fp32 ------------------------------------------------------------------

void layernorm(const float * x, float * y, int N, int D, const float * g, const float * b, float eps);
void silu_inplace(float * x, size_t n);
float sigmoidf_(float x);
void exp_inplace(float * x, size_t n);   // vectorised expf
// C[m][n] = alpha * sum_k A[m][k] * B[n][k]   (A: M x K, lda; B: N x K, ldb)
void gemm_f32_nt(const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, int M, int N, int K,
                 float alpha);

}  // namespace pk
