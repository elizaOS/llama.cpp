/* fused-q4-polar-dot.h — internal declarations for the fused Q4_POLAR x
 * Q8_0 dot kernels. The scalar reference and each SIMD variant live in
 * their own translation unit so they can be compiled with -mavx2 /
 * -march=armv8-a etc. independently. They share one prototype set so
 * the dispatcher (fused-q4-polar-dot.c) and the SIMD definitions agree
 * and -Werror=missing-prototypes is satisfied on every target.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

double ggml_vec_dot_q4_polar_q8_0_fused_ref(int nb_polar,
                                            const block_q4_polar * x,
                                            const block_q8_0 * y,
                                            bool use_qjl);

#if defined(__AVX2__)
double ggml_vec_dot_q4_polar_q8_0_fused_avx2(int nb_polar,
                                             const block_q4_polar * x,
                                             const block_q8_0 * y,
                                             bool use_qjl);
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
double ggml_vec_dot_q4_polar_q8_0_fused_neon(int nb_polar,
                                             const block_q4_polar * x,
                                             const block_q8_0 * y,
                                             bool use_qjl);
#endif

#ifdef __cplusplus
}
#endif
