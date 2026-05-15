/* fused-attn-qjl-tbq.h — internal declarations for the per-ISA inner
 * kernels used by the fused QJL+TBQ attention op. The dispatcher in
 * fused-attn-qjl-tbq.c calls these and each SIMD variant defines them
 * in its own translation unit; sharing one prototype set keeps
 * -Werror=missing-prototypes happy on every target.
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#if defined(__AVX2__)
float qjl_score_one_avx2(const float * q_sketch, const uint8_t * signs);
void  fused_attn_v_mix_avx2(int n_tokens, const float * w,
                            const uint8_t * v_codes, const uint16_t * v_scales,
                            float * out);
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
float qjl_score_one_neon(const float * q_sketch, const uint8_t * signs);
void  fused_attn_v_mix_neon(int n_tokens, const float * w,
                            const uint8_t * v_codes, const uint16_t * v_scales,
                            float * out);
#endif

#ifdef __cplusplus
}
#endif
