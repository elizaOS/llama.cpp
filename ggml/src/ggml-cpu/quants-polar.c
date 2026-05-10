/* quants-polar.c - Q4_POLAR CPU-side dispatch.
 *
 * Hosts the CPU `quantize_row_q4_polar` (currently == ref) and the
 * matmul kernel `ggml_vec_dot_q4_polar_q8_0` (uses dequantize_row_q4_polar
 * from ggml-base then accumulates against Q8_0 activations). Future
 * SIMD lands here without disturbing the reference path.
 */

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include <assert.h>

void quantize_row_q4_polar(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    /* Production path == reference path until SIMD encoder lands. */
    quantize_row_q4_polar_ref(x, (block_q4_polar *)y, k);
}

void ggml_vec_dot_q4_polar_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_POLAR == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    const block_q4_polar * GGML_RESTRICT x = (const block_q4_polar *) vx;
    const block_q8_0     * GGML_RESTRICT y = (const block_q8_0 *) vy;

    const int nb_polar       = n / QK_POLAR;
    const int n_q8_per_polar = QK_POLAR / QK8_0;     /* = 4 */

    float buf[QK_POLAR];
    double acc = 0.0;

    for (int b = 0; b < nb_polar; b++) {
        /* Materialise the dequantized PolarQuant block in fp32, then
         * sweep the 4 matching Q8_0 activation blocks and accumulate.
         */
        dequantize_row_q4_polar(x + b, buf, QK_POLAR);

        for (int qb = 0; qb < n_q8_per_polar; qb++) {
            const block_q8_0 * yb = y + b * n_q8_per_polar + qb;
            const float scale = GGML_CPU_FP16_TO_FP32(yb->d);
            const float * xchunk = buf + qb * QK8_0;

            float local = 0.0f;
            for (int i = 0; i < QK8_0; i++) {
                local += xchunk[i] * (float)yb->qs[i];
            }
            acc += (double)scale * (double)local;
        }
    }

    *s = (float)acc;
}
