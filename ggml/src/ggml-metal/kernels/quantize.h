#pragma once

#include "common.h"

void quantize_q1_0(device const float * src, device block_q1_0 & dst) {
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    for (int j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK1_0; j++) {
        if (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_q2_0(device const float * src, device block_q2_0 & dst) {
    float amax = 0.0f;
    for (int j = 0; j < QK2_0; j++) {
        float a = fabs(src[j]);
        if (a > amax) amax = a;
    }
    const float d = amax;
    dst.d = d;

    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    for (int j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK2_0; j++) {
        int q = (int)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

void quantize_q4_0(device const float * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK4_0/2 + j]*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q4_1(device const float * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < QK4_1; j++) {
        const float v = src[j];
        if (min > v) min = v;
        if (max < v) max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK4_1/2 + j] - min)*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q5_0(device const float * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK5_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK5_0/2 + j]*id;

        const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q5_1(device const float * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    float max = src[0];
    float min = src[0];

    for (int j = 1; j < QK5_1; j++) {
        const float v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q8_0(device const float * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = src[j]*id;

        dst.qs[j] = round(x0);
    }
}

void quantize_iq4_nl(device const float * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_NL; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / kvalues_iq4nl_f[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = src[0        + j]*id;
        const float x1 = src[QK4_NL/2 + j]*id;

        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        const float v0 = kvalues_iq4nl_f[xi0];
        const float v1 = kvalues_iq4nl_f[xi1];
        const float w0 = src[0        + j]*src[0        + j];
        const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Match quantize_iq4_nl's weighted scale search for contiguous CPU copies.
void quantize_iq4_nl_full(device const float * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < QK4_NL; ++j) {
        if (fabs(src[j]) > amax) {
            amax = fabs(src[j]);
            max = src[j];
        }
    }
    if (amax < 1e-15f) {
        dst.d = 0.0h;
        for (int j = 0; j < QK4_NL/2; ++j) {
            dst.qs[j] = 0;
        }
        return;
    }

    float d = -max/kvalues_iq4nl_f[0];
    float id = 1.0f/d;
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int j = 0; j < QK4_NL; ++j) {
        const float q = kvalues_iq4nl_f[best_index_int8(16, kvalues_iq4nl_f, id*src[j])];
        const float w = src[j]*src[j];
        sumqx += w*q*src[j];
        sumq2 += w*q*q;
    }
    d = sumq2 > 0.0f ? sumqx/sumq2 : 0.0f;
    float best = d*sumqx;
    for (int itry = -7; itry <= 7; ++itry) {
        id = (itry + kvalues_iq4nl_f[0])/max;
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < QK4_NL; ++j) {
            const float q = kvalues_iq4nl_f[best_index_int8(16, kvalues_iq4nl_f, id*src[j])];
            const float w = src[j]*src[j];
            sumqx += w*q*src[j];
            sumq2 += w*q*q;
        }
        if (sumq2 > 0.0f && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2;
            best = d*sumqx;
        }
    }
    dst.d = d;
    id = d ? 1.0f/d : 0.0f;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const uint8_t lo = best_index_int8(16, kvalues_iq4nl_f, id*src[j]);
        const uint8_t hi = best_index_int8(16, kvalues_iq4nl_f, id*src[QK4_NL/2 + j]);
        dst.qs[j] = lo | (hi << 4);
    }
}

void quantize_tbq3_0(device const float * src, device block_tbq3_0 & dst) {
#pragma METAL fp math_mode(safe)
    thread float rotated[QK_TBQ];
    tbq_precondition_block_set_rows(src, rotated);

    float sumsq = 0.0f;
    for (int i = 0; i < QK_TBQ; ++i) {
        sumsq = fma(rotated[i], rotated[i], sumsq);
    }

    const float d = sqrt(sumsq / float(QK_TBQ));
    dst.d = (half) d;
    for (int i = 0; i < QK_TBQ * 3 / 8; ++i) {
        dst.qs[i] = 0;
    }

    if (d == 0.0f) {
        return;
    }

    const float id = 1.0f / d;
    for (int i = 0; i < QK_TBQ; ++i) {
        const uint8_t code = tbq_best_index_set_rows(k_tbq3_codebook_set_rows, 8, rotated[i] * id);
        tbq3_set_code_set_rows(dst.qs, i, code);
    }
}

void quantize_tbq4_0(device const float * src, device block_tbq4_0 & dst) {
#pragma METAL fp math_mode(safe)
    thread float rotated[QK_TBQ];
    tbq_precondition_block_set_rows(src, rotated);

    float sumsq = 0.0f;
    for (int i = 0; i < QK_TBQ; ++i) {
        sumsq = fma(rotated[i], rotated[i], sumsq);
    }

    const float d = sqrt(sumsq / float(QK_TBQ));
    dst.d = (half) d;
    for (int i = 0; i < QK_TBQ / 2; ++i) {
        dst.qs[i] = 0;
    }

    if (d == 0.0f) {
        return;
    }

    const float id = 1.0f / d;
    for (int i = 0; i < QK_TBQ; ++i) {
        const uint8_t code = tbq_best_index_set_rows(k_tbq4_codebook_set_rows, 16, rotated[i] * id);
        tbq4_set_code_set_rows(dst.qs, i, code);
    }
}

void quantize_q4_polar(device const float * src, device block_q4_polar & dst) {
#pragma METAL fp math_mode(safe)
    float sumsq = 0.0f;
    for (int i = 0; i < QK_POLAR; ++i) {
        sumsq = fma(src[i], src[i], sumsq);
    }

    const float l2 = sqrt(sumsq);
    const float inv_l2 = l2 > 1e-10f ? 1.0f / l2 : 0.0f;
    dst.d = (half) l2;

    thread float buf[QK_POLAR];
    for (int i = 0; i < QK_POLAR; ++i) {
        buf[i] = src[i] * inv_l2;
    }

    polar_hadamard128_set_rows(buf);

    thread uint8_t codes[QK_POLAR];
    for (int i = 0; i < QK_POLAR; ++i) {
        codes[i] = polar_q4_bucketize_set_rows(buf[i]);
    }

    for (int i = 0; i < QK_POLAR / 2; ++i) {
        const uint8_t lo = codes[2 * i];
        const uint8_t hi = codes[2 * i + 1];
        dst.qs[i] = (uint8_t) ((hi << 4) | (lo & 0x0F));
    }

    for (int i = 0; i < QJL_RESIDUAL_BYTES; ++i) {
        dst.qjl[i] = 0;
    }

    float proj = 0.0f;
    uint state = 42u;
    for (int i = 0; i < QK_POLAR; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const float sign = (state & 1u) ? 1.0f : -1.0f;
        const float c = k_polar_q4_centroids_set_rows[codes[i]];
        proj = fma(buf[i] - c, sign, proj);
    }
    dst.qjl[0] = proj >= 0.0f ? 1u : 0u;
}

void quantize_tq2_0(device const float * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK_K; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    for (int j = 0; j < QK_K/4; j += 32) {
        for (int m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (int n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = (int)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
