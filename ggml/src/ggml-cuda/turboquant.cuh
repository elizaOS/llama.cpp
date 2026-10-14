#pragma once

#include "common.cuh"

static constexpr __device__ float k_tbq3_codebook_cuda[8] = {
    -2.1519457f, -1.3439093f, -0.7560053f, -0.2450942f,
     0.2450942f,  0.7560053f,  1.3439093f,  2.1519457f,
};

static constexpr __device__ float k_tbq4_codebook_cuda[16] = {
    -2.7321365f, -2.0685055f, -1.6175243f, -1.2557391f,
    -0.9419147f, -0.6564307f, -0.3878412f, -0.1283243f,
     0.1283243f,  0.3878412f,  0.6564307f,  0.9419147f,
     1.2557391f,  1.6175243f,  2.0685055f,  2.7321365f,
};

static constexpr __device__ int8_t k_tbq_signs_cuda[QK_TBQ] = {
     1, -1,  1,  1, -1,  1, -1, -1,
     1,  1, -1,  1, -1, -1,  1, -1,
    -1,  1,  1, -1,  1, -1, -1,  1,
     1, -1,  1, -1, -1,  1, -1,  1,
};

static __device__ __forceinline__ uint8_t tbq3_get_code_cuda(const uint8_t * qs, int idx) {
    const int bit = idx * 3;
    const int byte = bit >> 3;
    const int shift = bit & 7;

    uint32_t bits = qs[byte] >> shift;
    if (shift > 5 && byte + 1 < (QK_TBQ * 3 / 8)) {
        bits |= (uint32_t) qs[byte + 1] << (8 - shift);
    }

    return bits & 0x7u;
}

static __device__ __forceinline__ void tbq3_set_code_cuda(uint8_t * qs, int idx, uint8_t code) {
    const int bit = idx * 3;
    const int byte = bit >> 3;
    const int shift = bit & 7;

    qs[byte] = (uint8_t) (qs[byte] | ((code & 0x7u) << shift));
    if (shift > 5 && byte + 1 < (QK_TBQ * 3 / 8)) {
        qs[byte + 1] = (uint8_t) (qs[byte + 1] | ((code & 0x7u) >> (8 - shift)));
    }
}

static __device__ __forceinline__ uint8_t tbq4_get_code_cuda(const uint8_t * qs, int idx) {
    const int j = idx % (QK_TBQ/2);
    return idx < QK_TBQ/2 ? (qs[j] & 0x0F) : (qs[j] >> 4);
}

static __device__ __forceinline__ void tbq4_set_code_cuda(uint8_t * qs, int idx, uint8_t code) {
    const int j = idx % (QK_TBQ/2);
    if (idx < QK_TBQ/2) {
        qs[j] = (uint8_t) ((qs[j] & 0xF0) | (code & 0x0F));
    } else {
        qs[j] = (uint8_t) ((qs[j] & 0x0F) | ((code & 0x0F) << 4));
    }
}

static __device__ __forceinline__ void tbq_hadamard32_cuda(float * x) {
    for (int len = 1; len < QK_TBQ; len <<= 1) {
        for (int i = 0; i < QK_TBQ; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }

    const float norm = 0.1767766952966369f;
    for (int i = 0; i < QK_TBQ; ++i) {
        x[i] *= norm;
    }
}

static __device__ __forceinline__ void tbq_precondition_block_cuda(const float * x, float * y) {
    for (int i = 0; i < QK_TBQ; ++i) {
        y[i] = x[i] * (float) k_tbq_signs_cuda[i];
    }
    tbq_hadamard32_cuda(y);
}

static __device__ __forceinline__ void tbq_uncondition_block_cuda(float * x) {
    tbq_hadamard32_cuda(x);
    for (int i = 0; i < QK_TBQ; ++i) {
        x[i] *= (float) k_tbq_signs_cuda[i];
    }
}

template <int N>
static __device__ __forceinline__ uint8_t tbq_best_index_cuda(const float (&codebook)[N], float x) {
    uint8_t best = 0;
    float best_dist = fabsf(x - codebook[0]);

    for (int i = 1; i < N; ++i) {
        const float dist = fabsf(x - codebook[i]);
        if (dist < best_dist) {
            best = (uint8_t) i;
            best_dist = dist;
        }
    }

    return best;
}

static __device__ __forceinline__ void tbq_decode_rotated_cuda(const block_tbq3_0 & block, float * y) {
    const float d = __half2float(block.d);
    if (d == 0.0f) {
        for (int i = 0; i < QK_TBQ; ++i) {
            y[i] = 0.0f;
        }
        return;
    }

    for (int i = 0; i < QK_TBQ; ++i) {
        y[i] = d * k_tbq3_codebook_cuda[tbq3_get_code_cuda(block.qs, i)];
    }
}

static __device__ __forceinline__ void tbq_decode_rotated_cuda(const block_tbq4_0 & block, float * y) {
    const float d = __half2float(block.d);
    if (d == 0.0f) {
        for (int i = 0; i < QK_TBQ; ++i) {
            y[i] = 0.0f;
        }
        return;
    }

    for (int i = 0; i < QK_TBQ; ++i) {
        y[i] = d * k_tbq4_codebook_cuda[tbq4_get_code_cuda(block.qs, i)];
    }
}

static __device__ __forceinline__ void tbq_decode_block_cuda(const block_tbq3_0 & block, float * x) {
    tbq_decode_rotated_cuda(block, x);
    tbq_uncondition_block_cuda(x);
}

static __device__ __forceinline__ void tbq_decode_block_cuda(const block_tbq4_0 & block, float * x) {
    tbq_decode_rotated_cuda(block, x);
    tbq_uncondition_block_cuda(x);
}

static __device__ __forceinline__ void quantize_f32_tbq3_0_block(const float * __restrict__ x, block_tbq3_0 * __restrict__ y) {
    float rotated[QK_TBQ];
    tbq_precondition_block_cuda(x, rotated);

    float sumsq = 0.0f;
    for (int i = 0; i < QK_TBQ; ++i) {
        sumsq += rotated[i] * rotated[i];
    }

    const float d = sqrtf(sumsq / QK_TBQ);
    y->d = __float2half(d);
    for (int i = 0; i < QK_TBQ * 3/8; ++i) {
        y->qs[i] = 0;
    }

    if (d == 0.0f) {
        return;
    }

    const float id = 1.0f / d;
    for (int i = 0; i < QK_TBQ; ++i) {
        tbq3_set_code_cuda(y->qs, i, tbq_best_index_cuda(k_tbq3_codebook_cuda, rotated[i] * id));
    }
}

static __device__ __forceinline__ void quantize_f32_tbq4_0_block(const float * __restrict__ x, block_tbq4_0 * __restrict__ y) {
    float rotated[QK_TBQ];
    tbq_precondition_block_cuda(x, rotated);

    float sumsq = 0.0f;
    for (int i = 0; i < QK_TBQ; ++i) {
        sumsq += rotated[i] * rotated[i];
    }

    const float d = sqrtf(sumsq / QK_TBQ);
    y->d = __float2half(d);
    for (int i = 0; i < QK_TBQ/2; ++i) {
        y->qs[i] = 0;
    }

    if (d == 0.0f) {
        return;
    }

    const float id = 1.0f / d;
    for (int i = 0; i < QK_TBQ; ++i) {
        tbq4_set_code_cuda(y->qs, i, tbq_best_index_cuda(k_tbq4_codebook_cuda, rotated[i] * id));
    }
}
