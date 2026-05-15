// dac-parity.cpp - elizaOS/eliza#7660
//
// Verifies that the DAC dac_conv_t1d migration to ggml_conv_transpose_1d
// matches the legacy decomposition:
//   transpose -> mul_mat -> col2im_1d -> output_pad -> add_bias
//
// The current ggml op only accepts p0=0, so the migrated path runs the native
// op with p0=0, crops `pad` samples from both sides, then applies output_pad.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Case {
    const char * label;
    int          T_in;
    int          IC;
    int          OC;
    int          K;
    int          stride;
    int          pad;
    int          output_pad;
};

struct CmpStats {
    double mse;
    double max_abs;
    double l2_ref;
    double l2_new;
    size_t n;
};

static void legacy_col2im_1d_f32(
        const float * col, int64_t K_OC, int64_t T_in,
        int stride, int OC, int pad,
        float * dst, int64_t T_out) {
    const int64_t K = K_OC / OC;

    for (int64_t oc = 0; oc < OC; oc++) {
        for (int64_t t_out = 0; t_out < T_out; t_out++) {
            const int64_t t_abs = t_out + pad;
            int64_t       t_min = (t_abs - K + 1 + stride - 1) / stride;
            int64_t       t_max = t_abs / stride;
            t_min               = std::max<int64_t>(t_min, 0);
            t_max               = std::min<int64_t>(t_max, T_in - 1);

            float sum = 0.0f;
            for (int64_t t_in = t_min; t_in <= t_max; t_in++) {
                const int64_t k = t_abs - t_in * stride;
                if (k >= 0 && k < K) {
                    sum += col[t_in * K_OC + oc * K + k];
                }
            }
            dst[oc * T_out + t_out] = sum;
        }
    }
}

static void legacy_dac_conv_t1d_cpu(
        const ggml_fp16_t * w_src,
        const float *       bias,
        const float *       x_in,
        const Case &        c,
        std::vector<float> & out,
        int &               T_out_final) {
    const int64_t K_OC     = (int64_t) c.K * c.OC;
    const int64_t T_no_pad = (int64_t) (c.T_in - 1) * c.stride + c.K - 2 * c.pad;
    T_out_final            = (int) T_no_pad + c.output_pad;

    std::vector<float> col((size_t) K_OC * c.T_in);

    // Legacy mul_mat over weight rows (oc, k) and input channels.
    for (int64_t row = 0; row < K_OC; row++) {
        const int oc = (int) (row / c.K);
        const int k  = (int) (row % c.K);

        for (int t = 0; t < c.T_in; t++) {
            float acc = 0.0f;
            for (int ic = 0; ic < c.IC; ic++) {
                const size_t wi = ((size_t) ic * c.OC + oc) * c.K + k;
                acc += x_in[(size_t) t * c.IC + ic] * ggml_fp16_to_fp32(w_src[wi]);
            }
            col[(size_t) t * K_OC + row] = acc;
        }
    }

    std::vector<float> no_pad((size_t) T_no_pad * c.OC);
    legacy_col2im_1d_f32(col.data(), K_OC, c.T_in, c.stride, c.OC, c.pad, no_pad.data(), T_no_pad);

    out.assign((size_t) T_out_final * c.OC, 0.0f);
    for (int oc = 0; oc < c.OC; oc++) {
        const float bv = bias ? bias[oc] : 0.0f;
        for (int t = 0; t < T_out_final; t++) {
            const float v = t < T_no_pad ? no_pad[(size_t) oc * T_no_pad + t] : 0.0f;
            out[(size_t) t * c.OC + oc] = v + bv;
        }
    }
}

static void new_dac_conv_t1d_ggml(
        const ggml_fp16_t * w_src,
        const float *       bias,
        const float *       x_in,
        const Case &        c,
        std::vector<float> & out,
        int &               T_out_final) {
    const size_t ctx_size = 512u * 1024u * 1024u;
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_init_params params{ ctx_size, ctx_buf.data(), false };
    ggml_context *   ctx = ggml_init(params);
    GGML_ASSERT(ctx);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, c.K, c.OC, c.IC);
    memcpy(w->data, w_src, (size_t) c.IC * c.OC * c.K * sizeof(ggml_fp16_t));

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.T_in, c.IC);
    float *       xd = (float *) x->data;
    for (int t = 0; t < c.T_in; t++) {
        for (int ic = 0; ic < c.IC; ic++) {
            xd[(size_t) ic * c.T_in + t] = x_in[(size_t) t * c.IC + ic];
        }
    }

    ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, c.stride, 0, 1);
    y               = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);

    if (c.pad > 0) {
        const int64_t cropped = y->ne[0] - 2 * (int64_t) c.pad;
        GGML_ASSERT(cropped > 0);
        y = ggml_view_2d(ctx, y, cropped, y->ne[1], y->nb[1], (size_t) c.pad * y->nb[0]);
        y = ggml_cont(ctx, y);
    }

    if (c.output_pad > 0) {
        y = ggml_pad(ctx, y, c.output_pad, 0, 0, 0);
    }

    if (bias) {
        ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, c.OC);
        memcpy(b->data, bias, (size_t) c.OC * sizeof(float));
        y = ggml_add(ctx, y, b);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    T_out_final = (int) y->ne[0];
    GGML_ASSERT((int) y->ne[1] == c.OC);

    out.assign((size_t) T_out_final * c.OC, 0.0f);
    const float * yd = (const float *) y->data;
    for (int oc = 0; oc < c.OC; oc++) {
        for (int t = 0; t < T_out_final; t++) {
            out[(size_t) t * c.OC + oc] = yd[(size_t) oc * T_out_final + t];
        }
    }

    ggml_free(ctx);
}

static CmpStats compare(const std::vector<float> & ref, const std::vector<float> & got) {
    CmpStats s{ 0, 0, 0, 0, ref.size() };
    if (ref.size() != got.size()) {
        s.n = 0;
        return s;
    }

    double se = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double) ref[i] - (double) got[i];
        se += d * d;
        s.max_abs = std::max(s.max_abs, std::fabs(d));
        s.l2_ref += (double) ref[i] * ref[i];
        s.l2_new += (double) got[i] * got[i];
    }
    s.mse    = se / (double) ref.size();
    s.l2_ref = std::sqrt(s.l2_ref);
    s.l2_new = std::sqrt(s.l2_new);
    return s;
}

static bool run_one(
        const Case &                 c,
        const std::vector<ggml_fp16_t> & w_src,
        const std::vector<float> &    bias,
        const std::vector<float> &    x_in,
        double                       threshold) {
    printf("\n=== %s ===\n", c.label);
    printf("  T_in=%d IC=%d OC=%d K=%d stride=%d pad=%d output_pad=%d\n",
            c.T_in, c.IC, c.OC, c.K, c.stride, c.pad, c.output_pad);

    std::vector<float> ref;
    std::vector<float> got;
    int                T_ref = 0;
    int                T_got = 0;

    legacy_dac_conv_t1d_cpu(w_src.data(), bias.empty() ? nullptr : bias.data(), x_in.data(), c, ref, T_ref);
    new_dac_conv_t1d_ggml(w_src.data(), bias.empty() ? nullptr : bias.data(), x_in.data(), c, got, T_got);

    printf("  T_out: legacy=%d new=%d\n", T_ref, T_got);
    if (T_ref != T_got) {
        printf("  PARITY: FAIL (shape mismatch)\n");
        return false;
    }

    const CmpStats s   = compare(ref, got);
    const double   rel = s.l2_ref > 0.0 ? std::sqrt(s.mse * (double) s.n) / s.l2_ref : 0.0;
    printf("  mse=%.6e max_abs=%.6e l2_legacy=%.4e l2_new=%.4e rel_err=%.4e\n",
            s.mse, s.max_abs, s.l2_ref, s.l2_new, rel);

    const bool ok = s.mse <= threshold;
    printf("  PARITY: %s (mse <= %.1e)\n", ok ? "PASS" : "FAIL", threshold);
    return ok;
}

static bool load_real_block(
        ggml_context *             mctx,
        int                        block,
        std::vector<ggml_fp16_t> & w_src,
        std::vector<float> &       bias,
        int &                      IC,
        int &                      OC,
        int &                      K) {
    char wname[128];
    char bname[128];
    snprintf(wname, sizeof(wname), "acoustic_decoder.block.%d.conv_t1.weight", block);
    snprintf(bname, sizeof(bname), "acoustic_decoder.block.%d.conv_t1.bias", block);

    ggml_tensor * wt = ggml_get_tensor(mctx, wname);
    ggml_tensor * bt = ggml_get_tensor(mctx, bname);
    if (!wt || !bt) {
        fprintf(stderr, "[dac-parity] missing %s/%s\n", wname, bname);
        return false;
    }
    if (wt->type != GGML_TYPE_F16 || bt->type != GGML_TYPE_F32) {
        fprintf(stderr, "[dac-parity] unsupported real tensor types in block %d\n", block);
        return false;
    }

    K  = (int) wt->ne[0];
    OC = (int) wt->ne[1];
    IC = (int) wt->ne[2];

    w_src.assign((size_t) IC * OC * K, ggml_fp16_t{ 0 });
    bias.assign((size_t) OC, 0.0f);
    memcpy(w_src.data(), wt->data, w_src.size() * sizeof(ggml_fp16_t));
    memcpy(bias.data(), bt->data, bias.size() * sizeof(float));
    return true;
}

int main(int argc, char ** argv) {
    const char * gguf_path   = nullptr;
    int          real_T_in   = 8;
    int          max_blocks  = 5;
    double       threshold   = 1e-5;
    bool         skip_real   = false;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--gguf" && i + 1 < argc) {
            gguf_path = argv[++i];
        } else if (arg == "--T-in" && i + 1 < argc) {
            real_T_in = std::atoi(argv[++i]);
        } else if (arg == "--max-blocks" && i + 1 < argc) {
            max_blocks = std::atoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::atof(argv[++i]);
        } else if (arg == "--no-real") {
            skip_real = true;
        } else if (arg == "--help" || arg == "-h") {
            printf("usage: %s [--no-real] [--gguf tokenizer.gguf] [--T-in n] [--threshold mse]\n", argv[0]);
            return 0;
        }
    }
    if (!gguf_path && !skip_real) {
        gguf_path = std::getenv("DAC_GGUF");
    }

    const Case synthetic[] = {
        { "synthetic block0 (s=8, k=16)", 4, 1024, 512, 16, 8, 4, 0 },
        { "synthetic block1 (s=5, k=10)", 8, 512, 256, 10, 5, 3, 1 },
        { "synthetic block2 (s=4, k=8)", 16, 256, 128, 8, 4, 2, 0 },
        { "synthetic block3 (s=2, k=4)", 32, 128, 64, 4, 2, 1, 0 },
        { "synthetic block4 (s=3, k=6)", 48, 64, 32, 6, 3, 2, 1 },
    };

    std::mt19937 rng(0x7660DACu);
    std::uniform_real_distribution<float> wd(-0.05f, 0.05f);
    std::uniform_real_distribution<float> bd(-0.10f, 0.10f);
    std::uniform_real_distribution<float> xd(-1.00f, 1.00f);

    int total = 0;
    int pass  = 0;

    printf("\n==============================\n");
    printf(" SYNTHETIC DAC CONV_T1D PARITY\n");
    printf("==============================\n");

    for (const Case & c : synthetic) {
        std::vector<ggml_fp16_t> w((size_t) c.IC * c.OC * c.K);
        std::vector<float>       b((size_t) c.OC);
        std::vector<float>       x((size_t) c.T_in * c.IC);

        for (ggml_fp16_t & v : w) {
            v = ggml_fp32_to_fp16(wd(rng));
        }
        for (float & v : b) {
            v = bd(rng);
        }
        for (float & v : x) {
            v = xd(rng);
        }

        total++;
        if (run_one(c, w, b, x, threshold)) {
            pass++;
        }
    }

    if (!skip_real && gguf_path) {
        printf("\n==============================\n");
        printf(" REAL DAC WEIGHT PARITY (%s)\n", gguf_path);
        printf("==============================\n");

        ggml_context *    mctx = nullptr;
        gguf_init_params  gp{ false, &mctx };
        gguf_context *    gctx = gguf_init_from_file(gguf_path, gp);
        if (!gctx || !mctx) {
            fprintf(stderr, "[dac-parity] failed to open %s\n", gguf_path);
            return 2;
        }

        const int strides[] = { 8, 5, 4, 2, 3 };
        for (int block = 0; block < max_blocks && block < 5; block++) {
            std::vector<ggml_fp16_t> w;
            std::vector<float>       b;
            int IC = 0;
            int OC = 0;
            int K  = 0;
            if (!load_real_block(mctx, block, w, b, IC, OC, K)) {
                continue;
            }

            char label[64];
            snprintf(label, sizeof(label), "real block%d (s=%d, k=%d)", block, strides[block], K);
            const Case c{
                label,
                real_T_in,
                IC,
                OC,
                K,
                strides[block],
                (strides[block] + 1) / 2,
                strides[block] % 2,
            };
            std::vector<float> x((size_t) c.T_in * c.IC);
            for (float & v : x) {
                v = xd(rng);
            }

            total++;
            if (run_one(c, w, b, x, threshold)) {
                pass++;
            }
        }

        gguf_free(gctx);
        ggml_free(mctx);
    } else {
        printf("\n[dac-parity] real-weight pass skipped; set DAC_GGUF or pass --gguf to enable it.\n");
    }

    printf("\n==============================\n");
    printf(" SUMMARY: total=%d pass=%d threshold=%.1e\n", total, pass, threshold);
    printf("==============================\n");

    return pass == total ? 0 : 1;
}
