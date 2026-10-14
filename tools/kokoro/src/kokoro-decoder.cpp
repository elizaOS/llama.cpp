// SPDX-License-Identifier: MIT
//
// kokoro-decoder.cpp — assemble decoder_front + Generator into the full
// StyleTTS-2 / iSTFTNet decoder, reading weights from the model's all-F32
// ggml context (dequantized at load). Validated against the PyTorch reference
// stage-by-stage (#9588).
//
// Channel widths, kernel sizes and the style width are read from the tensor
// shapes (ggml ne[] is the reversed PyTorch shape), so Kokoro-82M and its
// width-scaled students share this file.

#include "kokoro-decoder.h"
#include "kokoro-decoder-front.h"   // DecoderFrontWeights, DecAdainResBlk, decoder_front
#include "kokoro-generator.h"       // GeneratorWeights, generator_forward
#include "kokoro-model-internal.h"
#include "kokoro-profile.h"

#include "ggml.h"

#include <string>

namespace eliza_kokoro {

namespace {

struct Lk {
    ggml_context * ctx;
    std::string ns;
    std::string full(const std::string & rel) const { return ns + rel; }
    ggml_tensor * tensor(const std::string & rel) const {
        return kokoro_find_tensor_compat(ctx, full(rel));
    }
    const float * get(const std::string & rel) const {
        ggml_tensor * t = tensor(rel);
        return t ? (const float *) t->data : nullptr;
    }
};

// Fill an AdainResBlk1d (decode flavor) from a tensor-name prefix. Cin/Cout
// come from conv1.weight [Cout, Cin, 3] and Sdim from norm1.fc.weight
// [2*Cin, Sdim].
bool fill_dec_block(const Lk & L, DecAdainResBlk & b, const std::string & pfx,
                    bool upsample, bool affine) {
    ggml_tensor * c1 = L.tensor(pfx + ".conv1.weight");
    ggml_tensor * n1 = L.tensor(pfx + ".norm1.fc.weight");
    if (!c1 || !n1) return false;
    const int Cin = (int) c1->ne[1], Cout = (int) c1->ne[2];
    b.Cin = Cin; b.Cout = Cout; b.Sdim = (int) n1->ne[0];
    b.upsample = upsample;
    b.learned_sc = (Cin != Cout);
    b.norm1_fc_w = L.get(pfx + ".norm1.fc.weight");
    b.norm1_fc_b = L.get(pfx + ".norm1.fc.bias");
    b.norm2_fc_w = L.get(pfx + ".norm2.fc.weight");
    b.norm2_fc_b = L.get(pfx + ".norm2.fc.bias");
    b.conv1_w    = L.get(pfx + ".conv1.weight");
    b.conv1_b    = L.get(pfx + ".conv1.bias");
    b.conv2_w    = L.get(pfx + ".conv2.weight");
    b.conv2_b    = L.get(pfx + ".conv2.bias");
    b.conv1x1_w  = b.learned_sc ? L.get(pfx + ".conv1x1.weight") : nullptr;
    b.conv1x1_b  = nullptr;  // conv1x1 bias=False upstream
    b.pool_w     = upsample ? L.get(pfx + ".pool.weight") : nullptr;
    b.pool_b     = upsample ? L.get(pfx + ".pool.bias")   : nullptr;
    if (affine) {
        b.norm1_w = L.get(pfx + ".norm1.norm.weight");
        b.norm1_b = L.get(pfx + ".norm1.norm.bias");
        b.norm2_w = L.get(pfx + ".norm2.norm.weight");
        b.norm2_b = L.get(pfx + ".norm2.norm.bias");
        if (!b.norm1_w || !b.norm1_b || !b.norm2_w || !b.norm2_b) return false;
    }
    return true;
}

// Fill an AdaINResBlock1 (generator flavor: 3 sub-blocks, Snake1D) from a prefix.
bool fill_gen_block(const Lk & L, GenAdaResBlockWeights & b, const std::string & pfx, bool affine) {
    for (int j = 0; j < 3; ++j) {
        const std::string js = std::to_string(j);
        GenSubBlockWeights & s = b.sub[j];
        s.conv1_w     = L.get(pfx + ".convs1." + js + ".weight");
        s.conv1_b     = L.get(pfx + ".convs1." + js + ".bias");
        s.conv2_w     = L.get(pfx + ".convs2." + js + ".weight");
        s.conv2_b     = L.get(pfx + ".convs2." + js + ".bias");
        s.adain1_fc_w = L.get(pfx + ".adain1." + js + ".fc.weight");
        s.adain1_fc_b = L.get(pfx + ".adain1." + js + ".fc.bias");
        s.adain2_fc_w = L.get(pfx + ".adain2." + js + ".fc.weight");
        s.adain2_fc_b = L.get(pfx + ".adain2." + js + ".fc.bias");
        s.alpha1      = L.get(pfx + ".alpha1." + js);
        s.alpha2      = L.get(pfx + ".alpha2." + js);
        if (affine) {
            s.adain1_norm_w = L.get(pfx + ".adain1." + js + ".norm.weight");
            s.adain1_norm_b = L.get(pfx + ".adain1." + js + ".norm.bias");
            s.adain2_norm_w = L.get(pfx + ".adain2." + js + ".norm.weight");
            s.adain2_norm_b = L.get(pfx + ".adain2." + js + ".norm.bias");
            if (!s.adain1_norm_w || !s.adain1_norm_b || !s.adain2_norm_w || !s.adain2_norm_b) return false;
        }
    }
    return true;
}

} // namespace

bool decoder_forward(
        ggml_context * ctx,
        DecoderConfig & cfg,
        const float * asr_ct, int asr_dim, int T_frame,
        const float * F0, const float * N,
        const float * style,
        std::vector<float> & audio,
        std::string & err) {
    audio.clear();
    cfg.cancelled = false;
    if (!ctx) { err = "null model context"; return false; }
    if (T_frame <= 0) { err = "non-positive T_frame"; return false; }
    Lk L{ctx, cfg.ns};
    const bool affine = cfg.instnorm_affine;

    // --- decoder_front weights ---
    DecoderFrontWeights W;
    W.F0_conv_w = L.get(".decoder.F0_conv.weight");
    W.F0_conv_b = L.get(".decoder.F0_conv.bias");
    W.N_conv_w  = L.get(".decoder.N_conv.weight");
    W.N_conv_b  = L.get(".decoder.N_conv.bias");
    W.asr_res_w = L.get(".decoder.asr_res.weight");
    W.asr_res_b = L.get(".decoder.asr_res.bias");
    ggml_tensor * asr_res_t = L.tensor(".decoder.asr_res.weight");   // [Cr, asr_dim, 1]
    bool ok = asr_res_t != nullptr;
    if (ok) W.asr_res_C = (int) asr_res_t->ne[2];
    ok = ok && fill_dec_block(L, W.encode, ".decoder.encode", /*upsample*/false, affine);
    for (int b = 0; b < 4 && ok; ++b) {
        ok = fill_dec_block(L, W.decode[b], ".decoder.decode." + std::to_string(b),
                            /*upsample*/ b == 3, affine);
    }
    if (!ok || !W.F0_conv_w || !W.asr_res_w || !W.encode.conv1_w || !W.decode[3].pool_w) {
        err = "missing decoder weights (is the GGUF a full " + cfg.ns + " model?)";
        return false;
    }
    if (W.encode.Cin != asr_dim + 2 || (int) asr_res_t->ne[1] != asr_dim) {
        err = "decoder input width " + std::to_string(W.encode.Cin - 2) +
              " != asr width " + std::to_string(asr_dim);
        return false;
    }

    // --- generator weights ---
    GeneratorWeights G;
    G.l_linear_w = L.get(".gen.m_source.l_linear.weight");
    G.l_linear_b = L.get(".gen.m_source.l_linear.bias");
    for (int i = 0; i < 2 && ok; ++i) {
        const std::string is = std::to_string(i);
        G.ups_w[i]         = L.get(".gen.ups." + is + ".weight");
        G.ups_b[i]         = L.get(".gen.ups." + is + ".bias");
        G.noise_convs_w[i] = L.get(".gen.noise_convs." + is + ".weight");
        G.noise_convs_b[i] = L.get(".gen.noise_convs." + is + ".bias");
        ok = fill_gen_block(L, G.noise_res[i], ".gen.noise_res." + is, affine);
    }
    for (int i = 0; i < 6 && ok; ++i) {
        ok = fill_gen_block(L, G.resblocks[i], ".gen.resblocks." + std::to_string(i), affine);
    }
    G.conv_post_w = L.get(".gen.conv_post.weight");
    G.conv_post_b = L.get(".gen.conv_post.bias");

    ggml_tensor * ups0 = L.tensor(".gen.ups.0.weight");        // [Cin, ch0, K0]
    ggml_tensor * ups1 = L.tensor(".gen.ups.1.weight");        // [ch0, ch1, K1]
    ggml_tensor * post = L.tensor(".gen.conv_post.weight");    // [n_fft+2, ch1, 7]
    ggml_tensor * lin  = L.tensor(".gen.m_source.l_linear.weight");  // [1, harmonics]
    if (!ok || !G.l_linear_w || !G.ups_w[0] || !G.conv_post_w || !G.resblocks[5].sub[2].conv2_w ||
        !ups0 || !ups1 || !post || !lin) {
        err = "missing generator weights (is the GGUF a full " + cfg.ns + " model?)";
        return false;
    }

    GeneratorConfig gc;
    gc.Cin = (int) ups0->ne[2];
    gc.ch[0] = (int) ups0->ne[1];
    gc.ch[1] = (int) ups1->ne[1];
    gc.up_kernels[0] = (int) ups0->ne[0];
    gc.up_kernels[1] = (int) ups1->ne[0];
    gc.up_rates[0] = cfg.up_rates[0];
    gc.up_rates[1] = cfg.up_rates[1];
    for (int j = 0; j < 3; ++j) {
        ggml_tensor * rb = L.tensor(".gen.resblocks." + std::to_string(j) + ".convs1.0.weight");
        if (!rb) { err = "missing generator resblock weights"; return false; }
        gc.rb_kernels[j] = (int) rb->ne[0];
    }
    gc.n_fft = (int) post->ne[2] - 2;
    gc.hop = cfg.istft_hop;
    gc.harmonics = (int) lin->ne[0];
    ggml_tensor * gen_fc = L.tensor(".gen.resblocks.0.adain1.0.fc.weight");  // [2C, Sdim]
    gc.Sdim = gen_fc ? (int) gen_fc->ne[0] : W.encode.Sdim;
    gc.canonical_source_stft = cfg.canonical_source_stft;
    gc.precise_source = cfg.precise_source;
    gc.instnorm_affine = affine;
    gc.har_source_out = cfg.har_source_out;
    gc.har_source_in = cfg.har_source_in;
    gc.cancel = cfg.cancel;
    gc.cancel_user = cfg.cancel_user;
    if (W.decode[3].Cout != gc.Cin || (int) ups1->ne[2] != gc.ch[0] || (int) post->ne[1] != gc.ch[1] ||
        gc.Sdim != W.encode.Sdim || gc.n_fft <= 0) {
        err = "inconsistent decoder/generator tensor shapes";
        return false;
    }

    // --- run: decoder_front -> generator ---
    std::vector<float> x, F0_down, N_down;
    {
        kokoro_phase_timer t("decoder front");
        decoder_front(W, asr_ct, asr_dim, T_frame, F0, N, style, x, F0_down, N_down);
    }
    if (cfg.cancel && cfg.cancel(cfg.cancel_user)) {
        cfg.cancelled = true;
        err = "cancelled";
        return false;
    }

    const int T0 = 2 * T_frame;  // decoder_front upsamples T_frame -> 2*T_frame
    if ((int) (x.size() / (size_t) gc.Cin) != T0) {
        err = "decoder_front output width mismatch";
        return false;
    }
    {
        kokoro_phase_timer t("generator (iSTFTNet)");
        if (!generator_forward(x.data(), T0, style, F0, G, gc, audio)) {
            cfg.cancelled = true;
            err = "cancelled";
            return false;
        }
    }
    return true;
}

bool kokoro_decoder_forward(
        const kokoro_model * model,
        const float * asr_ct, int T_frame,
        const float * F0, const float * N,
        const float * ref_s_dec,
        std::vector<float> & audio,
        std::string & err) {
    audio.clear();
    if (!model) { err = "null model"; return false; }
    DecoderConfig cfg;   // Kokoro-82M: namespace "kokoro", identity InstanceNorm affine, shipped source STFT
    ggml_context * ctx = kokoro_model_ggml_ctx(model);
    // asr width from asr_res.weight [64, asr_dim, 1] (512 for Kokoro-82M).
    ggml_tensor * asr_res = kokoro_find_tensor_compat(ctx, "kokoro.decoder.asr_res.weight");
    if (!asr_res) { err = "missing decoder weights (is the GGUF a full kokoro model?)"; return false; }
    return decoder_forward(ctx, cfg, asr_ct, (int) asr_res->ne[1], T_frame,
                           F0, N, ref_s_dec, audio, err);
}

} // namespace eliza_kokoro
