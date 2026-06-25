// SPDX-License-Identifier: MIT
//
// kokoro-decoder.cpp — assemble decoder_front + Generator into the full
// StyleTTS-2 / iSTFTNet decoder, reading weights from the model's all-F32
// ggml context (dequantized at load). Validated against the PyTorch reference
// stage-by-stage (#9588).

#include "kokoro-decoder.h"
#include "kokoro-decoder-front.h"   // DecoderFrontWeights, DecAdainResBlk, decoder_front
#include "kokoro-generator.h"       // GeneratorWeights, kokoro_generator_forward

#include "ggml.h"

#include <string>

namespace eliza_kokoro {

// Defined in kokoro.cpp — the all-F32 working context the predictor reads.
ggml_context * kokoro_model_ggml_ctx(const kokoro_model * model);

namespace {

struct Lk {
    ggml_context * ctx;
    const float * get(const std::string & name) const {
        ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        return t ? (const float *) t->data : nullptr;
    }
};

// Fill an AdainResBlk1d (decode flavor) from a tensor-name prefix.
void fill_dec_block(const Lk & L, DecAdainResBlk & b, const std::string & pfx,
                    int Cin, int Cout, bool upsample) {
    b.Cin = Cin; b.Cout = Cout; b.Sdim = 128;
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
}

// Fill an AdaINResBlock1 (generator flavor: 3 sub-blocks, Snake1D) from a prefix.
void fill_gen_block(const Lk & L, GenAdaResBlockWeights & b, const std::string & pfx) {
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
    }
}

} // namespace

bool kokoro_decoder_forward(
        const kokoro_model * model,
        const float * asr_ct, int T_frame,
        const float * F0, const float * N,
        const float * ref_s_dec,
        std::vector<float> & audio,
        std::string & err) {
    audio.clear();
    if (!model) { err = "null model"; return false; }
    if (T_frame <= 0) { err = "non-positive T_frame"; return false; }

    ggml_context * ctx = kokoro_model_ggml_ctx(model);
    if (!ctx) { err = "null model context"; return false; }
    Lk L{ctx};

    // --- decoder_front weights ---
    DecoderFrontWeights W;
    W.F0_conv_w = L.get("kokoro.decoder.F0_conv.weight");
    W.F0_conv_b = L.get("kokoro.decoder.F0_conv.bias");
    W.N_conv_w  = L.get("kokoro.decoder.N_conv.weight");
    W.N_conv_b  = L.get("kokoro.decoder.N_conv.bias");
    W.asr_res_w = L.get("kokoro.decoder.asr_res.weight");
    W.asr_res_b = L.get("kokoro.decoder.asr_res.bias");
    fill_dec_block(L, W.encode, "kokoro.decoder.encode", 514, 1024, /*upsample*/false);
    fill_dec_block(L, W.decode[0], "kokoro.decoder.decode.0", 1090, 1024, false);
    fill_dec_block(L, W.decode[1], "kokoro.decoder.decode.1", 1090, 1024, false);
    fill_dec_block(L, W.decode[2], "kokoro.decoder.decode.2", 1090, 1024, false);
    fill_dec_block(L, W.decode[3], "kokoro.decoder.decode.3", 1090, 512,  /*upsample*/true);

    if (!W.F0_conv_w || !W.asr_res_w || !W.encode.conv1_w || !W.decode[3].pool_w) {
        err = "missing decoder weights (is the GGUF a full Kokoro model?)";
        return false;
    }

    // --- generator weights ---
    GeneratorWeights G;
    G.l_linear_w = L.get("kokoro.gen.m_source.l_linear.weight");
    G.l_linear_b = L.get("kokoro.gen.m_source.l_linear.bias");
    for (int i = 0; i < 2; ++i) {
        const std::string is = std::to_string(i);
        G.ups_w[i]         = L.get("kokoro.gen.ups." + is + ".weight");
        G.ups_b[i]         = L.get("kokoro.gen.ups." + is + ".bias");
        G.noise_convs_w[i] = L.get("kokoro.gen.noise_convs." + is + ".weight");
        G.noise_convs_b[i] = L.get("kokoro.gen.noise_convs." + is + ".bias");
        fill_gen_block(L, G.noise_res[i], "kokoro.gen.noise_res." + is);
    }
    for (int i = 0; i < 6; ++i) {
        fill_gen_block(L, G.resblocks[i], "kokoro.gen.resblocks." + std::to_string(i));
    }
    G.conv_post_w = L.get("kokoro.gen.conv_post.weight");
    G.conv_post_b = L.get("kokoro.gen.conv_post.bias");

    if (!G.l_linear_w || !G.ups_w[0] || !G.conv_post_w || !G.resblocks[5].sub[2].conv2_w) {
        err = "missing generator weights (is the GGUF a full Kokoro model?)";
        return false;
    }

    // --- run: decoder_front -> generator ---
    std::vector<float> x, F0_down, N_down;
    decoder_front(W, asr_ct, /*Cin_asr*/512, T_frame, F0, N, ref_s_dec, x, F0_down, N_down);

    const int T0 = 2 * T_frame;  // decoder_front upsamples T_frame -> 2*T_frame
    if ((int) (x.size() / 512) != T0) {
        err = "decoder_front output width mismatch";
        return false;
    }
    kokoro_generator_forward(x.data(), T0, ref_s_dec, F0, G, audio);
    return true;
}

} // namespace eliza_kokoro
