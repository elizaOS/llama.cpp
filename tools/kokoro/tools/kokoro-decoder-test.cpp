// SPDX-License-Identifier: MIT
//
// kokoro-decoder-test — validate kokoro_decoder_forward in-repo against the
// PyTorch reference: load an F32 Kokoro GGUF + reference asr/F0/N/style bins,
// run the decoder, write a 24 kHz WAV (compare to dec_audio_ref via whisper).
//
// Usage: kokoro-decoder-test <model.gguf> <asr.f32[512,T]> <F0.f32[2T]> <N.f32[2T]> <style.f32[128]> <T> <out.wav>

#include "kokoro.h"
#include "kokoro-decoder.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> rd(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t n = (size_t) f.tellg() / sizeof(float);
    f.seekg(0);
    std::vector<float> v(n);
    f.read((char *) v.data(), (std::streamsize) (n * sizeof(float)));
    return v;
}

static bool write_wav(const std::string & path, const std::vector<float> & s, int sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t n = (uint32_t) s.size(), data = n * 2, riff = 36 + data, br = (uint32_t) sr * 2;
    auto p32 = [&](uint32_t v){ char b[4]={(char)v,(char)(v>>8),(char)(v>>16),(char)(v>>24)}; f.write(b,4); };
    auto p16 = [&](uint16_t v){ char b[2]={(char)v,(char)(v>>8)}; f.write(b,2); };
    f.write("RIFF",4); p32(riff); f.write("WAVE",4); f.write("fmt ",4); p32(16); p16(1); p16(1);
    p32((uint32_t)sr); p32(br); p16(2); p16(16); f.write("data",4); p32(data);
    for (uint32_t i=0;i<n;++i){ float v=s[i]; v=v>1?1:(v<-1?-1:v); int16_t q=(int16_t)std::lrintf(v*32767.f); char b[2]={(char)(q&0xff),(char)((q>>8)&0xff)}; f.write(b,2);}
    return (bool) f;
}

int main(int argc, char ** argv) {
    if (argc < 8) { std::fprintf(stderr, "usage: %s model asr F0 N style T out.wav\n", argv[0]); return 2; }
    std::string model_p=argv[1], out=argv[7]; int T=std::atoi(argv[6]);
    std::string err;
    auto model = eliza_kokoro::kokoro_load_model(model_p, err);
    if (!model) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    auto asr=rd(argv[2]), F0=rd(argv[3]), N=rd(argv[4]), sty=rd(argv[5]);
    std::printf("asr=%zu F0=%zu N=%zu style=%zu T=%d (expect asr=512*T=%d, F0=2T=%d)\n",
                asr.size(), F0.size(), N.size(), sty.size(), T, 512*T, 2*T);
    std::vector<float> audio;
    if (!eliza_kokoro::kokoro_decoder_forward(model.get(), asr.data(), T, F0.data(), N.data(), sty.data(), audio, err)) {
        std::fprintf(stderr, "decoder: %s\n", err.c_str()); return 1;
    }
    std::printf("audio samples=%zu (%.2fs @24k)\n", audio.size(), audio.size()/24000.0);
    if (!write_wav(out, audio, 24000)) { std::fprintf(stderr, "write failed\n"); return 1; }
    std::printf("wrote %s\n", out.c_str());
    return 0;
}
