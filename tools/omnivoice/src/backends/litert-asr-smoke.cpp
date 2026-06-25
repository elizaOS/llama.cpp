/*
 * litert-asr-smoke.cpp — real-proof harness for the LiteRT-LM ASR backend.
 *
 * Reads a 16-bit PCM mono WAV, converts to 16 kHz mono fp32, and runs it
 * through litert_asr_transcribe() (which drives the LiteRT-LM C API audio path).
 * Prints the transcript. This is the linkage + end-to-end proof that the
 * prebuilt liblitert-lm.so transcribes audio through the backend's C-API path.
 *
 * Build (link against the prebuilt .so):
 *   g++ -std=c++17 -DELIZA_ENABLE_LITERT -I<sdk>/include -I<omnivoice>/src \
 *       -I<omnivoice>/include litert-asr-smoke.cpp litert-asr.cpp \
 *       <sdk>/lib/liblitert-lm.so -Wl,-rpath,<sdk>/lib -o /tmp/litert-asr-smoke
 *
 * Usage:
 *   litert-asr-smoke <model.litertlm> <audio.wav>
 */

#include "litert-asr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

/* Minimal 16-bit PCM WAV reader: finds the `data` chunk and returns mono fp32
 * samples plus the sample rate. Mixes down to mono if the file is stereo. */
bool read_wav_pcm16(const char * path, std::vector<float> & pcm,
                    int & sample_rate_hz) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "[asr-smoke] cannot open %s\n", path); return false; }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 44) { std::fclose(f); return false; }
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        std::fclose(f); return false;
    }
    std::fclose(f);

    if (std::memcmp(buf.data(), "RIFF", 4) != 0 ||
        std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
        std::fprintf(stderr, "[asr-smoke] not a RIFF/WAVE file\n");
        return false;
    }

    auto u16 = [&](size_t o) { return uint16_t(buf[o] | (buf[o + 1] << 8)); };
    auto u32 = [&](size_t o) {
        return uint32_t(buf[o] | (buf[o + 1] << 8) | (buf[o + 2] << 16) |
                        (uint32_t(buf[o + 3]) << 24));
    };

    uint16_t channels = 1, bits = 16;
    int rate = 0;
    size_t data_off = 0, data_len = 0;
    size_t p = 12;
    while (p + 8 <= buf.size()) {
        const char * id = reinterpret_cast<const char *>(&buf[p]);
        uint32_t clen = u32(p + 4);
        if (std::memcmp(id, "fmt ", 4) == 0 && p + 24 <= buf.size()) {
            channels = u16(p + 10);
            rate = int(u32(p + 12));
            bits = u16(p + 22);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_off = p + 8;
            data_len = clen;
            break;
        }
        p += 8 + clen + (clen & 1);
    }
    if (data_off == 0 || bits != 16 || channels == 0) {
        std::fprintf(stderr, "[asr-smoke] unsupported WAV (bits=%u ch=%u)\n",
                     bits, channels);
        return false;
    }
    if (data_off + data_len > buf.size()) data_len = buf.size() - data_off;

    const size_t total = data_len / 2;          /* int16 samples */
    const size_t frames = total / channels;
    pcm.resize(frames);
    const int16_t * s = reinterpret_cast<const int16_t *>(&buf[data_off]);
    for (size_t i = 0; i < frames; ++i) {
        int acc = 0;
        for (uint16_t c = 0; c < channels; ++c) acc += s[i * channels + c];
        pcm[i] = float(acc) / (float(channels) * 32768.0f);
    }
    sample_rate_hz = rate;
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const char * model_path =
        argc > 1 ? argv[1] : "/home/shaw/milady/eliza/.cache/gemma-4-E2B-it.litertlm";
    const char * wav_path =
        argc > 2 ? argv[2] : "/home/shaw/milady/eliza/.cache/jfk.wav";

    std::vector<float> pcm;
    int rate = 0;
    if (!read_wav_pcm16(wav_path, pcm, rate)) {
        std::fprintf(stderr, "[asr-smoke] failed to read %s\n", wav_path);
        return 1;
    }
    std::printf("[asr-smoke] %s: %zu samples @ %d Hz (%.2fs)\n", wav_path,
                pcm.size(), rate, double(pcm.size()) / (rate ? rate : 1));
    std::printf("[asr-smoke] supported=%d\n", litert_asr_supported());

    char out_text[8192];
    char * err = nullptr;
    int rc = litert_asr_transcribe(model_path, pcm.data(), pcm.size(), rate,
                                   out_text, sizeof(out_text), &err);
    if (rc < 0) {
        std::fprintf(stderr, "[asr-smoke] transcribe failed rc=%d: %s\n", rc,
                     err ? err : "(no error)");
        std::free(err);
        return 1;
    }
    std::printf("TRANSCRIPT (%d bytes): %s\n", rc, out_text);
    return 0;
}
