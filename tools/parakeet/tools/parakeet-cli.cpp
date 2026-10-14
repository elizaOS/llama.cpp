// tools/parakeet/tools/parakeet-cli.cpp - transcribe WAV files with a
// parakeet-tdt GGUF.
//
//   parakeet-cli -m phonon-2-fv5.gguf [-t N] [--kernel scalar|dotprod|i8mm|avx2]
//                [--stream] [--chunk-ms 80] [--words] file.wav...
#include "parakeet.h"
#include "pk-wav.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model;
    pk_params pp = pk_default_params();
    bool stream = false, words = false;
    int chunk_ms = 80;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) model = argv[++i];
        else if ((a == "-t" || a == "--threads") && i + 1 < argc) pp.n_threads = std::atoi(argv[++i]);
        else if (a == "--kernel" && i + 1 < argc) {
            const std::string k = argv[++i];
            pp.kernel = k == "scalar" ? 1 : k == "dotprod" ? 2 : k == "i8mm" ? 3 : k == "avx2" ? 4 : 0;
        } else if (a == "--stream") stream = true;
        else if (a == "--words") words = true;
        else if (a == "--chunk-ms" && i + 1 < argc) chunk_ms = std::atoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s -m model.gguf [-t N] [--kernel K] [--stream] [--words] file.wav...\n", argv[0]);
            return 0;
        } else files.push_back(a);
    }
    if (model.empty() || files.empty()) {
        std::fprintf(stderr, "usage: %s -m model.gguf [-t N] file.wav...\n", argv[0]);
        return 2;
    }
    char err[512] = { 0 };
    auto t0 = std::chrono::steady_clock::now();
    pk_model * m = pk_model_load(model.c_str(), pp, err, sizeof(err));
    if (!m) {
        std::fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }
    std::fprintf(stderr, "loaded %s in %.0f ms (kernel %s, %d threads)\n", model.c_str(),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                 pk_model_kernel_name(m), pk_model_n_threads(m));
    int rc = 0;
    for (const auto & f : files) {
        std::vector<float> pcm;
        int sr = 0;
        std::string e;
        if (!pkwav::read_wav(f, pcm, sr, e)) {
            std::fprintf(stderr, "%s\n", e.c_str());
            rc = 1;
            continue;
        }
        pcm = pkwav::resample(pcm, sr, 16000);
        const double audio_s = pcm.size() / 16000.0;
        if (stream) {
            pk_stream * s = pk_stream_open(m, pk_stream_default_params());
            const size_t step = (size_t) chunk_ms * 16;
            std::string last;
            char buf[8192];
            for (size_t p = 0; p < pcm.size(); p += step) {
                pk_stream_feed(s, pcm.data() + p, std::min(step, pcm.size() - p));
                if (pk_stream_partial(s, buf, sizeof(buf), nullptr, nullptr) >= 0 && last != buf) {
                    last = buf;
                    std::fprintf(stderr, "  [%6.2fs] %s\n", (p + step) / 16000.0, buf);
                }
            }
            pk_stream_finish(s, buf, sizeof(buf), nullptr, nullptr);
            int nd = 0;
            double dms = 0, first = 0;
            pk_stream_stats(s, &nd, &dms, &first);
            std::printf("%s\t%s\n", f.c_str(), buf);
            std::fprintf(stderr, "  stream: %d decodes, %.0f ms total, first partial %.0f ms, audio %.2fs\n", nd, dms,
                         first, audio_s);
            pk_stream_close(s);
            continue;
        }
        pk_result r;
        t0 = std::chrono::steady_clock::now();
        const int st = pk_transcribe(m, pcm.data(), pcm.size(), nullptr, nullptr, &r);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (st != PK_OK) {
            std::fprintf(stderr, "%s: transcribe failed (%d)\n", f.c_str(), st);
            rc = 1;
            continue;
        }
        std::printf("%s\t%s\n", f.c_str(), r.text);
        std::fprintf(stderr, "  %.2fs audio, %.1f ms (mel %.1f, encoder %.1f, decoder %.1f), RTFx %.1f\n", audio_s, ms,
                     r.t_mel_ms, r.t_encoder_ms, r.t_decoder_ms, audio_s * 1000.0 / ms);
        if (words) {
            std::vector<int> a(4096), b(4096);
            size_t n = a.size();
            pk_result_words(m, &r, a.data(), b.data(), &n);
            for (size_t i = 0; i < n; ++i) std::fprintf(stderr, "    [%5d-%5d ms]\n", a[i], b[i]);
        }
        pk_result_free(&r);
    }
    pk_model_free(m);
    return rc;
}
