// SPDX-License-Identifier: Apache-2.0
// misaki-g2p and Kokoro both drive the process-global libespeak-ng. They share
// one lock and init owner (tools/espeak-shared/eliza-espeak-shared.h); this
// runs both phonemizers concurrently from two threads and checks every result
// against the single-threaded one. Meant to be run under TSan as well.
//
// usage: test-espeak-shared MINI_LEX [ESPEAK_DATA_DIR]
// exit 77 (skip) when either side was built without libespeak-ng.
#include "kokoro-phonemes.h"
#include "misaki-g2p.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static std::string misaki_ps(misaki_g2p * g, const std::string & text) {
    char * out = nullptr;
    char * err = nullptr;
    std::string s;
    if (misaki_g2p_phonemize(g, text.data(), text.size(), &out, &err) == 0 && out) s = out;
    else s = std::string("ERROR: ") + (err ? err : "?");
    misaki_g2p_free(out);
    misaki_g2p_free(err);
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MINI_LEX [ESPEAK_DATA_DIR]\n", argv[0]);
        return 2;
    }
    if (!misaki_g2p_espeak_linked() || !eliza_kokoro::espeak_available()) {
        std::printf("SKIP: misaki-g2p or Kokoro built without libespeak-ng\n");
        return 77;
    }
    const char * data_dir = argc > 2 ? argv[2] : "";
    char * err = nullptr;
    misaki_g2p * g = misaki_g2p_open(argv[1], data_dir, &err);
    if (!g) {
        std::printf("SKIP: eSpeak NG data not usable: %s\n", err ? err : "?");
        misaki_g2p_free(err);
        return 77;
    }
    if (misaki_g2p_fallback_kind(g) != MISAKI_G2P_FALLBACK_ESPEAK) {
        std::fprintf(stderr, "FAIL: misaki opened without the eSpeak fallback\n");
        misaki_g2p_close(g);
        return 1;
    }

    // Out-of-lexicon words, so every misaki call goes through eSpeak.
    const std::string misaki_text = "Zyx quorbled the flimbit, glarf!";
    const std::string kokoro_text = "Hello world, this is a concurrency check.";
    const std::string misaki_want = misaki_ps(g, misaki_text);
    const std::vector<int32_t> kokoro_want = eliza_kokoro::phonemize_ipa(kokoro_text);
    if (misaki_want.empty() || misaki_want.rfind("ERROR", 0) == 0 || kokoro_want.empty()) {
        std::fprintf(stderr, "FAIL: baseline: misaki '%s', kokoro %zu ids\n", misaki_want.c_str(), kokoro_want.size());
        misaki_g2p_close(g);
        return 1;
    }

    const int iters = 200;
    std::atomic<int> misaki_bad{0}, kokoro_bad{0};
    std::atomic<bool> go{false};
    std::thread tm([&] {
        while (!go.load()) std::this_thread::yield();
        for (int i = 0; i < iters; i++) if (misaki_ps(g, misaki_text) != misaki_want) misaki_bad++;
    });
    std::thread tk([&] {
        while (!go.load()) std::this_thread::yield();
        for (int i = 0; i < iters; i++) if (eliza_kokoro::phonemize_ipa(kokoro_text) != kokoro_want) kokoro_bad++;
    });
    go = true;
    tm.join();
    tk.join();
    misaki_g2p_close(g);

    std::printf("misaki: %d/%d differ, kokoro: %d/%d differ\n", misaki_bad.load(), iters, kokoro_bad.load(), iters);
    return misaki_bad || kokoro_bad ? 1 : 0;
}
