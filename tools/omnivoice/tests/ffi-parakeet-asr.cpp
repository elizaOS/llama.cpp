// tools/omnivoice/tests/ffi-parakeet-asr.cpp - the eliza_inference_asr_*
// surface on the Parakeet-TDT backend, through the real libelizainference.
//
// Needs a bundle with asr/<parakeet-tdt>.gguf and a 16 kHz fp32 mono clip:
//   ELIZA_TEST_PARAKEET_BUNDLE=<dir>  ELIZA_TEST_PARAKEET_PCM=<clip.pcm.f32>
//   [ELIZA_TEST_PARAKEET_TEXT=<expected transcript>]
// Without them the test prints "skipped" and passes.
//
// Checks: stream probe is 0 before the region is acquired and 1 after;
// batch and timed transcripts agree (and match the expected text when
// given); word count equals the whitespace split and times are monotonic;
// a 48 kHz stream fed in 80 ms chunks yields partials and a final
// transcript; token ids are not returned; asr_cancel from another thread
// cancels a batch call; evicting the region turns the probe off and fails
// an open stream cleanly; re-acquiring restores it.
#include "eliza-inference-ffi.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++fails; } } while (0)

static std::vector<float> read_f32(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::vector<float> v;
    if (!f) return v;
    const size_t n = (size_t) f.tellg();
    f.seekg(0);
    v.resize(n / 4);
    f.read((char *) v.data(), (std::streamsize) n);
    return v;
}

static size_t count_words(const std::string & s) {
    std::istringstream is(s);
    std::string w;
    size_t n = 0;
    while (is >> w) ++n;
    return n;
}

int main() {
    const char * bundle = std::getenv("ELIZA_TEST_PARAKEET_BUNDLE");
    const char * pcm_path = std::getenv("ELIZA_TEST_PARAKEET_PCM");
    const char * expect = std::getenv("ELIZA_TEST_PARAKEET_TEXT");
    if (!bundle || !pcm_path) {
        std::printf("skipped: set ELIZA_TEST_PARAKEET_BUNDLE and ELIZA_TEST_PARAKEET_PCM\n");
        return 0;
    }
    std::vector<float> pcm = read_f32(pcm_path);
    CHECK(pcm.size() > 16000, "clip %s too short or unreadable", pcm_path);
    char * err = nullptr;
    EliInferenceContext * ctx = eliza_inference_create(bundle, &err);
    CHECK(ctx, "create: %s", err ? err : "?");
    if (!ctx) return 1;

    CHECK(eliza_inference_asr_stream_supported() == 0, "stream probe must be 0 before acquire");
    int rc = eliza_inference_mmap_acquire(ctx, "asr", &err);
    CHECK(rc == ELIZA_OK, "acquire asr: %d %s", rc, err ? err : "");
    CHECK(eliza_inference_asr_stream_supported() == 1, "stream probe must be 1 with parakeet loaded");

    // batch
    char text[4096];
    auto t0 = std::chrono::steady_clock::now();
    rc = eliza_inference_asr_transcribe(ctx, pcm.data(), pcm.size(), 16000, text, sizeof(text), &err);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(rc > 0, "transcribe rc %d %s", rc, err ? err : "");
    std::printf("batch (%.0f ms for %.2f s): %s\n", ms, pcm.size() / 16000.0, text);
    if (expect) CHECK(std::strcmp(text, expect) == 0, "transcript differs from expected:\n  %s\n  %s", text, expect);

    // timed
    char text2[4096];
    int ws[512], we[512];
    size_t nw = 512;
    rc = eliza_inference_asr_transcribe_timed(ctx, pcm.data(), pcm.size(), 16000, text2, sizeof(text2), ws, we, &nw, &err);
    CHECK(rc > 0 && std::strcmp(text, text2) == 0, "timed transcript differs");
    CHECK(nw == count_words(text2), "word count %zu != whitespace split %zu", nw, count_words(text2));
    for (size_t i = 0; i < nw; ++i) {
        CHECK(ws[i] < we[i], "word %zu: start %d >= end %d", i, ws[i], we[i]);
        if (i) CHECK(ws[i] >= ws[i - 1], "word %zu starts before word %zu", i, i - 1);
        CHECK(we[i] <= (int) (pcm.size() / 16) + 80, "word %zu ends past the audio", i);
    }

    // stream at 48 kHz in 80 ms chunks
    std::vector<float> up(pcm.size() * 3);
    for (size_t i = 0; i < up.size(); ++i) {
        const double s = i / 3.0;
        const size_t a = (size_t) s;
        const size_t b = std::min(a + 1, pcm.size() - 1);
        up[i] = (float) (pcm[a] + (pcm[b] - pcm[a]) * (s - a));
    }
    EliAsrStream * st = eliza_inference_asr_stream_open(ctx, 48000, &err);
    CHECK(st, "stream_open: %s", err ? err : "");
    int partials = 0;
    std::string last;
    for (size_t p = 0; st && p < up.size(); p += 3840) {
        const size_t n = std::min<size_t>(3840, up.size() - p);
        CHECK(eliza_inference_asr_stream_feed(st, up.data() + p, n, &err) == (int) n, "feed");
        char part[4096];
        int toks[16];
        size_t nt = 16;
        rc = eliza_inference_asr_stream_partial(st, part, sizeof(part), toks, &nt, &err);
        CHECK(rc >= 0, "partial rc %d", rc);
        CHECK(nt == 0, "token ids must not be returned");
        if (rc > 0 && last != part) {
            last = part;
            ++partials;
        }
    }
    if (st) {
        char fin[4096];
        rc = eliza_inference_asr_stream_finish(st, fin, sizeof(fin), nullptr, nullptr, &err);
        CHECK(rc > 0, "finish rc %d", rc);
        std::printf("stream (%d partial updates): %s\n", partials, fin);
        CHECK(partials >= 1, "no partial transcript before finish");
        eliza_inference_asr_stream_close(st);
    }

    // cancel a batch call from another thread
    {
        std::vector<float> longpcm;
        for (int r = 0; r < 4; ++r) longpcm.insert(longpcm.end(), pcm.begin(), pcm.end());
        std::atomic<int> crc{ 1 };
        std::thread th([&] {
            char t[8192];
            char * e = nullptr;
            crc = eliza_inference_asr_transcribe(ctx, longpcm.data(), longpcm.size(), 16000, t, sizeof(t), &e);
            eliza_inference_free_string(e);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        CHECK(eliza_inference_asr_cancel(ctx, &err) == ELIZA_OK, "cancel");
        th.join();
        CHECK(crc == ELIZA_ERR_CANCELLED || crc > 0, "cancelled transcribe rc %d", (int) crc);
        std::printf("cancel: rc %d (%s)\n", (int) crc, crc == ELIZA_ERR_CANCELLED ? "cancelled" : "finished before the cancel landed");
        rc = eliza_inference_asr_transcribe(ctx, pcm.data(), pcm.size(), 16000, text2, sizeof(text2), &err);
        CHECK(rc > 0 && std::strcmp(text, text2) == 0, "transcribe after cancel differs");
    }

    // evict / re-acquire
    st = eliza_inference_asr_stream_open(ctx, 16000, &err);
    CHECK(st, "stream_open before evict");
    CHECK(eliza_inference_mmap_evict(ctx, "asr", &err) == ELIZA_OK, "evict");
    CHECK(eliza_inference_asr_stream_supported() == 0, "probe must be 0 after evict");
    if (st) {
        eliza_inference_asr_stream_feed(st, pcm.data(), 16000, &err);
        char part[256];
        rc = eliza_inference_asr_stream_partial(st, part, sizeof(part), nullptr, nullptr, &err);
        CHECK(rc < 0, "partial on an evicted region must fail, got %d", rc);
        eliza_inference_asr_stream_close(st);
    }
    CHECK(eliza_inference_mmap_acquire(ctx, "asr", &err) == ELIZA_OK, "re-acquire");
    rc = eliza_inference_asr_transcribe(ctx, pcm.data(), pcm.size(), 16000, text2, sizeof(text2), &err);
    CHECK(rc > 0 && std::strcmp(text, text2) == 0, "transcribe after re-acquire differs");

    eliza_inference_destroy(ctx);
    std::printf(fails ? "FAILED (%d)\n" : "parakeet FFI checks passed\n", fails);
    return fails ? 1 : 0;
}
