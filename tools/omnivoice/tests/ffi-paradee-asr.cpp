// tools/omnivoice/tests/ffi-paradee-asr.cpp
//
// Unit test for the ABI v17 additions in src/eliza-inference-ffi.cpp:
//   - Paradee load / g2p kind / phonemize / synth, <=510-id chunking
//   - cancel_tts during an in-flight Paradee synth, plus a load/synth/
//     cancel/evict race across threads
//   - mmap_evict("tts") frees Paradee + lexicon, next synth or
//     mmap_acquire("tts") reloads
//   - eliza_inference_asr_cancel during prefill and during decode, over a
//     scripted ASR engine that polls the same abort callback llama_decode
//     uses
//   - ABI string "19"
//
// The bridge is compiled into this TU (ELIZA_INFERENCE_TEST_HOOKS) against
// the Paradee / misaki doubles in tests/paradee-contract/, so no weights are
// needed. The real-library and real-model checks are in tests/ffi-abi.c.

#include "../src/eliza-inference-ffi.cpp"

#include "fake-paradee-control.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <thread>

/* Atomic: CHECK runs on the worker threads of the cancel / race tests too, and
 * a lost ++g_failures there would turn a failure into a PASS. */
static std::atomic<int> g_failures{0};
static std::atomic<int> g_checks{0};

#define CHECK(cond) do { \
        ++g_checks; \
        if (!(cond)) { \
            ++g_failures; \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        } \
    } while (0)

#define CHECK_RC(rc, want) do { \
        const int _rc = (rc); \
        ++g_checks; \
        if (_rc != (want)) { \
            ++g_failures; \
            std::fprintf(stderr, "FAIL %s:%d: %s = %d, want %d\n", __FILE__, __LINE__, #rc, _rc, (int) (want)); \
        } \
    } while (0)

namespace fs = std::filesystem;
using clock_type = std::chrono::steady_clock;

static void clear_err(char ** err) {
    if (*err) {
        eliza_inference_free_string(*err);
        *err = nullptr;
    }
}

static fs::path make_bundle(const char * name) {
    fs::path dir = fs::temp_directory_path() / (std::string("eliza-ffi-test-") + name + "-" +
        std::to_string((long long) clock_type::now().time_since_epoch().count()));
    fs::create_directories(dir);
    return dir;
}

static bool wait_until(const std::atomic<bool> & flag, int timeout_ms) {
    const auto deadline = clock_type::now() + std::chrono::milliseconds(timeout_ms);
    while (!flag.load()) {
        if (clock_type::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

static int synth_text(EliInferenceContext * ctx, const std::string & text, size_t * n_out, char ** err) {
    float * pcm = nullptr;
    size_t n = 0;
    const int rc = eliza_inference_paradee_synthesize_alloc(ctx, text.c_str(), text.size(), 1.0f, &pcm, &n, err);
    if (rc == ELIZA_OK && n > 0) CHECK(pcm != nullptr);
    if (rc != ELIZA_OK) CHECK(pcm == nullptr && n == 0);
    eliza_inference_free_pcm(pcm);
    if (n_out) *n_out = n;
    return rc;
}

static void test_abi_and_probes() {
    CHECK(std::string(eliza_inference_abi_version()) == "19");
    CHECK_RC(eliza_inference_paradee_supported(), 1);
    CHECK_RC(eliza_inference_asr_supported(), 1);
    CHECK_RC(eliza_inference_asr_stream_supported(), 0);
}

static void test_paradee_load_and_synth() {
    fake_paradee_reset();
    const fs::path bundle = make_bundle("load");
    char * err = nullptr;
    EliInferenceContext * ctx = eliza_inference_create(bundle.string().c_str(), &err);
    CHECK(ctx != nullptr);
    clear_err(&err);

    // Nothing loaded yet.
    CHECK_RC(synth_text(ctx, "hello", nullptr, &err), ELIZA_ERR_INVALID_ARG);
    CHECK(err != nullptr);
    clear_err(&err);
    CHECK(eliza_inference_paradee_sample_rate(ctx) < 0);
    CHECK(eliza_inference_paradee_g2p_kind(ctx) < 0);
    CHECK_RC(eliza_inference_paradee_load(ctx, nullptr, "lex.bin", nullptr, &err), ELIZA_ERR_INVALID_ARG);
    clear_err(&err);

    // A failed load reports BUNDLE_INVALID and leaves the ctx unconfigured.
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/missing.gguf", "/x/lex.bin", nullptr, &err), ELIZA_ERR_BUNDLE_INVALID);
    CHECK(err != nullptr && std::string(err).find("paradee_load_model") != std::string::npos);
    clear_err(&err);
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/paradee.gguf", "/x/missing-lex.bin", nullptr, &err), ELIZA_ERR_BUNDLE_INVALID);
    clear_err(&err);
    CHECK_RC(fake_paradee().live_models.load(), 0);

    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/paradee.gguf", "/x/lex.bin", nullptr, &err), ELIZA_OK);
    CHECK_RC(fake_paradee().model_loads.load(), 2);   // the failed lexicon attempt loaded a model too
    CHECK_RC(fake_paradee().live_models.load(), 1);
    // Same paths: no reload.
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/paradee.gguf", "/x/lex.bin", nullptr, &err), ELIZA_OK);
    CHECK_RC(fake_paradee().model_loads.load(), 2);
    CHECK_RC(eliza_inference_paradee_sample_rate(ctx), 24000);
    CHECK_RC(eliza_inference_paradee_g2p_kind(ctx), ELIZA_PARADEE_G2P_RULES);
    // A failed reload keeps the loaded model.
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/missing.gguf", "/x/lex.bin", nullptr, &err), ELIZA_ERR_BUNDLE_INVALID);
    clear_err(&err);
    CHECK_RC(eliza_inference_paradee_sample_rate(ctx), 24000);
    // eSpeak data dir -> ESPEAK fallback kind (different config -> reload).
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/paradee.gguf", "/x/lex.bin", "/x/espeak-ng-data", &err), ELIZA_OK);
    CHECK_RC(eliza_inference_paradee_g2p_kind(ctx), ELIZA_PARADEE_G2P_ESPEAK);
    CHECK_RC(fake_paradee().live_models.load(), 1);
    CHECK_RC(fake_paradee().live_lexicons.load(), 1);

    // Phonemize.
    char * ph = nullptr;
    CHECK_RC(eliza_inference_paradee_phonemize(ctx, "Hello, World", 12, &ph, &err), 12);
    CHECK(ph != nullptr && std::string(ph) == "hello, world");
    eliza_inference_free_string(ph);
    ph = nullptr;
    CHECK_RC(eliza_inference_paradee_phonemize(ctx, "\x01" "fail", 5, &ph, &err), ELIZA_ERR_FFI_FAULT);
    CHECK(ph == nullptr);
    clear_err(&err);

    // Short text: one forward pass, 0-padded input, exact sample count.
    {
        std::lock_guard<std::mutex> lock(fake_paradee().sizes_mutex);
        fake_paradee().input_sizes.clear();
    }
    size_t n = 0;
    CHECK_RC(synth_text(ctx, "Hi there.", &n, &err), ELIZA_OK);
    CHECK_RC((int) n, 9 * FAKE_PARADEE_SAMPLES_PER_ID);
    {
        std::lock_guard<std::mutex> lock(fake_paradee().sizes_mutex);
        CHECK(fake_paradee().input_sizes.size() == 1 && fake_paradee().input_sizes[0] == 11);
    }
    // Text whose phonemes are all out of vocab / dropped -> zero samples, OK.
    CHECK_RC(synth_text(ctx, "12345", &n, &err), ELIZA_OK);
    CHECK_RC((int) n, 0);
    CHECK_RC(synth_text(ctx, "\x01" "fail", &n, &err), ELIZA_ERR_FFI_FAULT);
    clear_err(&err);

    // Phoneme input path: '#' is out of vocab and dropped.
    {
        float * pcm = nullptr;
        const std::string phon = "h#\xC9\x99lo\xCA\x8A";   // h # ə l o ʊ
        CHECK_RC(eliza_inference_paradee_synthesize_phonemes_alloc(ctx, phon.c_str(), phon.size(), 1.2f, &pcm, &n, &err), ELIZA_OK);
        CHECK_RC((int) n, 5 * FAKE_PARADEE_SAMPLES_PER_ID);
        CHECK(pcm != nullptr && pcm[0] == (float) (43 + ('h' - 'a')) / 1000.0f);
        eliza_inference_free_pcm(pcm);
    }
    // Out-of-range speeds are clamped to the library's range, not rejected.
    for (float speed : { 0.01f, 100.0f }) {
        float * pcm = nullptr;
        CHECK_RC(eliza_inference_paradee_synthesize_alloc(ctx, "hi", 2, speed, &pcm, &n, &err), ELIZA_OK);
        CHECK_RC((int) n, 2 * FAKE_PARADEE_SAMPLES_PER_ID);
        eliza_inference_free_pcm(pcm);
    }
    CHECK_RC(eliza_inference_paradee_synthesize_alloc(ctx, "x", 1, 1.0f, nullptr, &n, &err), ELIZA_ERR_INVALID_ARG);
    clear_err(&err);

    eliza_inference_destroy(ctx);
    CHECK_RC(fake_paradee().live_models.load(), 0);
    CHECK_RC(fake_paradee().live_lexicons.load(), 0);
    fs::remove_all(bundle);
}

static void test_paradee_streaming() {
    const fs::path bundle = make_bundle("paradee-streaming");
    char * err = nullptr;
    EliInferenceContext * ctx = eliza_inference_create(bundle.string().c_str(), &err);
    CHECK_RC(eliza_inference_paradee_stream_supported(), 1);
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/paradee.gguf", "/x/lex.bin", nullptr, &err), ELIZA_OK);
    std::string text;
    for (int i = 0; i < 40; ++i) text += "the quick brown fox jumps now. ";
    float * expected = nullptr; size_t count = 0;
    CHECK_RC(eliza_inference_paradee_synthesize_alloc(ctx, text.data(), text.size(), 1, &expected, &count, &err), ELIZA_OK);
    struct Output { std::vector<float> pcm; int final = 0; int calls = 0; bool cancel = false; } output;
    auto receive = [](const float * pcm, size_t n, int final, void * user) {
        auto & out = *static_cast<Output *>(user);
        if (final) { ++out.final; CHECK(n == 0 && pcm == nullptr); return 0; }
        CHECK(n <= 24000 && out.final == 0);
        ++out.calls;
        out.pcm.insert(out.pcm.end(), pcm, pcm + n);
        return out.cancel ? 1 : 0;
    };
    CHECK_RC(eliza_inference_paradee_synthesize_stream(ctx, text.data(), text.size(), 1, receive, &output, &err), ELIZA_OK);
    CHECK(output.calls > 1 && output.final == 1);
    CHECK(output.pcm == std::vector<float>(expected, expected + count));
    eliza_inference_free_pcm(expected);
    output = Output{}; output.cancel = true;
    CHECK_RC(eliza_inference_paradee_synthesize_stream(ctx, text.data(), text.size(), 1, receive, &output, &err), ELIZA_ERR_CANCELLED);
    CHECK(output.calls == 1 && output.final == 0);
    clear_err(&err);
    output = Output{};
    CHECK_RC(eliza_inference_paradee_synthesize_stream(ctx, "12345", 5, 1, receive, &output, &err), ELIZA_OK);
    CHECK(output.calls == 0 && output.final == 1);
    CHECK_RC(eliza_inference_paradee_synthesize_stream(ctx, "hi", 2, 1, nullptr, &output, &err), ELIZA_ERR_INVALID_ARG);
    clear_err(&err);
    eliza_inference_destroy(ctx);
    fs::remove_all(bundle);
}

static void test_paradee_chunking() {
    fake_paradee_reset();
    const fs::path bundle = make_bundle("chunk");
    char * err = nullptr;
    EliInferenceContext * ctx = eliza_inference_create(bundle.string().c_str(), &err);
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/paradee.gguf", "/x/lex.bin", nullptr, &err), ELIZA_OK);

    // 40 sentences of 30 chars each (1200 ids): must split at sentence ends,
    // every chunk <= 510 ids + 2 pads, and lose nothing.
    std::string text;
    for (int i = 0; i < 40; ++i) text += "the quick brown fox jumps now. ";
    const size_t expected_ids = text.size();
    {
        std::lock_guard<std::mutex> lock(fake_paradee().sizes_mutex);
        fake_paradee().input_sizes.clear();
    }
    size_t n = 0;
    CHECK_RC(synth_text(ctx, text, &n, &err), ELIZA_OK);
    CHECK_RC((int) n, (int) (expected_ids * FAKE_PARADEE_SAMPLES_PER_ID));
    {
        std::lock_guard<std::mutex> lock(fake_paradee().sizes_mutex);
        const auto & sizes = fake_paradee().input_sizes;
        CHECK(sizes.size() == 3);
        size_t total = 0;
        for (size_t s : sizes) {
            CHECK(s <= 512);
            total += s - 2;
        }
        CHECK(total == expected_ids);
        // Greedy sentence packing: 31-char sentences, 16 per 510-id chunk.
        CHECK(!sizes.empty() && sizes[0] == 16 * 31 + 2);
    }

    // One 1300-letter "word": no boundaries at all -> hard slices.
    const std::string word(1300, 'a');
    {
        std::lock_guard<std::mutex> lock(fake_paradee().sizes_mutex);
        fake_paradee().input_sizes.clear();
    }
    CHECK_RC(synth_text(ctx, word, &n, &err), ELIZA_OK);
    CHECK_RC((int) n, 1300 * FAKE_PARADEE_SAMPLES_PER_ID);
    {
        std::lock_guard<std::mutex> lock(fake_paradee().sizes_mutex);
        const auto & sizes = fake_paradee().input_sizes;
        CHECK(sizes.size() == 3 && sizes[0] == 512 && sizes[1] == 512 && sizes[2] == 1300 - 1020 + 2);
    }

    // Clause fallback: one 600-char sentence with commas.
    std::string clause_text;
    for (int i = 0; i < 20; ++i) clause_text += "and then some more words, ";
    clause_text += "done.";
    {
        std::lock_guard<std::mutex> lock(fake_paradee().sizes_mutex);
        fake_paradee().input_sizes.clear();
    }
    CHECK_RC(synth_text(ctx, clause_text, &n, &err), ELIZA_OK);
    CHECK_RC((int) n, (int) (clause_text.size() * FAKE_PARADEE_SAMPLES_PER_ID));
    {
        std::lock_guard<std::mutex> lock(fake_paradee().sizes_mutex);
        const auto & sizes = fake_paradee().input_sizes;
        CHECK(sizes.size() == 2);
        // 26-char clauses: 19 fit in 510.
        CHECK(!sizes.empty() && sizes[0] == 19 * 26 + 2);
    }

    eliza_inference_destroy(ctx);
    fs::remove_all(bundle);
}

static void test_paradee_cancel() {
    fake_paradee_reset();
    const fs::path bundle = make_bundle("cancel");
    char * err = nullptr;
    EliInferenceContext * ctx = eliza_inference_create(bundle.string().c_str(), &err);
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/paradee.gguf", "/x/lex.bin", nullptr, &err), ELIZA_OK);

    // In-flight cancel stops the forward pass well before it would finish.
    fake_paradee().block_ms = 5000;
    int rc = 1;
    char * thread_err = nullptr;
    const auto t0 = clock_type::now();
    std::thread synth([&]() { rc = synth_text(ctx, "a long utterance.", nullptr, &thread_err); });
    CHECK(wait_until(fake_paradee().in_flight, 2000));
    CHECK_RC(eliza_inference_cancel_tts(ctx, &err), ELIZA_OK);
    synth.join();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clock_type::now() - t0).count();
    CHECK_RC(rc, ELIZA_ERR_CANCELLED);
    CHECK(thread_err != nullptr && std::string(thread_err).find("cancelled") != std::string::npos);
    clear_err(&thread_err);
    CHECK(ms < 2000);

    // The flag does not carry over into the next call.
    fake_paradee().block_ms = 0;
    CHECK_RC(synth_text(ctx, "next.", nullptr, &err), ELIZA_OK);
    // Cancel with nothing in flight is not an error and does not poison the
    // next call either.
    CHECK_RC(eliza_inference_cancel_tts(ctx, &err), ELIZA_OK);
    CHECK_RC(synth_text(ctx, "after idle cancel.", nullptr, &err), ELIZA_OK);

    // Cancel during a multi-chunk synth: the current chunk stops and no later
    // chunk runs.
    fake_paradee().block_ms = 300;
    std::string long_text;
    for (int i = 0; i < 40; ++i) long_text += "the quick brown fox jumps now. ";
    const int calls_before = fake_paradee().synth_calls.load();
    fake_paradee().in_flight = false;
    std::thread synth2([&]() { rc = synth_text(ctx, long_text, nullptr, &thread_err); });
    CHECK(wait_until(fake_paradee().in_flight, 2000));
    CHECK_RC(eliza_inference_cancel_tts(ctx, &err), ELIZA_OK);
    synth2.join();
    clear_err(&thread_err);
    CHECK_RC(rc, ELIZA_ERR_CANCELLED);
    CHECK_RC(fake_paradee().synth_calls.load() - calls_before, 1);

    // Flag isolation: an OmniVoice call starting (ElizaScopedTtsForward
    // clears tts_cancel) must not clear a cancel aimed at the Paradee call in
    // flight.
    fake_paradee().block_ms = 3000;
    fake_paradee().in_flight = false;
    std::thread synth3([&]() { rc = synth_text(ctx, "isolated.", nullptr, &thread_err); });
    CHECK(wait_until(fake_paradee().in_flight, 2000));
    CHECK_RC(eliza_inference_cancel_tts(ctx, &err), ELIZA_OK);
    { ElizaScopedTtsForward omnivoice_call_starts(ctx); }
    synth3.join();
    clear_err(&thread_err);
    CHECK_RC(rc, ELIZA_ERR_CANCELLED);

    eliza_inference_destroy(ctx);
    fs::remove_all(bundle);
}

static void test_paradee_race() {
    fake_paradee_reset();
    const fs::path bundle = make_bundle("race");
    char * err = nullptr;
    EliInferenceContext * ctx = eliza_inference_create(bundle.string().c_str(), &err);
    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/a.gguf", "/x/lex.bin", nullptr, &err), ELIZA_OK);
    fake_paradee().block_ms = 2;

    std::atomic<int> bad{0};
    std::atomic<int> ok{0};
    std::atomic<int> cancelled{0};
    auto worker = [&](int seed) {
        std::mt19937 rng((unsigned) seed);
        char * e = nullptr;
        for (int i = 0; i < 150; ++i) {
            int rc = ELIZA_OK;
            switch (rng() % 5) {
                case 0:
                    rc = eliza_inference_paradee_load(ctx, (rng() & 1) ? "/x/a.gguf" : "/x/b.gguf", "/x/lex.bin", nullptr, &e);
                    break;
                case 1:
                case 2:
                    rc = synth_text(ctx, "race text, with a clause. and more.", nullptr, &e);
                    break;
                case 3:
                    rc = eliza_inference_cancel_tts(ctx, &e);
                    break;
                default:
                    rc = (rng() & 1) ? eliza_inference_mmap_evict(ctx, "tts", &e)
                                     : eliza_inference_mmap_acquire(ctx, "tts", &e);
                    break;
            }
            if (rc == ELIZA_OK) ok.fetch_add(1);
            else if (rc == ELIZA_ERR_CANCELLED) cancelled.fetch_add(1);
            else {
                bad.fetch_add(1);
                std::fprintf(stderr, "race: unexpected rc=%d err=%s\n", rc, e ? e : "(none)");
            }
            if (e) { eliza_inference_free_string(e); e = nullptr; }
        }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < 6; ++t) threads.emplace_back(worker, 1234 + t);
    for (auto & t : threads) t.join();
    CHECK_RC(bad.load(), 0);
    CHECK(ok.load() > 0);
    CHECK(fake_paradee().live_models.load() <= 1);
    CHECK(fake_paradee().live_lexicons.load() <= 1);
    std::fprintf(stderr, "race: ok=%d cancelled=%d\n", ok.load(), cancelled.load());

    // Still usable afterwards.
    fake_paradee().block_ms = 0;
    CHECK_RC(synth_text(ctx, "after the race.", nullptr, &err), ELIZA_OK);
    eliza_inference_destroy(ctx);
    CHECK_RC(fake_paradee().live_models.load(), 0);
    fs::remove_all(bundle);
}

static void test_evict_reload() {
    fake_paradee_reset();
    const fs::path bundle = make_bundle("evict");
    char * err = nullptr;
    EliInferenceContext * ctx = eliza_inference_create(bundle.string().c_str(), &err);

    // No TTS head configured and no OmniVoice weights: acquire fails as before.
    CHECK_RC(eliza_inference_mmap_acquire(ctx, "tts", &err), ELIZA_ERR_BUNDLE_INVALID);
    clear_err(&err);
    CHECK_RC(eliza_inference_mmap_evict(ctx, "tts", &err), ELIZA_OK);

    CHECK_RC(eliza_inference_paradee_load(ctx, "/x/paradee.gguf", "/x/lex.bin", nullptr, &err), ELIZA_OK);
    CHECK_RC(fake_paradee().model_loads.load(), 1);

    // Evict frees model + lexicon; rate / g2p kind still answer.
    CHECK_RC(eliza_inference_mmap_evict(ctx, "tts", &err), ELIZA_OK);
    CHECK_RC(fake_paradee().live_models.load(), 0);
    CHECK_RC(fake_paradee().live_lexicons.load(), 0);
    CHECK_RC(eliza_inference_paradee_sample_rate(ctx), 24000);
    CHECK_RC(eliza_inference_paradee_g2p_kind(ctx), ELIZA_PARADEE_G2P_RULES);
    // Evicting "asr" leaves TTS alone.
    CHECK_RC(eliza_inference_mmap_evict(ctx, "asr", &err), ELIZA_OK);

    // Next synth reloads lazily.
    size_t n = 0;
    CHECK_RC(synth_text(ctx, "reloaded.", &n, &err), ELIZA_OK);
    CHECK_RC((int) n, 9 * FAKE_PARADEE_SAMPLES_PER_ID);
    CHECK_RC(fake_paradee().model_loads.load(), 2);
    CHECK_RC(fake_paradee().lexicon_loads.load(), 2);
    CHECK_RC(fake_paradee().live_models.load(), 1);

    // Phonemize reloads too.
    CHECK_RC(eliza_inference_mmap_evict(ctx, "tts", &err), ELIZA_OK);
    char * ph = nullptr;
    CHECK_RC(eliza_inference_paradee_phonemize(ctx, "ab", 2, &ph, &err), 2);
    eliza_inference_free_string(ph);
    CHECK_RC(fake_paradee().model_loads.load(), 3);

    // acquire("tts") reloads; with no OmniVoice weights in the bundle it
    // still succeeds because a Paradee head is configured. Repeat acquire is
    // a no-op.
    CHECK_RC(eliza_inference_mmap_evict(ctx, "tts", &err), ELIZA_OK);
    CHECK_RC(eliza_inference_mmap_acquire(ctx, "tts", &err), ELIZA_OK);
    clear_err(&err);
    CHECK_RC(fake_paradee().model_loads.load(), 4);
    CHECK_RC(eliza_inference_mmap_acquire(ctx, "tts", &err), ELIZA_OK);
    CHECK_RC(fake_paradee().model_loads.load(), 4);
    // OmniVoice entries still report the region as not acquired.
    float buf[16];
    CHECK_RC(eliza_inference_tts_synthesize(ctx, "x", 1, nullptr, buf, 16, &err), ELIZA_ERR_INVALID_ARG);
    clear_err(&err);

    // A Paradee GGUF under tts/ is never picked as the OmniVoice model.
    fs::create_directories(bundle / "tts" / "paradee");
    { std::ofstream(bundle / "tts" / "paradee" / "paradee-8m.gguf") << "x"; }
    { std::ofstream(bundle / "tts" / "paradee-8m-v1.0.gguf") << "x"; }
    std::string tts_model, codec_model;
    CHECK(!eliza_pick_voice_files(bundle, tts_model, codec_model));

    // Evict while a synth is in flight: the synth finishes, then the model is freed.
    fake_paradee().block_ms = 200;
    fake_paradee().in_flight = false;
    int rc = 1;
    char * thread_err = nullptr;
    std::thread synth([&]() { rc = synth_text(ctx, "in flight.", &n, &thread_err); });
    CHECK(wait_until(fake_paradee().in_flight, 2000));
    CHECK_RC(eliza_inference_mmap_evict(ctx, "tts", &err), ELIZA_OK);
    synth.join();
    clear_err(&thread_err);
    CHECK_RC(rc, ELIZA_OK);
    CHECK_RC((int) n, 10 * FAKE_PARADEE_SAMPLES_PER_ID);
    CHECK_RC(fake_paradee().live_models.load(), 0);

    eliza_inference_destroy(ctx);
    fs::remove_all(bundle);
}

// Scripted ASR engine. Prefill: `n_chunks` chunks; chunk `slow_chunk` spins
// until the abort callback fires (like a long audio-embedding llama_decode)
// and returns 2. Decode: emits `tokens` then EOG; when `slow_decode` is set,
// each decode() polls the abort callback for up to 5 s (like a long
// llama_decode) before returning.
struct ScriptedAsrEngine final : EliAsrEngine {
    EliInferenceContext * ctx;
    size_t chunks = 3;
    long slow_chunk = -1;
    std::vector<std::string> tokens;
    bool infinite = false;
    bool slow_decode = false;
    size_t pos = 0;
    std::atomic<bool> in_prefill{false};
    std::atomic<bool> in_decode{false};
    std::atomic<int> decoded{0};

    explicit ScriptedAsrEngine(EliInferenceContext * c) : ctx(c) {}
    size_t n_chunks() override { return chunks; }
    int32_t eval_chunk(size_t i, bool) override {
        if ((long) i != slow_chunk) return 0;
        in_prefill = true;
        for (int t = 0; t < 5000; ++t) {
            if (eliza_test_asr_abort_requested(ctx)) return 2;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return 0;
    }
    llama_token sample() override { return (llama_token) pos; }
    bool is_eog(llama_token t) override { return !infinite && (size_t) t >= tokens.size(); }
    std::string piece(llama_token t) override { return infinite ? std::string("a ") : tokens[(size_t) t]; }
    void accept(llama_token) override {}
    int32_t decode(llama_token) override {
        ++pos;
        decoded.fetch_add(1);
        in_decode = true;
        if (slow_decode) {
            for (int t = 0; t < 5000; ++t) {
                if (eliza_test_asr_abort_requested(ctx)) return 2;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        return 0;
    }
};

static void test_asr_cancel() {
    const fs::path bundle = make_bundle("asr");
    char * err = nullptr;
    EliInferenceContext * ctx = eliza_inference_create(bundle.string().c_str(), &err);
    char text[256];

    CHECK_RC(eliza_inference_asr_cancel(nullptr, &err), ELIZA_ERR_INVALID_ARG);
    clear_err(&err);

    // Uncancelled run produces the transcript.
    {
        ScriptedAsrEngine e(ctx);
        e.tokens = { "hello", " world" };
        CHECK_RC(eliza_test_asr_run_engine(ctx, &e, 64, text, sizeof(text), &err), 11);
        CHECK(std::string(text) == "hello world");
    }
    // A cancel with nothing in flight does not affect the next call.
    CHECK_RC(eliza_inference_asr_cancel(ctx, &err), ELIZA_OK);
    {
        ScriptedAsrEngine e(ctx);
        e.tokens = { "still", " fine" };
        CHECK_RC(eliza_test_asr_run_engine(ctx, &e, 64, text, sizeof(text), &err), 10);
    }

    // Cancel during prefill (abort callback inside a chunk eval).
    {
        ScriptedAsrEngine e(ctx);
        e.slow_chunk = 1;
        e.tokens = { "never" };
        int rc = 1;
        char * terr = nullptr;
        const auto t0 = clock_type::now();
        std::thread th([&]() { rc = eliza_test_asr_run_engine(ctx, &e, 64, text, sizeof(text), &terr); });
        CHECK(wait_until(e.in_prefill, 2000));
        CHECK_RC(eliza_inference_asr_cancel(ctx, &err), ELIZA_OK);
        th.join();
        CHECK_RC(rc, ELIZA_ERR_CANCELLED);
        CHECK(terr != nullptr && std::string(terr).find("cancelled") != std::string::npos);
        clear_err(&terr);
        CHECK_RC(e.decoded.load(), 0);
        CHECK(std::chrono::duration_cast<std::chrono::milliseconds>(clock_type::now() - t0).count() < 2000);
    }

    // Cancel during decode (per-token check): an endless token stream.
    {
        ScriptedAsrEngine e(ctx);
        e.infinite = true;
        int rc = 1;
        char * terr = nullptr;
        char big[1 << 16];
        std::thread th([&]() { rc = eliza_test_asr_run_engine(ctx, &e, 1 << 20, big, sizeof(big), &terr); });
        CHECK(wait_until(e.in_decode, 2000));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        CHECK_RC(eliza_inference_asr_cancel(ctx, &err), ELIZA_OK);
        th.join();
        clear_err(&terr);
        CHECK_RC(rc, ELIZA_ERR_CANCELLED);
        CHECK(e.decoded.load() > 0);
    }

    // Cancel during one long decode step (abort callback inside llama_decode).
    {
        ScriptedAsrEngine e(ctx);
        e.tokens = { "one", " two", " three" };
        e.slow_decode = true;
        int rc = 1;
        char * terr = nullptr;
        const auto t0 = clock_type::now();
        std::thread th([&]() { rc = eliza_test_asr_run_engine(ctx, &e, 64, text, sizeof(text), &terr); });
        CHECK(wait_until(e.in_decode, 2000));
        CHECK_RC(eliza_inference_asr_cancel(ctx, &err), ELIZA_OK);
        th.join();
        clear_err(&terr);
        CHECK_RC(rc, ELIZA_ERR_CANCELLED);
        CHECK_RC(e.decoded.load(), 1);
        CHECK(std::chrono::duration_cast<std::chrono::milliseconds>(clock_type::now() - t0).count() < 2000);
    }

    // A cancel issued while a second call waits for the ASR lock stops the
    // call holding the lock, not the waiter.
    {
        ScriptedAsrEngine first(ctx);
        first.slow_chunk = 0;
        ScriptedAsrEngine second(ctx);
        second.tokens = { "second" };
        int rc1 = 1, rc2 = 1;
        char * e1 = nullptr;
        char * e2 = nullptr;
        char text2[64];
        std::thread t1([&]() { rc1 = eliza_test_asr_run_engine(ctx, &first, 64, text, sizeof(text), &e1); });
        CHECK(wait_until(first.in_prefill, 2000));
        std::thread t2([&]() { rc2 = eliza_test_asr_run_engine(ctx, &second, 64, text2, sizeof(text2), &e2); });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK_RC(eliza_inference_asr_cancel(ctx, &err), ELIZA_OK);
        t1.join();
        t2.join();
        clear_err(&e1);
        clear_err(&e2);
        CHECK_RC(rc1, ELIZA_ERR_CANCELLED);
        CHECK_RC(rc2, 6);
    }

    // The real entry still requires an acquired ASR region.
    float pcm[160] = {0};
    CHECK_RC(eliza_inference_asr_transcribe(ctx, pcm, 160, 16000, text, sizeof(text), &err), ELIZA_ERR_INVALID_ARG);
    clear_err(&err);

    eliza_inference_destroy(ctx);
    fs::remove_all(bundle);
}

int main() {
    test_abi_and_probes();
    test_paradee_load_and_synth();
    test_paradee_chunking();
    test_paradee_streaming();
    test_paradee_cancel();
    test_paradee_race();
    test_evict_reload();
    test_asr_cancel();
    std::fprintf(stderr, "%s: %d checks, %d failures\n",
        g_failures.load() ? "FAIL" : "PASS", g_checks.load(), g_failures.load());
    return g_failures.load() ? 1 : 0;
}
