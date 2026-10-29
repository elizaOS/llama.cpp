/* tools/omnivoice/tests/ffi-abi.c
 *
 * C11 test against the real libelizainference (ABI v20). Always runs:
 *   - ABI string "20" and the v17 / v20 capability probes
 *   - v18 llm_context_capacity argument validation and the no-text-GGUF error
 *   - cancel_tts / asr_cancel are no-ops with nothing in flight
 *   - mmap_evict / mmap_acquire("tts") on a bundle with no TTS weights
 *   - Paradee entries before any load (or NOT_IMPLEMENTED when not built)
 *
 * Opt-in real-model ASR cancel (skipped unless both are set):
 *   ELIZA_TEST_ASR_DIR  bundle dir containing asr/eliza-1-asr.gguf +
 *                       asr/eliza-1-asr-mmproj.gguf
 *   ELIZA_TEST_ASR_WAV  16-bit PCM mono WAV
 * It transcribes uncancelled (after one warm-up run), cancels a run 20 ms in
 * (prefill) and a run late in its decode, expecting ELIZA_ERR_CANCELLED,
 * then checks an uncancelled run still returns the same text.
 *
 * Opt-in real-model Paradee voice slots (ABI v20; skipped unless set):
 *   ELIZA_TEST_PARADEE_EN_GGUF + ELIZA_TEST_PARADEE_LEXICON  English slot
 *   ELIZA_TEST_PARADEE_ES_GGUF                              Spanish slot
 *   ELIZA_TEST_ESPEAK_DATA      eSpeak NG data (es_dict for the es slot)
 *   ELIZA_TEST_PARADEE_OUT      optional dir: writes <slot>.f32 PCM
 * Each slot synthesizes a sentence; the PCM must be non-empty, finite and
 * speech-level, and the info JSON must name the slot's language.
 */
#include "eliza-inference-ffi.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond) do { \
        ++g_checks; \
        if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

#define CHECK_RC(expr, want) do { \
        int rc_ = (expr); ++g_checks; \
        if (rc_ != (want)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: %s = %d, want %d\n", __FILE__, __LINE__, #expr, rc_, (int) (want)); } \
    } while (0)

static void clear_err(char ** err) {
    if (*err) { eliza_inference_free_string(*err); *err = NULL; }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1e6;
}

static void test_probes_and_noops(void) {
    char * err = NULL;
    CHECK(strcmp(eliza_inference_abi_version(), "20") == 0);
    CHECK_RC(eliza_inference_asr_supported(), 1);
#if defined(ELIZA_TEST_EXPECT_PARADEE)
    CHECK_RC(eliza_inference_paradee_supported(), 1);
    CHECK_RC(eliza_inference_paradee_voices_supported(), 1);
#else
    CHECK_RC(eliza_inference_paradee_supported(), 0);
    CHECK_RC(eliza_inference_paradee_voices_supported(), 0);
#endif

    char dir_template[] = "/tmp/eliza-ffi-abi-XXXXXX";
    char * dir = mkdtemp(dir_template);
    CHECK(dir != NULL);
    if (!dir) return;
    EliInferenceContext * ctx = eliza_inference_create(dir, &err);
    CHECK(ctx != NULL);
    clear_err(&err);
    if (!ctx) return;

    CHECK_RC(eliza_inference_cancel_tts(ctx, &err), ELIZA_OK);
    CHECK_RC(eliza_inference_asr_cancel(ctx, &err), ELIZA_OK);
    CHECK_RC(eliza_inference_asr_cancel(NULL, &err), ELIZA_ERR_INVALID_ARG);
    clear_err(&err);
    CHECK_RC(eliza_inference_mmap_evict(ctx, "tts", &err), ELIZA_OK);
    CHECK_RC(eliza_inference_mmap_evict(ctx, "asr", &err), ELIZA_OK);
    /* v18: capacity resolution validates its arguments and reports a bundle
     * without a text GGUF instead of returning a capacity. */
    int32_t capacity = -12345;
    CHECK_RC(eliza_inference_llm_context_capacity(NULL, 0, 0, &capacity, &err), ELIZA_ERR_INVALID_ARG);
    CHECK(capacity == -12345);
    clear_err(&err);
    CHECK_RC(eliza_inference_llm_context_capacity(ctx, 0, 0, NULL, &err), ELIZA_ERR_INVALID_ARG);
    clear_err(&err);
    CHECK_RC(eliza_inference_llm_context_capacity(ctx, 4096, 0, &capacity, &err), ELIZA_ERR_BUNDLE_INVALID);
    CHECK(capacity == -12345);
    CHECK(err != NULL);
    clear_err(&err);
    /* No TTS weights and no configured Kokoro/Paradee head. */
    CHECK_RC(eliza_inference_mmap_acquire(ctx, "tts", &err), ELIZA_ERR_BUNDLE_INVALID);
    clear_err(&err);

    float * pcm = NULL;
    size_t n = 0;
    char * ph = NULL;
#if defined(ELIZA_TEST_EXPECT_PARADEE)
    const int want = ELIZA_ERR_INVALID_ARG;
#else
    const int want = ELIZA_ERR_NOT_IMPLEMENTED;
#endif
    CHECK_RC(eliza_inference_paradee_synthesize_alloc(ctx, "hi", 2, 1.0f, &pcm, &n, &err), want);
    CHECK(pcm == NULL && n == 0);
    clear_err(&err);
    CHECK_RC(eliza_inference_paradee_synthesize_phonemes_alloc(ctx, "hi", 2, 1.0f, &pcm, &n, &err), want);
    clear_err(&err);
    CHECK_RC(eliza_inference_paradee_phonemize(ctx, "hi", 2, &ph, &err), want);
    CHECK(ph == NULL);
    clear_err(&err);
    CHECK(eliza_inference_paradee_sample_rate(ctx) < 0);
    CHECK(eliza_inference_paradee_g2p_kind(ctx) < 0);
#if !defined(ELIZA_TEST_EXPECT_PARADEE)
    CHECK_RC(eliza_inference_paradee_load(ctx, "a.gguf", "lex.bin", NULL, &err), ELIZA_ERR_NOT_IMPLEMENTED);
    clear_err(&err);
#endif

    eliza_inference_destroy(ctx);
    rmdir(dir);
}

/* ---- opt-in real-model Paradee voice slots (ABI v20) --------------- */

static void real_voice_slot(EliInferenceContext * ctx, const char * id, const char * gguf, const char * lexicon,
                            const char * espeak, const char * text, const char * language) {
    char * err = NULL;
    const int rc = eliza_inference_paradee_load_voice(ctx, id, gguf, lexicon, espeak, NULL, &err);
    CHECK_RC(rc, ELIZA_OK);
    if (rc != ELIZA_OK) { fprintf(stderr, "load %s: %s\n", id, err ? err : "?"); clear_err(&err); return; }
    char * json = NULL;
    CHECK(eliza_inference_paradee_voice_info(ctx, id, &json, &err) > 0);
    char want[64];
    snprintf(want, sizeof(want), "\"language\":\"%s\"", language);
    CHECK(json && strstr(json, want));
    fprintf(stderr, "voice %s: %s\n", id, json ? json : "?");
    eliza_inference_free_string(json);
    float * pcm = NULL;
    size_t n = 0;
    const double t0 = now_ms();
    CHECK_RC(eliza_inference_paradee_synthesize_voice_alloc(ctx, id, text, strlen(text), 1.0f, &pcm, &n, &err), ELIZA_OK);
    const double ms = now_ms() - t0;
    clear_err(&err);
    double e = 0;
    int finite = n > 0;
    for (size_t i = 0; i < n; ++i) {
        if (!(pcm[i] == pcm[i]) || pcm[i] > 4.0f || pcm[i] < -4.0f) finite = 0;
        e += (double) pcm[i] * (double) pcm[i];
    }
    const double rms = n ? sqrt(e / (double) n) : 0;
    CHECK(finite && rms > 0.01 && rms < 0.3);
    fprintf(stderr, "voice %s: %zu samples (%.2f s) in %.0f ms, rms %.4f\n", id, n, n / 24000.0, ms, rms);
    const char * out = getenv("ELIZA_TEST_PARADEE_OUT");
    if (out && pcm) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s.f32", out, id);
        FILE * f = fopen(path, "wb");
        if (f) { fwrite(pcm, sizeof(float), n, f); fclose(f); }
    }
    eliza_inference_free_pcm(pcm);
}

static void test_real_voice_slots(void) {
    const char * en = getenv("ELIZA_TEST_PARADEE_EN_GGUF");
    const char * lex = getenv("ELIZA_TEST_PARADEE_LEXICON");
    const char * es = getenv("ELIZA_TEST_PARADEE_ES_GGUF");
    const char * espeak = getenv("ELIZA_TEST_ESPEAK_DATA");
    if (!(en && lex) && !(es && espeak)) {
        fprintf(stderr, "skip: real-model Paradee voice slots (set ELIZA_TEST_PARADEE_EN_GGUF + "
                        "ELIZA_TEST_PARADEE_LEXICON and/or ELIZA_TEST_PARADEE_ES_GGUF + ELIZA_TEST_ESPEAK_DATA)\n");
        return;
    }
    char dir[] = "/tmp/eliza-ffi-voices-XXXXXX";
    if (!mkdtemp(dir)) { CHECK(0); return; }
    char * err = NULL;
    EliInferenceContext * ctx = eliza_inference_create(dir, &err);
    CHECK(ctx != NULL);
    clear_err(&err);
    if (en && lex) {
        real_voice_slot(ctx, "en", en, lex, espeak, "Good morning. Your appointment is at ten thirty.", "en");
    }
    if (es && espeak) {
        real_voice_slot(ctx, "es", es, NULL, espeak, "Buenos días. Su cita es a las diez y media.", "es");
    }
    eliza_inference_destroy(ctx);
    rmdir(dir);
}

/* ---- opt-in real-model ASR cancel ---------------------------------- */

static float * read_wav_mono16(const char * path, size_t * n_out, int * rate_out) {
    FILE * f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fclose(f);
        return NULL;
    }
    int rate = 0, channels = 0, bits = 0;
    float * out = NULL;
    for (;;) {
        unsigned char ch[8];
        if (fread(ch, 1, 8, f) != 8) break;
        uint32_t size = (uint32_t) ch[4] | ((uint32_t) ch[5] << 8) | ((uint32_t) ch[6] << 16) | ((uint32_t) ch[7] << 24);
        if (memcmp(ch, "fmt ", 4) == 0) {
            unsigned char fmt[16];
            if (size < 16 || fread(fmt, 1, 16, f) != 16) break;
            channels = fmt[2] | (fmt[3] << 8);
            rate = (int) ((uint32_t) fmt[4] | ((uint32_t) fmt[5] << 8) | ((uint32_t) fmt[6] << 16) | ((uint32_t) fmt[7] << 24));
            bits = fmt[14] | (fmt[15] << 8);
            if (size > 16) fseek(f, (long) (size - 16), SEEK_CUR);
        } else if (memcmp(ch, "data", 4) == 0) {
            if (channels != 1 || bits != 16) break;
            size_t n = size / 2;
            int16_t * raw = (int16_t *) malloc(size);
            out = (float *) malloc(n * sizeof(float));
            if (!raw || !out || fread(raw, 1, size, f) != size) { free(raw); free(out); out = NULL; break; }
            for (size_t i = 0; i < n; ++i) out[i] = (float) raw[i] / 32768.0f;
            free(raw);
            *n_out = n;
            *rate_out = rate;
            break;
        } else {
            fseek(f, (long) (size + (size & 1)), SEEK_CUR);
        }
    }
    fclose(f);
    return out;
}

struct asr_job {
    EliInferenceContext * ctx;
    const float * pcm;
    size_t n;
    int rate;
    char text[4096];
    int rc;
    double ms;
};

static void * asr_thread(void * arg) {
    struct asr_job * job = (struct asr_job *) arg;
    char * err = NULL;
    const double t0 = now_ms();
    job->rc = eliza_inference_asr_transcribe(job->ctx, job->pcm, job->n, job->rate, job->text, sizeof(job->text), &err);
    job->ms = now_ms() - t0;
    if (job->rc < 0 && job->rc != ELIZA_ERR_CANCELLED) fprintf(stderr, "asr error: %s\n", err ? err : "(none)");
    clear_err(&err);
    return NULL;
}

static void run_cancel_after(struct asr_job * job, double after_ms) {
    pthread_t th;
    char * err = NULL;
    pthread_create(&th, NULL, asr_thread, job);
    usleep((useconds_t) (after_ms * 1000.0));
    CHECK_RC(eliza_inference_asr_cancel(job->ctx, &err), ELIZA_OK);
    pthread_join(th, NULL);
}

static void test_real_asr_cancel(void) {
    const char * dir = getenv("ELIZA_TEST_ASR_DIR");
    const char * wav = getenv("ELIZA_TEST_ASR_WAV");
    if (!dir || !wav) {
        fprintf(stderr, "skip: real-model ASR cancel (set ELIZA_TEST_ASR_DIR and ELIZA_TEST_ASR_WAV)\n");
        return;
    }
    size_t n = 0;
    int rate = 0;
    float * pcm = read_wav_mono16(wav, &n, &rate);
    CHECK(pcm != NULL);
    if (!pcm) return;
    char * err = NULL;
    EliInferenceContext * ctx = eliza_inference_create(dir, &err);
    CHECK(ctx != NULL);
    clear_err(&err);
    if (!ctx) { free(pcm); return; }
    CHECK_RC(eliza_inference_mmap_acquire(ctx, "asr", &err), ELIZA_OK);
    if (err) fprintf(stderr, "acquire asr: %s\n", err);
    clear_err(&err);

    static struct asr_job warm, base, early, late, again;
    warm.ctx = ctx; warm.pcm = pcm; warm.n = n; warm.rate = rate;
    asr_thread(&warm);   /* first run pays graph / kernel warm-up */
    base = warm;
    asr_thread(&base);
    CHECK(base.rc > 0);
    fprintf(stderr, "real asr: uncancelled %.0f ms (warm-up run %.0f ms): \"%s\"\n", base.ms, warm.ms, base.text);

    /* Early cancel: lands in prefill (audio encode / audio-embedding decode). */
    early = base; early.text[0] = '\0';
    run_cancel_after(&early, 20.0);
    CHECK_RC(early.rc, ELIZA_ERR_CANCELLED);
    fprintf(stderr, "real asr: cancel at 20 ms -> rc=%d after %.0f ms\n", early.rc, early.ms);

    /* Late cancel: aims at the token decode. Run time varies with machine
     * load, so if the run finished before the cancel landed, retry earlier.
     * A run that was not cancelled must still return the full transcript. */
    const double fractions[] = { 0.7, 0.5, 0.35, 0.2 };
    int late_cancelled = 0;
    for (size_t i = 0; i < sizeof(fractions) / sizeof(fractions[0]) && !late_cancelled; ++i) {
        late = base; late.text[0] = '\0';
        const double at = base.ms * fractions[i];
        run_cancel_after(&late, at);
        fprintf(stderr, "real asr: cancel at %.0f ms -> rc=%d after %.0f ms\n", at, late.rc, late.ms);
        if (late.rc == ELIZA_ERR_CANCELLED) {
            late_cancelled = 1;
        } else {
            CHECK(late.rc == base.rc && strcmp(late.text, base.text) == 0);
        }
    }
    CHECK(late_cancelled);

    /* The cancel flag does not leak into the next transcription. */
    again = base; again.text[0] = '\0';
    asr_thread(&again);
    CHECK(again.rc == base.rc);
    CHECK(strcmp(again.text, base.text) == 0);

    eliza_inference_destroy(ctx);
    free(pcm);
}

int main(void) {
    test_probes_and_noops();
    test_real_asr_cancel();
    test_real_voice_slots();
    fprintf(stderr, "%s: %d checks, %d failures\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
