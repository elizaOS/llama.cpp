/*
 * Diarizer GGUF metadata contract for the vendored fused reader.
 *
 * The fused reader consumes pyannote LSTM gates in the converter's IFGO
 * packing and is fail-closed: artifacts must carry explicit
 * `voice_diarizer.converter_epoch >= 2` and `lstm_gate_order == "IFGO"`
 * metadata, or `voice_diarizer_open` rejects them loudly before tensor load.
 * Epoch-less legacy artifacts (the previously published IOFC bake) and
 * explicit non-IFGO artifacts must never silently scramble gates
 * (elizaOS/eliza#11377).
 *
 * The test writes tiny metadata-only GGUF files, so no pyannote artifact is
 * required. Guard rejections are distinguished from ordinary
 * missing-tensor failures by capturing stderr.
 */

#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE 700

#include "voice_classifier/voice_classifier.h"
#include "voice_gguf_loader.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define VC_GGUF_MAGIC "GGUF"
#define VC_GGUF_VERSION 3
#define DIAR_CONVERTER_EPOCH 2
#define DIAR_WINDOW_SAMPLES 80000
#define DIAR_FRAMES_PER_WINDOW 293
#define DIAR_LSTM_LAYERS 4
#define DIAR_LSTM_HIDDEN 128
#define DIAR_LINEAR0_OUT 128
#define DIAR_LINEAR1_OUT 128

enum vc_gguf_type {
    VC_GGUF_TYPE_UINT32 = 4,
    VC_GGUF_TYPE_STRING = 8,
};

static void w_u32(FILE * f, uint32_t v) {
    fwrite(&v, sizeof(v), 1, f);
}

static void w_u64(FILE * f, uint64_t v) {
    fwrite(&v, sizeof(v), 1, f);
}

static void w_i64(FILE * f, int64_t v) {
    fwrite(&v, sizeof(v), 1, f);
}

static void w_str(FILE * f, const char * s) {
    const uint64_t n = strlen(s);
    w_u64(f, n);
    fwrite(s, 1, n, f);
}

static void w_kv_u32(FILE * f, const char * key, uint32_t val) {
    w_str(f, key);
    w_u32(f, VC_GGUF_TYPE_UINT32);
    w_u32(f, val);
}

static void w_kv_str(FILE * f, const char * key, const char * val) {
    w_str(f, key);
    w_u32(f, VC_GGUF_TYPE_STRING);
    w_str(f, val);
}

/* Write a metadata-only diarizer GGUF. Pass converter_epoch < 0 or
 * gate_order == NULL to omit that key (legacy artifact shape). */
static int write_diarizer_meta(const char * path, int converter_epoch, const char * gate_order) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return -1;
    }

    int64_t kv_count = 9;
    if (converter_epoch >= 0) {
        ++kv_count;
    }
    if (gate_order) {
        ++kv_count;
    }

    fwrite(VC_GGUF_MAGIC, 1, 4, f);
    w_u32(f, VC_GGUF_VERSION);
    w_i64(f, 0);
    w_i64(f, kv_count);

    w_kv_u32(f, "voice_diarizer.sample_rate", VOICE_CLASSIFIER_SAMPLE_RATE_HZ);
    w_kv_u32(f, "voice_diarizer.num_classes", VOICE_DIARIZER_NUM_CLASSES);
    w_kv_u32(f, "voice_diarizer.window_samples", DIAR_WINDOW_SAMPLES);
    w_kv_u32(f, "voice_diarizer.frames_per_window", DIAR_FRAMES_PER_WINDOW);
    if (converter_epoch >= 0) {
        w_kv_u32(f, "voice_diarizer.converter_epoch", (uint32_t)converter_epoch);
    }
    w_kv_u32(f, "voice_diarizer.lstm_layers", DIAR_LSTM_LAYERS);
    w_kv_u32(f, "voice_diarizer.lstm_hidden", DIAR_LSTM_HIDDEN);
    w_kv_u32(f, "voice_diarizer.linear0_out", DIAR_LINEAR0_OUT);
    w_kv_u32(f, "voice_diarizer.linear1_out", DIAR_LINEAR1_OUT);
    w_kv_str(f, "voice_diarizer.variant", "pyannote-segmentation-3.0");
    if (gate_order) {
        w_kv_str(f, "voice_diarizer.lstm_gate_order", gate_order);
    }

    /* Pad to the GGUF data alignment (32) so the tensor mapper accepts the
     * zero-tensor file and metadata-guard passes reach tensor resolution. */
    long pos = ftell(f);
    while (pos > 0 && (pos % 32) != 0) {
        fputc(0, f);
        ++pos;
    }

    fclose(f);
    return 0;
}

/* Run voice_diarizer_open with stderr captured to err_path. Returns the
 * open() rc; the captured text lands in err_buf. */
static int open_capture_stderr(const char * gguf, const char * err_path,
                               char * err_buf, size_t err_cap) {
    err_buf[0] = '\0';
    fflush(stderr);
    const int saved = dup(2);
    const int errfd = open(err_path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (saved < 0 || errfd < 0) {
        perror("stderr capture setup");
        exit(1);
    }
    dup2(errfd, 2);

    voice_diarizer_handle h = NULL;
    const int rc = voice_diarizer_open(gguf, &h);
    if (h) {
        voice_diarizer_close(h);
    }

    fflush(stderr);
    dup2(saved, 2);
    close(errfd);
    close(saved);

    FILE * f = fopen(err_path, "rb");
    if (f) {
        const size_t n = fread(err_buf, 1, err_cap - 1, f);
        err_buf[n] = '\0';
        fclose(f);
    }
    return rc;
}

int main(void) {
    int failures = 0;
    char tmpl[] = "/tmp/omnivoice_diarizer_metadata_XXXXXX";
    char errtmpl[] = "/tmp/omnivoice_diarizer_stderr_XXXXXX";
    char err_buf[4096];
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    close(fd);
    fd = mkstemp(errtmpl);
    if (fd < 0) {
        perror("mkstemp");
        unlink(tmpl);
        return 1;
    }
    close(fd);

    /* 1. Explicit epoch-2 IFGO metadata parses correctly. */
    if (write_diarizer_meta(tmpl, DIAR_CONVERTER_EPOCH, "IFGO") != 0) {
        fprintf(stderr, "cannot write IFGO metadata GGUF\n");
        unlink(tmpl);
        unlink(errtmpl);
        return 1;
    }

    voice_gguf_metadata_t meta;
    int rc = voice_gguf_load_metadata(tmpl, "voice_diarizer", &meta);
    if (rc != 0) {
        fprintf(stderr, "IFGO metadata load returned %d\n", rc);
        ++failures;
    }
    if (meta.converter_epoch != DIAR_CONVERTER_EPOCH || strcmp(meta.lstm_gate_order, "IFGO") != 0) {
        fprintf(stderr, "IFGO metadata parsed incorrectly: epoch=%d gate=%s\n",
                meta.converter_epoch, meta.lstm_gate_order);
        ++failures;
    }
    if (meta.window_samples != DIAR_WINDOW_SAMPLES ||
        meta.frames_per_window != DIAR_FRAMES_PER_WINDOW ||
        meta.lstm_layers != DIAR_LSTM_LAYERS ||
        meta.lstm_hidden != DIAR_LSTM_HIDDEN ||
        meta.linear0_out != DIAR_LINEAR0_OUT ||
        meta.linear1_out != DIAR_LINEAR1_OUT) {
        fprintf(stderr, "diarizer shape metadata mismatch\n");
        ++failures;
    }

    /* 2. Epoch-2 IFGO clears the guard: the open failure must be the
     * missing-tensor stage, not a metadata rejection. */
    rc = open_capture_stderr(tmpl, errtmpl, err_buf, sizeof(err_buf));
    if (rc == 0) {
        fprintf(stderr, "metadata-only IFGO GGUF unexpectedly opened\n");
        ++failures;
    }
    if (!strstr(err_buf, "missing tensor") ||
        strstr(err_buf, "stale GGUF converter epoch") ||
        strstr(err_buf, "unsupported LSTM gate order")) {
        fprintf(stderr, "epoch-2 IFGO artifact did not clear the guard: %s\n", err_buf);
        ++failures;
    }

    /* 3. Explicit non-IFGO gate order is rejected loudly before tensor load. */
    if (write_diarizer_meta(tmpl, DIAR_CONVERTER_EPOCH, "IOFC") != 0) {
        fprintf(stderr, "cannot write IOFC metadata GGUF\n");
        unlink(tmpl);
        unlink(errtmpl);
        return 1;
    }
    rc = open_capture_stderr(tmpl, errtmpl, err_buf, sizeof(err_buf));
    if (rc != -EINVAL || !strstr(err_buf, "unsupported LSTM gate order 'IOFC'")) {
        fprintf(stderr, "IOFC artifact was not rejected by the gate-order guard: rc=%d err=%s\n",
                rc, err_buf);
        ++failures;
    }

    /* 4. Epoch-less legacy artifact (the previously published IOFC bake's
     * metadata shape) is rejected loudly, never silently scrambled. */
    if (write_diarizer_meta(tmpl, -1, NULL) != 0) {
        fprintf(stderr, "cannot write legacy metadata GGUF\n");
        unlink(tmpl);
        unlink(errtmpl);
        return 1;
    }
    rc = open_capture_stderr(tmpl, errtmpl, err_buf, sizeof(err_buf));
    if (rc != -EINVAL || !strstr(err_buf, "stale GGUF converter epoch 0")) {
        fprintf(stderr, "epoch-less legacy artifact was not rejected: rc=%d err=%s\n",
                rc, err_buf);
        ++failures;
    }

    /* 5. Explicit stale epoch (1) is rejected even with IFGO gates. */
    if (write_diarizer_meta(tmpl, 1, "IFGO") != 0) {
        fprintf(stderr, "cannot write epoch-1 metadata GGUF\n");
        unlink(tmpl);
        unlink(errtmpl);
        return 1;
    }
    rc = open_capture_stderr(tmpl, errtmpl, err_buf, sizeof(err_buf));
    if (rc != -EINVAL || !strstr(err_buf, "stale GGUF converter epoch 1")) {
        fprintf(stderr, "epoch-1 artifact was not rejected: rc=%d err=%s\n",
                rc, err_buf);
        ++failures;
    }

    unlink(tmpl);
    unlink(errtmpl);
    printf("omnivoice diarizer metadata failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
