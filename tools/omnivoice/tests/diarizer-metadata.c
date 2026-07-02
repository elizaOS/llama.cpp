/*
 * Diarizer GGUF metadata contract for the vendored fused reader.
 *
 * The current pinned fused library reads pyannote LSTM gates in raw ONNX IOFC
 * order. Future package-side converters emit explicit epoch/gate metadata, so
 * this test pins the fork loader's ability to see that metadata before tensor
 * load and reject mismatched artifacts loudly.
 */

#define _DEFAULT_SOURCE
#define _XOPEN_SOURCE 700

#include "voice_classifier/voice_classifier.h"
#include "voice_gguf_loader.h"

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

static int write_diarizer_meta(const char * path, uint32_t converter_epoch, const char * gate_order) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return -1;
    }

    fwrite(VC_GGUF_MAGIC, 1, 4, f);
    w_u32(f, VC_GGUF_VERSION);
    w_i64(f, 0);
    w_i64(f, 11);

    w_kv_u32(f, "voice_diarizer.sample_rate", VOICE_CLASSIFIER_SAMPLE_RATE_HZ);
    w_kv_u32(f, "voice_diarizer.num_classes", VOICE_DIARIZER_NUM_CLASSES);
    w_kv_u32(f, "voice_diarizer.window_samples", DIAR_WINDOW_SAMPLES);
    w_kv_u32(f, "voice_diarizer.frames_per_window", DIAR_FRAMES_PER_WINDOW);
    w_kv_u32(f, "voice_diarizer.converter_epoch", converter_epoch);
    w_kv_u32(f, "voice_diarizer.lstm_layers", DIAR_LSTM_LAYERS);
    w_kv_u32(f, "voice_diarizer.lstm_hidden", DIAR_LSTM_HIDDEN);
    w_kv_u32(f, "voice_diarizer.linear0_out", DIAR_LINEAR0_OUT);
    w_kv_u32(f, "voice_diarizer.linear1_out", DIAR_LINEAR1_OUT);
    w_kv_str(f, "voice_diarizer.variant", "pyannote-segmentation-3.0");
    w_kv_str(f, "voice_diarizer.lstm_gate_order", gate_order);

    fclose(f);
    return 0;
}

int main(void) {
    int failures = 0;
    char tmpl[] = "/tmp/omnivoice_diarizer_metadata_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    close(fd);

    if (write_diarizer_meta(tmpl, DIAR_CONVERTER_EPOCH, "IOFC") != 0) {
        fprintf(stderr, "cannot write IOFC metadata GGUF\n");
        unlink(tmpl);
        return 1;
    }

    voice_gguf_metadata_t meta;
    int rc = voice_gguf_load_metadata(tmpl, "voice_diarizer", &meta);
    if (rc != 0) {
        fprintf(stderr, "IOFC metadata load returned %d\n", rc);
        ++failures;
    }
    if (meta.converter_epoch != DIAR_CONVERTER_EPOCH || strcmp(meta.lstm_gate_order, "IOFC") != 0) {
        fprintf(stderr, "IOFC metadata parsed incorrectly: epoch=%d gate=%s\n",
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

    if (write_diarizer_meta(tmpl, DIAR_CONVERTER_EPOCH, "IFGO") != 0) {
        fprintf(stderr, "cannot write IFGO metadata GGUF\n");
        unlink(tmpl);
        return 1;
    }
    memset(&meta, 0, sizeof(meta));
    rc = voice_gguf_load_metadata(tmpl, "voice_diarizer", &meta);
    if (rc != 0 || strcmp(meta.lstm_gate_order, "IFGO") != 0) {
        fprintf(stderr, "IFGO metadata was not preserved for fail-fast validation\n");
        ++failures;
    }

    unlink(tmpl);
    printf("omnivoice diarizer metadata failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
