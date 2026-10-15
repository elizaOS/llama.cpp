/*
 * tools/parakeet: native Parakeet-TDT (FastConformer encoder + TDT decoder)
 * speech recognition for FermionResearch/Phonon-2, the five-value QAT
 * re-quantisation of nvidia/parakeet-tdt-0.6b-v3 (weights CC-BY-4.0).
 *
 * Pure C ABI so the fused libelizainference FFI, the CLI tools and tests can
 * share it. Everything runs on the CPU: the 264 encoder linears use the
 * five-value int8 kernels in pk-kernels*.cpp (NEON i8mm / dotprod, x86_64
 * AVX2, scalar reference), the rest is fp32. There is no ONNX, MLX or Core ML path.
 *
 * Threading: a model owns a worker pool sized at load (pk_params.n_threads).
 * Calls on one model are serialised by the caller (the FFI holds its ASR
 * mutex); streams on one model must not decode concurrently.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pk_model pk_model;
typedef struct pk_stream pk_stream;

enum pk_status {
    PK_OK = 0,
    PK_E_INVALID = -1,
    PK_E_IO = -2,
    PK_E_FORMAT = -3,
    PK_E_CANCELLED = -4,
    PK_E_NOMEM = -5,
};

typedef struct pk_params {
    int n_threads;      /* <= 0: one per big core (Android) / performance core (Apple), capped at 4 */
    int kernel;         /* 0 = best available, 1 = scalar reference, 2 = dotprod, 3 = i8mm, 4 = avx2 */
    int pin_threads;    /* 1 = pin workers to the fastest cores where the OS allows it (Linux/Android) */
    int use_mmap;       /* 1 = mmap the GGUF (clean, reclaimable pages); 0 = read it into private memory */
} pk_params;

pk_params pk_default_params(void);

/* Load a parakeet-tdt GGUF (convert_phonon2_to_gguf.py). The file is mmapped;
 * the five-value weights are used in place. Returns NULL and writes a
 * message into err (if err_len > 0) on failure. */
pk_model * pk_model_load(const char * path, pk_params params, char * err, size_t err_len);
void pk_model_free(pk_model * model);

/* 1 when the GGUF at `path` declares general.architecture = "parakeet-tdt". */
int pk_gguf_is_parakeet(const char * path);

/* Kernel tier actually in use: "i8mm", "dotprod", "avx2" or "scalar". */
const char * pk_model_kernel_name(const pk_model * model);
int pk_model_n_threads(const pk_model * model);
int pk_model_sample_rate(const pk_model * model);   /* 16000 */
/* The NOTICE / attribution text carried in the GGUF (CC-BY-4.0). */
const char * pk_model_notice(const pk_model * model);

/* Cancellation hook polled between encoder layers and decode steps. */
typedef int (*pk_cancel_cb)(void * user);

typedef struct pk_token {
    int32_t id;         /* SentencePiece id (0..8191) */
    int32_t frame;      /* encoder frame (80 ms) where the token was emitted */
    int32_t duration;   /* predicted TDT duration in frames */
} pk_token;

typedef struct pk_result {
    char * text;        /* UTF-8, malloc'd */
    pk_token * tokens;  /* malloc'd */
    size_t n_tokens;
    int n_frames;       /* encoder frames of the input */
    /* stage timings (ms, wall) of the call */
    double t_mel_ms, t_encoder_ms, t_decoder_ms;
} pk_result;

/* Transcribe 16 kHz mono fp32 PCM (callers resample). Fills *out (free with
 * pk_result_free). Long inputs are cut into windows of at most
 * PK_MAX_WINDOW_S seconds at low-energy points and the transcripts joined. */
int pk_transcribe(pk_model * model, const float * pcm16k, size_t n_samples,
                  pk_cancel_cb cancel, void * cancel_user, pk_result * out);
void pk_result_free(pk_result * res);

/* Words with times: splits the transcript at SentencePiece word starts. The
 * caller's arrays have capacity *io_n; *io_n receives the count written. */
int pk_result_words(const pk_model * model, const pk_result * res,
                    int * word_start_ms, int * word_end_ms, size_t * io_n);

/* ---- streaming (pause-delimited windowed re-decode, local agreement) -- */

typedef struct pk_stream_params {
    int right_context_ms;   /* tokens newer than this are never committed (default 800) */
    int min_hop_ms;         /* minimum new audio between decodes (default 400); the
                               interval grows to a sixth of the window */
    int max_window_ms;      /* longest decode window; past it stable tokens are committed
                               without agreement and the segment is cut (default 12000) */
} pk_stream_params;

pk_stream_params pk_stream_default_params(void);

pk_stream * pk_stream_open(pk_model * model, pk_stream_params params);
/* Append 16 kHz mono fp32 PCM. Never decodes. */
int pk_stream_feed(pk_stream * st, const float * pcm16k, size_t n_samples);
/* Decode if at least min_hop_ms of new audio arrived since the last decode,
 * then write committed + tentative text. Returns bytes written (excluding
 * NUL) or a negative pk_status. An insufficient output buffer returns
 * PK_E_INVALID without partial text; retry with more space. */
int pk_stream_partial(pk_stream * st, char * out, size_t cap, pk_cancel_cb cancel, void * cancel_user);
/* Decode the remaining audio with full right context and write the final
 * transcript. */
int pk_stream_finish(pk_stream * st, char * out, size_t cap, pk_cancel_cb cancel, void * cancel_user);
/* Stats for benchmarks: number of decodes and total decode wall time. */
void pk_stream_stats(const pk_stream * st, int * n_decodes, double * decode_ms, double * first_partial_ms);
void pk_stream_close(pk_stream * st);

#ifdef __cplusplus
}
#endif
