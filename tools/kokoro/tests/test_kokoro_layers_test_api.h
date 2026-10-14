/** Shared scalar and portable-fast test entrypoints keep callers and implementation translation units type-checked together. */
#pragma once

namespace kokoro_layer_test {
void ref_linear(const float *, int, const float *, const float *, int, float *);
void ref_conv1d(const float *, int, int, const float *, const float *, int, int, int, int, int, float *, int);
void ref_convtranspose1d(const float *, int, int, const float *, const float *, int, int, int, int, int, float *, int);
void ref_lstm_cell_step(const float *, int, int, const float *, const float *, const float *, const float *, const float *, const float *, float *, float *, float *);
void fast_linear(const float *, int, const float *, const float *, int, float *);
void fast_conv1d(const float *, int, int, const float *, const float *, int, int, int, int, int, float *, int);
void fast_convtranspose1d(const float *, int, int, const float *, const float *, int, int, int, int, int, float *, int);
void fast_lstm_cell_step(const float *, int, int, const float *, const float *, const float *, const float *, const float *, const float *, float *, float *, float *);
bool fast_is_neon();
int  fast_thread_count();
} // namespace kokoro_layer_test
