// Portable-fast (non-Apple) instantiation of kokoro-layers.h for the parity
// test (test_kokoro_layers_portable.cpp).
//
// KOKORO_NO_ACCELERATE forces KOKORO_USE_PORTABLE_FAST even on Apple hosts,
// so the threaded + NEON path that Android/Windows/Linux take in production
// is compiled and executed natively here (on arm64 Macs the NEON kernels are
// the exact code aarch64 Android runs). See the ref TU for the namespace
// rename rationale.

#define KOKORO_NO_ACCELERATE 1
#define eliza_kokoro eliza_kokoro_portable_fast
#include "kokoro-layers.h"
#undef eliza_kokoro

#if !defined(KOKORO_USE_PORTABLE_FAST)
#error "expected KOKORO_NO_ACCELERATE to select the portable-fast path"
#endif

namespace kokoro_layer_test {

void fast_linear(const float * x, int in_dim, const float * W, const float * b,
                 int out_dim, float * y) {
    eliza_kokoro_portable_fast::linear_forward(x, in_dim, W, b, out_dim, y);
}

void fast_conv1d(const float * x, int Cin, int T, const float * W, const float * b,
                 int Cout, int K, int stride, int pad, int dilation,
                 float * y, int T_out) {
    eliza_kokoro_portable_fast::conv1d_forward(x, Cin, T, W, b, Cout, K,
                                               stride, pad, dilation, y, T_out);
}

void fast_convtranspose1d(const float * x, int Cin, int T, const float * W,
                          const float * b, int Cout, int K, int stride, int pad,
                          int output_pad, float * y, int T_out) {
    eliza_kokoro_portable_fast::convtranspose1d_forward(x, Cin, T, W, b, Cout, K,
                                                        stride, pad, output_pad, y, T_out);
}

void fast_lstm_cell_step(const float * x, int I, int H,
                         const float * h_prev, const float * c_prev,
                         const float * W_ih, const float * b_ih,
                         const float * W_hh, const float * b_hh,
                         float * h_out, float * c_out, float * gates_scratch) {
    eliza_kokoro_portable_fast::lstm_cell_step(x, I, H, h_prev, c_prev,
                                               W_ih, b_ih, W_hh, b_hh,
                                               h_out, c_out, gates_scratch);
}

bool fast_is_neon() {
#if defined(KOKORO_USE_NEON)
    return true;
#else
    return false;
#endif
}

int fast_thread_count() {
    return eliza_kokoro_portable_fast::detail::kokoro_thread_count();
}

} // namespace kokoro_layer_test
