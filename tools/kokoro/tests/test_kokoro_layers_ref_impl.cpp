// Scalar-reference instantiation of kokoro-layers.h for the portable-path
// parity test (test_kokoro_layers_portable.cpp).
//
// KOKORO_FORCE_SCALAR compiles only the readable single-threaded loops.
// The namespace is renamed via macro so this TU and the portable-fast TU
// (different inline function bodies for the same declarations) can link into
// one test binary without an ODR violation. Test-only trick — production
// code always compiles the header exactly once per configuration.

#define KOKORO_FORCE_SCALAR 1
#define eliza_kokoro eliza_kokoro_scalar_ref
#include "kokoro-layers.h"
#undef eliza_kokoro

namespace kokoro_layer_test {

void ref_linear(const float * x, int in_dim, const float * W, const float * b,
                int out_dim, float * y) {
    eliza_kokoro_scalar_ref::linear_forward(x, in_dim, W, b, out_dim, y);
}

void ref_conv1d(const float * x, int Cin, int T, const float * W, const float * b,
                int Cout, int K, int stride, int pad, int dilation,
                float * y, int T_out) {
    eliza_kokoro_scalar_ref::conv1d_forward(x, Cin, T, W, b, Cout, K,
                                            stride, pad, dilation, y, T_out);
}

void ref_convtranspose1d(const float * x, int Cin, int T, const float * W,
                         const float * b, int Cout, int K, int stride, int pad,
                         int output_pad, float * y, int T_out) {
    eliza_kokoro_scalar_ref::convtranspose1d_forward(x, Cin, T, W, b, Cout, K,
                                                     stride, pad, output_pad, y, T_out);
}

void ref_lstm_cell_step(const float * x, int I, int H,
                        const float * h_prev, const float * c_prev,
                        const float * W_ih, const float * b_ih,
                        const float * W_hh, const float * b_hh,
                        float * h_out, float * c_out, float * gates_scratch) {
    eliza_kokoro_scalar_ref::lstm_cell_step(x, I, H, h_prev, c_prev,
                                            W_ih, b_ih, W_hh, b_hh,
                                            h_out, c_out, gates_scratch);
}

} // namespace kokoro_layer_test
