// test-cuda-extra-kernels.cpp
//
// Compile-only validation that the W4-B CUDA kernels for QJL, PolarQuant
// Q4, and TBQ3_TCQ are linkable. This test does not run the kernels —
// the host running the test suite has no NVIDIA driver bound. It only
// verifies that the symbols are present and reachable from a regular
// llama.cpp test target.
//
// Hardware-runtime equivalents (per the W3-D hardware-runner checklist)
// will live in test-backend-ops.cpp under the corresponding op
// dispatchers once a real-GPU runner is available.

#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------
// 1. The block layouts are visible from the C type-traits perspective.
// ---------------------------------------------------------------------

static_assert((int) GGML_TYPE_QJL1_256 == 46, "QJL1_256 GGML type slot");
static_assert((int) GGML_TYPE_Q4_POLAR == 47, "Q4_POLAR GGML type slot");
static_assert((int) GGML_TYPE_TBQ3_TCQ == 48, "TBQ3_TCQ GGML type slot (W4-B addition)");

// ---------------------------------------------------------------------
// 2. The CUDA host wrappers exposed by qjl.cuh / polarquant.cuh /
//    turbo-tcq.cuh are guarded by their feature macros. We declare
//    matching extern "C" prototypes here so the linker resolves them at
//    library-link time. If any of them are missing from libggml-cuda,
//    the link step for this test target would fail.
//
//    We intentionally take a "weak" reference: we declare the prototypes
//    + emit a function pointer table referencing each, but never invoke
//    them. nvcc cannot guarantee the kernels actually launch; that is
//    the next agent's job once GPU hardware is available.
// ---------------------------------------------------------------------

// Declare the unmangled C linkage entry points. The signatures are
// intentionally minimal (no cuda_runtime.h pulled in) — the linker only
// matches by name for extern "C" symbols, and the *.cuh definitions
// guarantee these names exist whenever the feature macro is defined.
#if defined(GGML_USE_CUDA) && defined(GGML_CUDA_QJL)
extern "C" void quantize_row_qjl1_256_cuda();
extern "C" void dequantize_row_qjl1_256_cuda();
extern "C" void attn_score_qjl_cuda();
#endif

#if defined(GGML_USE_CUDA) && defined(GGML_CUDA_POLARQUANT)
extern "C" void dequantize_row_q4_polar_cuda();
extern "C" void mul_mat_q4_polar_q8_0_cuda();
#endif

#if defined(GGML_USE_CUDA) && defined(GGML_CUDA_TBQ3_TCQ)
extern "C" void dequantize_row_tbq3_tcq_cuda();
extern "C" void mul_mat_tbq3_tcq_q8_0_cuda();
#endif

int main() {
    int seen = 0;

#if defined(GGML_USE_CUDA) && defined(GGML_CUDA_QJL)
    void * qjl_table[] = {
        (void *) &quantize_row_qjl1_256_cuda,
        (void *) &dequantize_row_qjl1_256_cuda,
        (void *) &attn_score_qjl_cuda,
    };
    for (size_t i = 0; i < sizeof(qjl_table) / sizeof(qjl_table[0]); ++i) {
        if (qjl_table[i] == nullptr) {
            fprintf(stderr, "FAIL: QJL CUDA symbol %zu is null\n", i);
            return 1;
        }
        ++seen;
    }
    fprintf(stderr, "[ok] QJL CUDA: 3 host wrappers linkable\n");
#else
    fprintf(stderr, "[skip] QJL CUDA path not built (GGML_CUDA_QJL undefined)\n");
#endif

#if defined(GGML_USE_CUDA) && defined(GGML_CUDA_POLARQUANT)
    void * polar_table[] = {
        (void *) &dequantize_row_q4_polar_cuda,
        (void *) &mul_mat_q4_polar_q8_0_cuda,
    };
    for (size_t i = 0; i < sizeof(polar_table) / sizeof(polar_table[0]); ++i) {
        if (polar_table[i] == nullptr) {
            fprintf(stderr, "FAIL: Polar CUDA symbol %zu is null\n", i);
            return 1;
        }
        ++seen;
    }
    fprintf(stderr, "[ok] Polar CUDA: 2 host wrappers linkable\n");
#else
    fprintf(stderr, "[skip] Polar CUDA path not built (GGML_CUDA_POLARQUANT undefined)\n");
#endif

#if defined(GGML_USE_CUDA) && defined(GGML_CUDA_TBQ3_TCQ)
    void * tcq_table[] = {
        (void *) &dequantize_row_tbq3_tcq_cuda,
        (void *) &mul_mat_tbq3_tcq_q8_0_cuda,
    };
    for (size_t i = 0; i < sizeof(tcq_table) / sizeof(tcq_table[0]); ++i) {
        if (tcq_table[i] == nullptr) {
            fprintf(stderr, "FAIL: TBQ3_TCQ CUDA symbol %zu is null\n", i);
            return 1;
        }
        ++seen;
    }
    fprintf(stderr, "[ok] TBQ3_TCQ CUDA: 2 host wrappers linkable\n");
#else
    fprintf(stderr, "[skip] TBQ3_TCQ CUDA path not built (GGML_CUDA_TBQ3_TCQ undefined)\n");
#endif

    fprintf(stderr, "test-cuda-extra-kernels: %d host wrappers verified\n", seen);
    return 0;
}
