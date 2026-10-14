// tools/parakeet/src/pk-kernels.cpp - scalar reference kernels and the
// runtime CPU-feature dispatch.
#include "pk-kernels.h"

#include <cstdlib>
#include <cstring>

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if PK_HAVE_X86_EXT
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace pk {

static void decode_fv5_scalar(const uint8_t * codes, size_t pair_stride, const int8_t * luts, int K, int8_t * panel) {
    const int G = K / 16;
    for (int p = 0; p < 4; ++p) {
        const uint8_t * src = codes + (size_t) p * pair_stride;
        const int8_t * lut = luts + p * 16;
        for (int g = 0; g < G; ++g) {
            int8_t * a = panel + ((size_t) (2 * g) * 4 + p) * 16;
            int8_t * b = panel + ((size_t) (2 * g + 1) * 4 + p) * 16;
            for (int j = 0; j < 16; ++j) {
                const uint8_t v = src[g * 16 + j];
                a[j] = lut[v & 0x0F];
                b[j] = lut[v >> 4];
            }
        }
    }
}

static void tile_scalar(const int8_t * panel, const int8_t * x, size_t x_stride, int K, int np, int32_t * out) {
    int32_t acc[8][8];
    std::memset(acc, 0, sizeof(acc));
    const int C = K / 8;
    for (int c = 0; c < C; ++c) {
        const int8_t * w = panel + (size_t) c * 64;
        for (int q = 0; q < np; ++q) {
            const int8_t * xv = x + (size_t) q * x_stride + (size_t) c * 16;
            for (int p = 0; p < 4; ++p) {
                const int8_t * wv = w + p * 16;
                for (int rr = 0; rr < 2; ++rr) {
                    for (int tt = 0; tt < 2; ++tt) {
                        int32_t s = 0;
                        for (int k = 0; k < 8; ++k) {
                            s += (int32_t) wv[rr * 8 + k] * (int32_t) xv[tt * 8 + k];
                        }
                        acc[2 * p + rr][2 * q + tt] += s;
                    }
                }
            }
        }
    }
    for (int r = 0; r < 8; ++r) {
        for (int t = 0; t < 2 * np; ++t) {
            out[t * 8 + r] = acc[r][t];
        }
    }
}

static void matvec_scalar(const int8_t * w, int O, int K, const int8_t * x, int32_t * y) {
    for (int o = 0; o < O; ++o) {
        const int8_t * row = w + (size_t) o * K;
        int32_t s = 0;
        for (int k = 0; k < K; ++k) {
            s += (int32_t) row[k] * (int32_t) x[k];
        }
        y[o] = s;
    }
}

const kernel_table & kernels_scalar() {
    static const kernel_table t = { tier::scalar, "scalar", decode_fv5_scalar, tile_scalar, matvec_scalar };
    return t;
}

bool cpu_has_dotprod() {
#if defined(__aarch64__) && defined(__APPLE__)
    int v = 0;
    size_t n = sizeof(v);
    return sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &n, nullptr, 0) == 0 && v != 0;
#elif defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

bool cpu_has_i8mm() {
#if defined(__aarch64__) && defined(__APPLE__)
    int v = 0;
    size_t n = sizeof(v);
    return sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &n, nullptr, 0) == 0 && v != 0;
#elif defined(__aarch64__) && defined(__linux__)
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    return false;
#endif
}

bool cpu_has_avx2() {
#if PK_HAVE_X86_EXT
    unsigned a = 0, b = 0, c = 0, d = 0;
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    c = (unsigned) r[2];
#else
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid(1, a, b, c, d);
#endif
    // OSXSAVE (ecx bit 27) and AVX (bit 28), then XCR0 must enable the XMM and
    // YMM state (bits 1 and 2) or the OS does not preserve the upper halves.
    if (!(c & (1u << 27)) || !(c & (1u << 28))) return false;
#if defined(_MSC_VER)
    const unsigned long long xcr0 = _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
    if ((xcr0 & 0x6) != 0x6) return false;
#if defined(_MSC_VER)
    __cpuidex(r, 7, 0);
    b = (unsigned) r[1];
#else
    __cpuid_count(7, 0, a, b, c, d);
#endif
    (void) a; (void) d;
    return (b & (1u << 5)) != 0;  // AVX2
#else
    return false;
#endif
}

const kernel_table & kernels_for(int want) {
    const char * env = std::getenv("PK_KERNEL");
    if (env && *env) {
        if (!std::strcmp(env, "scalar")) want = 1;
        else if (!std::strcmp(env, "dotprod")) want = 2;
        else if (!std::strcmp(env, "i8mm")) want = 3;
        else if (!std::strcmp(env, "avx2")) want = 4;
    }
#if PK_HAVE_ARM_EXT
    const bool dp = cpu_has_dotprod();
    const bool mm = dp && cpu_has_i8mm();
    if (want == 1) return kernels_scalar();
    if (want == 2 && dp) return kernels_dotprod();
    if (want == 3 && mm) return kernels_i8mm();
    if (mm) return kernels_i8mm();
    if (dp) return kernels_dotprod();
#elif PK_HAVE_X86_EXT
    if (want == 1) return kernels_scalar();
    if (cpu_has_avx2()) return kernels_avx2();
#else
    (void) want;
#endif
    return kernels_scalar();
}

}  // namespace pk
