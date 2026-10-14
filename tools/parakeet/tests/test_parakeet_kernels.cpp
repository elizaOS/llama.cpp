// tools/parakeet/tests/test_parakeet_kernels.cpp - every kernel tier must give
// bit-identical int32 results to the scalar reference (decode, tile, matvec,
// and the whole gemm_q8 driver), and gemm_q8 must match a double-precision
// GEMM over the same int8 operands.
#include "pk-kernels.h"
#include "pk-ops.h"
#include "pk-pool.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace pk;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++fails; } } while (0)

static void make_fv5(std::mt19937 & rng, int O, int K, std::vector<uint8_t> & codes, std::vector<float> & lohi) {
    std::uniform_int_distribution<int> lv(0, 4);
    std::uniform_real_distribution<float> mag(0.001f, 0.05f), ratio(1.6f, 2.6f);
    codes.assign((size_t) O / 2 * K, 0);
    // build levels row-major, then pack in fv5-pair16-v1 order
    std::vector<uint8_t> L((size_t) O * K);
    for (auto & v : L) v = (uint8_t) lv(rng);
    for (int p = 0; p < O / 2; ++p) {
        for (int g = 0; g < K / 16; ++g) {
            for (int j = 0; j < 16; ++j) {
                const int r = 2 * p + (j >= 8);
                const int k = 16 * g + (j & 7);
                const uint8_t a = L[(size_t) r * K + k] + (j >= 8 ? 8 : 0);
                const uint8_t b = L[(size_t) r * K + k + 8] + (j >= 8 ? 8 : 0);
                codes[(size_t) p * K + 16 * g + j] = (uint8_t) (a | (b << 4));
            }
        }
    }
    lohi.resize((size_t) O * 2);
    for (int r = 0; r < O; ++r) {
        const float lo = mag(rng);
        lohi[2 * r] = lo;
        lohi[2 * r + 1] = lo * ratio(rng);
    }
}

int main() {
    std::mt19937 rng(1234);
    const kernel_table & ref = kernels_scalar();
    std::vector<const kernel_table *> tiers;
#if PK_HAVE_ARM_EXT
    if (cpu_has_dotprod()) tiers.push_back(&kernels_dotprod());
    if (cpu_has_i8mm()) tiers.push_back(&kernels_i8mm());
#endif
    std::printf("tiers under test: scalar");
    for (auto * t : tiers) std::printf(", %s", t->name);
    std::printf("\n");

    for (int K : { 256, 1024, 4096 }) {
        const int O = 64, N = 21;
        std::vector<uint8_t> codes;
        std::vector<float> lohi;
        make_fv5(rng, O, K, codes, lohi);
        wmat W;
        wmat_from_fv5(W, codes.data(), lohi.data(), O, K);
        CHECK(W.grid_err < 0.01, "grid error %g", W.grid_err);
        std::vector<float> X((size_t) N * K);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & v : X) v = nd(rng);
        pool P1(1, false), P3(3, false);
        qact qa;
        quantize_acts(P1, X.data(), N, K, K, qa);
        // panel + tile identity
        std::vector<int8_t> pr((size_t) 8 * K), pt((size_t) 8 * K);
        int32_t a[64], b[64];
        ref.decode_fv5(codes.data(), K, W.lut.data(), K, pr.data());
        ref.tile(pr.data(), qa.buf.data(), qa.pair_stride(), K, 4, a);
        for (auto * t : tiers) {
            t->decode_fv5(codes.data(), K, W.lut.data(), K, pt.data());
            CHECK(std::memcmp(pr.data(), pt.data(), pr.size()) == 0, "%s decode differs (K=%d)", t->name, K);
            t->tile(pr.data(), qa.buf.data(), qa.pair_stride(), K, 4, b);
            CHECK(std::memcmp(a, b, sizeof(a)) == 0, "%s tile differs (K=%d)", t->name, K);
            for (int np = 1; np <= 3; ++np) {
                int32_t c[64];
                t->tile(pr.data(), qa.buf.data(), qa.pair_stride(), K, np, c);
                for (int tk = 0; tk < 2 * np; ++tk) {
                    CHECK(std::memcmp(a + tk * 8, c + tk * 8, 32) == 0, "%s tile np=%d differs (K=%d)", t->name, np, K);
                }
            }
        }
        // tile vs direct dot products over the decoded levels
        for (int r = 0; r < 8; ++r) {
            for (int tk = 0; tk < 8; ++tk) {
                int64_t s = 0;
                for (int k = 0; k < K; ++k) {
                    const int c = k / 8, j = k % 8;
                    const int8_t w = pr[((size_t) c * 4 + r / 2) * 16 + (r & 1) * 8 + j];
                    const int8_t x = qa.buf[(size_t) (tk / 2) * qa.pair_stride() + c * 16 + (tk & 1) * 8 + j];
                    s += (int64_t) w * x;
                }
                CHECK(s == a[tk * 8 + r], "scalar tile wrong at r=%d t=%d", r, tk);
            }
        }
        // full driver: tiers and thread counts agree bit for bit
        std::vector<float> y0((size_t) N * O), y1((size_t) N * O);
        gemm_q8(P1, ref, W, qa, y0.data(), O, epi::store);
        for (auto * t : tiers) {
            gemm_q8(P3, *t, W, qa, y1.data(), O, epi::store);
            CHECK(std::memcmp(y0.data(), y1.data(), y0.size() * 4) == 0, "%s gemm differs (K=%d)", t->name, K);
        }
        // and the driver result is the dequantised product
        double maxrel = 0;
        for (int t = 0; t < N; ++t) {
            for (int o = 0; o < O; ++o) {
                double s = 0, sw = 0;
                for (int k = 0; k < K; ++k) {
                    const int c = k / 8, j = k % 8;
                    const int8_t x = qa.buf[(size_t) (t / 2) * qa.pair_stride() + c * 16 + (t & 1) * 8 + j];
                    // decode the weight level from the codes
                    const int p = o / 2, g = k / 16, jj = (k % 16) % 8 + (o & 1) * 8;
                    const uint8_t byte = codes[(size_t) p * K + 16 * g + jj];
                    const int lv = ((k % 16) < 8 ? (byte & 15) : (byte >> 4)) - (o & 1) * 8;
                    const double mags[5] = { -lohi[2 * o + 1], -lohi[2 * o], 0.0, lohi[2 * o], lohi[2 * o + 1] };
                    s += mags[lv] * x * qa.scale[t];
                    sw += std::fabs(mags[lv] * x * qa.scale[t]);
                }
                maxrel = std::max(maxrel, std::fabs(s - y0[(size_t) t * O + o]) / (sw + 1e-12));
            }
        }
        CHECK(maxrel < 2e-3, "gemm vs exact five-value product: rel err %g (K=%d)", maxrel, K);
        std::printf("K=%d ok: grid_err %.2e, gemm rel err vs exact five-value %.2e\n", K, W.grid_err, maxrel);
        // matvec
        std::vector<int8_t> mw((size_t) 37 * K), mx(K);
        std::uniform_int_distribution<int> i8(-127, 127);
        for (auto & v : mw) v = (int8_t) i8(rng);
        for (auto & v : mx) v = (int8_t) i8(rng);
        std::vector<int32_t> m0(37), m1(37);
        ref.matvec(mw.data(), 37, K, mx.data(), m0.data());
        for (auto * t : tiers) {
            t->matvec(mw.data(), 37, K, mx.data(), m1.data());
            CHECK(m0 == m1, "%s matvec differs", t->name);
        }
    }
    std::printf(fails ? "FAILED (%d)\n" : "all kernel checks passed\n", fails);
    return fails ? 1 : 0;
}
