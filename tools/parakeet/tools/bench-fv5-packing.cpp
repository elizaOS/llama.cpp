// tools/parakeet/tools/bench-fv5-packing.cpp - cost of the `fv5-b5x3t7-v1`
// code layout (2.35-2.41 bits per weight) against `fv5-pair16-v1` (4 bits).
//
// Both layouts decode into the same int8 weight panel (pk-kernels.h), so only
// the decode step differs. The bench packs random levels with the real
// Phonon-2 level distribution in both layouts, checks that the selected tier
// decodes them to the same panel, then times decode alone and decode + the
// tier's int8 tile over N tokens, on one thread, for the encoder's linear
// shapes. It is what the gemm_q8 driver does per 8-row block.
//
// usage: bench-fv5-packing [reps]   (PK_KERNEL selects the tier)
#include "pk-kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

// level of lane j in panel vector v for an 8-row block of `lv` ([8][K], 0..4)
inline uint8_t panel_level(const uint8_t * lv, int K, int v, int j) {
    const int c = v / 4, p = v % 4;
    const int row = 2 * p + (j >= 8 ? 1 : 0);
    return lv[(size_t) row * K + 8 * c + (j & 7)];
}

// pair16 codes for one block: [4 pairs][K bytes], pair stride K
void pack_pair16(const uint8_t * lv, int K, uint8_t * out) {
    for (int p = 0; p < 4; ++p) {
        for (int g = 0; g < K / 16; ++g) {
            for (int j = 0; j < 16; ++j) {
                const int r = 2 * p + (j >= 8 ? 1 : 0);
                const int k = 16 * g + (j & 7);
                const uint8_t a = (uint8_t) (lv[(size_t) r * K + k] + (j >= 8 ? 8 : 0));
                const uint8_t b = (uint8_t) (lv[(size_t) r * K + k + 8] + (j >= 8 ? 8 : 0));
                out[(size_t) p * K + g * 16 + j] = (uint8_t) (a | (b << 4));
            }
        }
    }
}

int b5x3_vectors(int K) { return pk::fv5t7_codes(K); }
size_t b5x3_bytes(int K) { return (size_t) b5x3_vectors(K) * 16; }
int t7_groups(int K) { return (b5x3_vectors(K) + 7) / 8; }
size_t t7_bytes(int K) { return pk::fv5t7_block_bytes(K); }

void pack_b5x3(const uint8_t * lv, int K, uint8_t * out) {
    const int V = K / 2, T = b5x3_vectors(K);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < 16; ++j) {
            int code = 0, mul = 1;
            for (int d = 0; d < 3; ++d) {
                const int v = 3 * t + d;
                code += (v < V ? panel_level(lv, K, v, j) : 2) * mul;
                mul *= 5;
            }
            out[(size_t) t * 16 + j] = (uint8_t) code;
        }
    }
}

void pack_t7(const uint8_t * b5, int K, uint8_t * out) {
    const int T = b5x3_vectors(K), Gn = t7_groups(K);
    for (int g = 0; g < Gn; ++g) {
        for (int j = 0; j < 16; ++j) {
            uint8_t c[8];
            for (int m = 0; m < 8; ++m) {
                const int t = 8 * g + m;
                c[m] = t < T ? b5[(size_t) t * 16 + j] : 62;  // 62 = levels {2,2,2}
            }
            for (int m = 0; m < 7; ++m) {
                out[((size_t) g * 7 + m) * 16 + j] = (uint8_t) (c[m] | (((c[7] >> m) & 1) << 7));
            }
        }
    }
}

double now_s() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

}  // namespace

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 3;
    const pk::kernel_table & kt = pk::kernels_for(0);
    std::printf("tile tier: %s\n", kt.name);
    for (int x = 0; x < 125; ++x) {
        if (((x + 1) * 51) >> 8 != x / 5) { std::printf("div5 identity fails at %d\n", x); return 1; }
    }
    // encoder linears per conformer block (O x K): ff1 up/down, qkv, out, pos,
    // conv pw1/pw2, ff2 up/down
    struct shape { int O, K, per_block; };
    const shape shapes[] = { { 4096, 1024, 2 }, { 1024, 4096, 2 }, { 3072, 1024, 1 }, { 1024, 1024, 3 }, { 2048, 1024, 1 } };
    const int Ns[] = { 8, 16, 32, 64, 125, 375 };
    // level distribution of the real Phonon-2 encoder (container statistics)
    std::mt19937 rng(1234);
    std::discrete_distribution<int> dist({ 0.0909, 0.1526, 0.5133, 0.1525, 0.0908 });

    std::printf("%-6s %-5s %-6s %10s %10s %10s | %s\n", "O", "K", "layout", "bytes/blk", "dec ns/blk", "dec GW/s",
                "decode+tile ms per GEMM at N =");
    for (const shape & sh : shapes) {
        const int O = sh.O, K = sh.K, nb = O / 8;
        std::vector<uint8_t> lv((size_t) O * K);
        for (auto & v : lv) v = (uint8_t) dist(rng);
        std::vector<int8_t> luts((size_t) nb * 64);
        for (int i = 0; i < nb * 4; ++i) {
            const int8_t H = (int8_t) (64 + rng() % 64), L = (int8_t) (1 + rng() % 40);
            const int8_t row[5] = { (int8_t) -H, (int8_t) -L, 0, L, H };
            for (int r = 0; r < 2; ++r) for (int l = 0; l < 5; ++l) luts[(size_t) i * 16 + r * 8 + l] = row[l];
        }
        // packed codes per layout
        std::vector<uint8_t> p16((size_t) nb * 4 * K), b5((size_t) nb * b5x3_bytes(K)), t7((size_t) nb * t7_bytes(K));
        for (int rb = 0; rb < nb; ++rb) {
            pack_pair16(lv.data() + (size_t) rb * 8 * K, K, p16.data() + (size_t) rb * 4 * K);
            pack_b5x3(lv.data() + (size_t) rb * 8 * K, K, b5.data() + (size_t) rb * b5x3_bytes(K));
            pack_t7(b5.data() + (size_t) rb * b5x3_bytes(K), K, t7.data() + (size_t) rb * t7_bytes(K));
        }
        // correctness: both layouts give the same panel
        std::vector<int8_t> ref((size_t) 8 * K), got((size_t) 8 * K);
        for (int rb = 0; rb < nb; ++rb) {
            kt.decode_fv5(p16.data() + (size_t) rb * 4 * K, (size_t) K, luts.data() + (size_t) rb * 64, K, ref.data());
            kt.decode_fv5t7(t7.data() + (size_t) rb * t7_bytes(K), luts.data() + (size_t) rb * 64, K, got.data());
            if (std::memcmp(ref.data(), got.data(), ref.size())) { std::printf("fv5t7 panel mismatch\n"); return 1; }
        }
        // activations: N tokens, pair layout of pk-kernels.h
        const int Nmax = 375, npmax = (Nmax + 1) / 2;
        std::vector<int8_t> x((size_t) npmax * K * 2);
        for (auto & v : x) v = (int8_t) ((int) (rng() % 255) - 127);
        const size_t ps = (size_t) K * 2;

        struct layout { const char * name; size_t bytes; int id; };
        const layout ls[] = { { "pair16", (size_t) 4 * K, 0 }, { "b5x3t7", t7_bytes(K), 1 } };
        auto decode = [&](int id, int rb, int8_t * dst) {
            if (id == 0) {
                kt.decode_fv5(p16.data() + (size_t) rb * 4 * K, (size_t) K, luts.data() + (size_t) rb * 64, K, dst);
            } else {
                kt.decode_fv5t7(t7.data() + (size_t) rb * t7_bytes(K), luts.data() + (size_t) rb * 64, K, dst);
            }
        };
        std::vector<int8_t> panel((size_t) 8 * K);
        alignas(16) int32_t acc[64];
        for (const layout & L : ls) {
            double best_dec = 1e30;
            for (int r = 0; r < reps * 3; ++r) {
                const double t0 = now_s();
                for (int rb = 0; rb < nb; ++rb) decode(L.id, rb, panel.data());
                best_dec = std::min(best_dec, now_s() - t0);
            }
            std::printf("%-6d %-5d %-6s %10zu %10.0f %10.2f |", O, K, L.name, L.bytes, 1e9 * best_dec / nb,
                        (double) O * K / best_dec / 1e9);
            for (int N : Ns) {
                const int np = (N + 1) / 2, ntt = (np + 3) / 4;
                double best = 1e30;
                for (int r = 0; r < reps; ++r) {
                    const double t0 = now_s();
                    for (int rb = 0; rb < nb; ++rb) {
                        decode(L.id, rb, panel.data());
                        for (int tt = 0; tt < ntt; ++tt) {
                            kt.tile(panel.data(), x.data() + (size_t) tt * 4 * ps, ps, K, std::min(4, np - tt * 4), acc);
                        }
                    }
                    best = std::min(best, now_s() - t0);
                }
                std::printf(" %d:%.3f", N, 1e3 * best);
            }
            std::printf("\n");
        }
    }
    return 0;
}
