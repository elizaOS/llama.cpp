// tools/parakeet/tests/test_parakeet_parity.cpp - compare the native pipeline
// with reference_transformers.py (dense fp32 ParakeetForTDT) stage by stage.
//
//   test-parakeet-parity model.gguf fixtures_dir [--report out.tsv]
//
// fixtures_dir is written by tools/dump_parakeet_reference.py. Gates:
//   mel        max |diff| <= 2e-3 (normalised log-mel)
//   encoder    mean per-frame cosine >= PK_GATE_ENC_COS and relative L2
//              error <= PK_GATE_ENC_L2; the minimum frame cosine is reported
//              (near-silent frames have tiny norms)
//
// Two modes:
//   default               the shipped int8-activation path. Per-token int8
//                         activations through 24 layers leave a relative L2
//                         of ~0.07-0.2 on the encoder output (chaotic in the
//                         input), so the gates are cos >= 0.98, L2 <= 0.25;
//                         the transcript gate is the functional one.
//   PK_DEBUG_F32_ACTS=1   the same graph and weights with fp32 activations
//                         (slow diagnostic GEMM): checks the graph itself
//                         against the reference, cos >= 0.9999, L2 <= 0.02
//                         (what remains is the fp16 key cache and the
//                         per-row int8 grid of the five-value levels).
//   transcript at least PK_GATE_TEXT (default 0.9) of the utterances
//              text-identical to the reference
// The native encoder quantises activations to int8 (see parakeet.cpp), so
// it is not bit-exact; the report prints the actual numbers.
#include "parakeet.h"
#include "pk-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool read_f32(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const size_t n = (size_t) f.tellg();
    f.seekg(0);
    out.resize(n / 4);
    f.read((char *) out.data(), (std::streamsize) n);
    return (bool) f;
}

struct cmp {
    double max_abs = 0, min_cos = 1, mean_cos = 0, rel_l2 = 0;
};

static cmp compare(const std::vector<float> & a, const std::vector<float> & b, int rows, int cols) {
    cmp c;
    double sc = 0, de = 0, nb = 0;
    for (int r = 0; r < rows; ++r) {
        double ab = 0, aa = 0, bb = 0;
        for (int k = 0; k < cols; ++k) {
            const double x = a[(size_t) r * cols + k], y = b[(size_t) r * cols + k];
            c.max_abs = std::max(c.max_abs, std::fabs(x - y));
            ab += x * y;
            aa += x * x;
            bb += y * y;
            de += (x - y) * (x - y);
            nb += y * y;
        }
        const double cs = ab / (std::sqrt(aa * bb) + 1e-30);
        c.min_cos = std::min(c.min_cos, cs);
        sc += cs;
    }
    c.mean_cos = rows ? sc / rows : 1;
    c.rel_l2 = std::sqrt(de / (nb + 1e-30));
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf fixtures_dir\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    const bool f32_acts = std::getenv("PK_DEBUG_F32_ACTS") != nullptr;
    const double gate_cos = std::getenv("PK_GATE_ENC_COS") ? std::atof(std::getenv("PK_GATE_ENC_COS"))
                                                           : (f32_acts ? 0.9999 : 0.98);
    const double gate_l2 = std::getenv("PK_GATE_ENC_L2") ? std::atof(std::getenv("PK_GATE_ENC_L2"))
                                                         : (f32_acts ? 0.02 : 0.25);
    const double gate_text = std::getenv("PK_GATE_TEXT") ? std::atof(std::getenv("PK_GATE_TEXT")) : 0.9;
    char err[512] = { 0 };
    pk_params pp = pk_default_params();
    pk_model * m = pk_model_load(argv[1], pp, err, sizeof(err));
    if (!m) {
        std::fprintf(stderr, "load: %s\n", err);
        return 1;
    }
    std::printf("kernel %s, %d threads, fv5 int8 grid max rel err %.2e\n", pk_model_kernel_name(m),
                pk_model_n_threads(m), m->fv5_grid_err);
    std::ifstream man(dir + "/manifest.tsv");
    if (!man) {
        std::fprintf(stderr, "no %s/manifest.tsv\n", dir.c_str());
        return 1;
    }
    std::string line;
    int n = 0, same_text = 0, same_ids = 0, fails = 0, n_enc = 0;
    double worst_mel = 0, worst_cos = 1, worst_mean_cos = 1, worst_l2 = 0;
    while (std::getline(man, line)) {
        std::vector<std::string> f;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, '\t')) f.push_back(x);
        if (f.size() < 5) continue;
        const std::string id = f[0];
        const int n_mel = std::atoi(f[2].c_str()), T = std::atoi(f[3].c_str());
        std::vector<int> ref_ids;
        {
            std::stringstream is(f[4]);
            std::string t;
            while (std::getline(is, t, ',')) if (!t.empty()) ref_ids.push_back(std::atoi(t.c_str()));
        }
        const std::string ref_text = f.size() > 5 ? f[5] : "";
        std::vector<float> pcm, rmel, renc;
        if (!read_f32(dir + "/" + id + ".pcm.f32", pcm)) {
            std::fprintf(stderr, "%s: missing pcm\n", id.c_str());
            ++fails;
            continue;
        }
        std::vector<float> mel;
        int nm = 0;
        pk::compute_mel(m, pcm.data(), pcm.size(), mel, nm);
        std::printf("%-18s n_mel %d/%d", id.c_str(), nm, n_mel);
        if (nm != n_mel) ++fails;
        if (read_f32(dir + "/" + id + ".mel.f32", rmel) && (int) rmel.size() == n_mel * 128 && nm == n_mel) {
            const cmp c = compare(mel, rmel, nm, 128);
            worst_mel = std::max(worst_mel, c.max_abs);
            std::printf("  mel max|d| %.2e", c.max_abs);
        }
        std::vector<float> encp, enc, sub, l0;
        const int Tn = pk::encode(m, mel.data(), nm, encp, nullptr, nullptr, &enc, &sub, &l0);
        std::printf("  T %d/%d", Tn, T);
        if (Tn != T) ++fails;
        std::vector<float> r;
        if (Tn == T && read_f32(dir + "/" + id + ".sub.f32", r) && (int) r.size() == T * 1024) {
            const cmp c = compare(sub, r, T, 1024);
            std::printf("  sub cos min %.5f", c.min_cos);
        }
        if (Tn == T && read_f32(dir + "/" + id + ".l0.f32", r) && (int) r.size() == T * 1024) {
            const cmp c = compare(l0, r, T, 1024);
            std::printf("  l0 cos min %.5f", c.min_cos);
        }
        if (Tn == T && read_f32(dir + "/" + id + ".enc.f32", renc) && (int) renc.size() == T * 1024) {
            const cmp c = compare(enc, renc, T, 1024);
            worst_cos = std::min(worst_cos, c.min_cos);
            worst_mean_cos = std::min(worst_mean_cos, c.mean_cos);
            worst_l2 = std::max(worst_l2, c.rel_l2);
            ++n_enc;
            std::printf("  enc cos min %.5f mean %.5f relL2 %.4f", c.min_cos, c.mean_cos, c.rel_l2);
        }
        std::vector<pk_token> toks;
        pk::tdt_decode(m, encp.data(), Tn, toks, nullptr, nullptr);
        std::vector<int> ids;
        for (auto & t : toks) ids.push_back(t.id);
        const std::string text = pk::detokenize(m, toks.data(), toks.size());
        const bool st = text == ref_text, si = ids == ref_ids;
        same_text += st;
        same_ids += si;
        std::printf("  text %s\n", st ? "same" : "DIFF");
        if (!st) std::printf("    ref: %s\n    got: %s\n", ref_text.c_str(), text.c_str());
        ++n;
    }
    std::printf("\n%d utterances: text identical %d, token ids identical %d; worst mel max|d| %.2e; encoder over %d "
                "dumps: worst mean frame cosine %.5f, worst min frame cosine %.5f, worst relative L2 %.4f\n",
                n, same_text, same_ids, worst_mel, n_enc, worst_mean_cos, worst_cos, worst_l2);
    if (worst_mel > 2e-3) {
        std::printf("FAIL: mel differs by %.2e > 2e-3\n", worst_mel);
        ++fails;
    }
    if (n_enc && worst_mean_cos < gate_cos) {
        std::printf("FAIL: encoder mean frame cosine %.5f < %.5f\n", worst_mean_cos, gate_cos);
        ++fails;
    }
    if (n_enc && worst_l2 > gate_l2) {
        std::printf("FAIL: encoder relative L2 %.4f > %.4f\n", worst_l2, gate_l2);
        ++fails;
    }
    if (n && same_text < gate_text * n) {
        std::printf("FAIL: %d/%d transcripts identical < %.0f%%\n", same_text, n, gate_text * 100);
        ++fails;
    }
    pk_model_free(m);
    std::printf(fails ? "PARITY FAILED\n" : "PARITY OK\n");
    return fails ? 1 : 0;
}
