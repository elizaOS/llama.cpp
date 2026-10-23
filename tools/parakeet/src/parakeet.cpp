// tools/parakeet/src/parakeet.cpp - Parakeet-TDT (Phonon-2) loader, log-mel
// front end, FastConformer encoder, greedy TDT decoder and the batch API.
//
// Numerics follow transformers' ParakeetFeatureExtractor / ParakeetEncoder /
// ParakeetForTDT (the reference Fermion ships). Differences from the dense
// fp32 reference: activations entering the five-value and int6 linears are
// quantised to int8 per token, the five-value levels are put on a per-row
// int8 grid (max relative error reported as fv5_grid_err), and the decoder's
// LSTM / joint matvecs use int8 activations. tests/ gates the effect.
#include "parakeet.h"

#include "pk-model.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if (defined(__ARM_NEON) || defined(__aarch64__)) && !defined(_MSC_VER)
#include <arm_neon.h>
#define PK_NEON 1
#endif

using clk = std::chrono::steady_clock;

static constexpr double PK_PI = 3.14159265358979323846;

// PK_PROFILE=1: per-stage wall time of the encoder, printed at exit
namespace {
struct prof_t {
    bool on = std::getenv("PK_PROFILE") != nullptr;
    double t[16] = {};
    const char * name[16] = { "subsample", "layernorm", "quantize", "ff gemm", "qkv gemm", "pos cache", "attention",
                              "out gemm", "pw1 gemm", "glu+dwconv", "pw2 gemm", "enc_proj", "", "", "", "" };
    ~prof_t() {
        if (!on) return;
        double tot = 0;
        for (double v : t) tot += v;
        for (int i = 0; i < 12; ++i) std::fprintf(stderr, "[pk-profile] %-12s %9.1f ms %5.1f%%\n", name[i], t[i], 100 * t[i] / (tot + 1e-9));
    }
} g_prof;
struct ptimer {
    int k;
    clk::time_point t0;
    explicit ptimer(int k_) : k(k_), t0(g_prof.on ? clk::now() : clk::time_point()) {}
    ~ptimer() { if (g_prof.on) g_prof.t[k] += std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
};
}  // namespace
#define PKP(k) ptimer pk_pt_##__LINE__(k)

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

namespace pk {

// ------------------------------------------------------------- file map

mapped_file::~mapped_file() {
#if !defined(_WIN32)
    if (addr) {
        munmap(addr, size);
    }
#endif
}

bool mapped_file::open(const std::string & path, bool use_mmap, std::string & err) {
#if !defined(_WIN32)
    if (use_mmap) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            err = "cannot open " + path;
            return false;
        }
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            err = "cannot stat " + path;
            return false;
        }
        size = (size_t) st.st_size;
        void * p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) {
            err = "mmap failed for " + path;
            size = 0;
            return false;
        }
        addr = p;
        return true;
    }
#else
    (void) use_mmap;
#endif
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    size = (size_t) std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    heap.resize(size);
    const bool ok = std::fread(heap.data(), 1, size, f) == size;
    std::fclose(f);
    if (!ok) {
        err = "short read on " + path;
    }
    return ok;
}

int subsampled_length(int n) {
    for (int i = 0; i < 3; ++i) {
        n = (n + 2 - 3) / 2 + 1;   // kernel 3, stride 2, padding 1
        if (n < 0) n = 0;
    }
    return n;
}

}  // namespace pk

using namespace pk;

// --------------------------------------------------------------- loading

namespace {

struct loader {
    pk_model * m;
    std::string err;

    const ggml_tensor * meta_of(const std::string & name) {
        return ggml_get_tensor(m->meta, name.c_str());
    }

    const void * data(const std::string & name, ggml_type type, std::initializer_list<int64_t> ne) {
        if (!err.empty()) {
            return nullptr;
        }
        const int64_t id = gguf_find_tensor(m->gg, name.c_str());
        const ggml_tensor * t = meta_of(name);
        if (id < 0 || !t) {
            err = "missing tensor " + name;
            return nullptr;
        }
        if (t->type != type) {
            err = "tensor " + name + ": unexpected type";
            return nullptr;
        }
        int i = 0;
        for (int64_t d : ne) {
            if (t->ne[i] != d) {
                char buf[256];
                std::snprintf(buf, sizeof(buf), "tensor %s: ne[%d] = %lld, expected %lld", name.c_str(), i,
                              (long long) t->ne[i], (long long) d);
                err = buf;
                return nullptr;
            }
            ++i;
        }
        const size_t off = gguf_get_data_offset(m->gg) + gguf_get_tensor_offset(m->gg, id);
        if (off + ggml_nbytes(t) > m->file.size) {
            err = "tensor " + name + " runs past the end of the file";
            return nullptr;
        }
        return m->file.data() + off;
    }
    const float * f32(const std::string & n, std::initializer_list<int64_t> ne) {
        return (const float *) data(n, GGML_TYPE_F32, ne);
    }
    const int8_t * i8(const std::string & n, std::initializer_list<int64_t> ne) {
        return (const int8_t *) data(n, GGML_TYPE_I8, ne);
    }

    uint32_t u32(const char * key, uint32_t def) {
        const std::string k = std::string("parakeet.") + key;
        const int64_t id = gguf_find_key(m->gg, k.c_str());
        if (id < 0) return def;
        if (gguf_get_kv_type(m->gg, id) != GGUF_TYPE_UINT32) {
            if (err.empty()) err = k + ": expected uint32";
            return def;
        }
        return gguf_get_val_u32(m->gg, id);
    }
    float f32kv(const char * key, float def) {
        const std::string k = std::string("parakeet.") + key;
        const int64_t id = gguf_find_key(m->gg, k.c_str());
        if (id < 0) return def;
        if (gguf_get_kv_type(m->gg, id) != GGUF_TYPE_FLOAT32) {
            if (err.empty()) err = k + ": expected float32";
            return def;
        }
        return gguf_get_val_f32(m->gg, id);
    }

    void fv5(wmat & w, const std::string & name, int O, int K) {
        if (O % 8 || K % 16) {
            err = name + ": rows must be a multiple of 8 and width a multiple of 16";
            return;
        }
        const int8_t * codes = m->fv5t7 ? i8(name + ".fv5t7", { (int64_t) fv5t7_block_bytes(K), O / 8 })
                                        : i8(name + ".fv5", { K, O / 2 });
        const float * lohi = f32(name + ".lohi", { 2, O });
        if (!err.empty()) {
            return;
        }
        wmat_from_fv5(w, (const uint8_t *) codes, lohi, O, K, m->fv5t7);
        m->fv5_grid_err = std::max(m->fv5_grid_err, w.grid_err);
    }
    void intw(wmat & w, const std::string & name, int O, int K, const float * bias, const int * perm = nullptr) {
        const int8_t * q = i8(name + ".q", { K, O });
        const float * s = f32(name + ".scale", { O });
        if (!err.empty()) {
            return;
        }
        wmat_from_int8(w, q, s, O, K, perm);
        w.bias = bias;
    }
};

void setup_frontend(pk_model * m) {
    const auto & hp = m->hp;
    const int nb = hp.n_fft / 2 + 1;
    m->mel_lo.assign(hp.n_mels, 0);
    m->mel_hi.assign(hp.n_mels, 0);
    for (int i = 0; i < hp.n_mels; ++i) {
        int lo = nb, hi = 0;
        for (int k = 0; k < nb; ++k) {
            if (m->mel_fb[(size_t) i * nb + k] != 0.0f) {
                lo = std::min(lo, k);
                hi = k + 1;
            }
        }
        m->mel_lo[i] = lo < hi ? lo : 0;
        m->mel_hi[i] = lo < hi ? hi : 0;
    }
    // torch.hann_window(win, periodic=False), zero-padded to n_fft and centred
    m->window.assign(hp.n_fft, 0.0f);
    const int off = (hp.n_fft - hp.win_length) / 2;
    for (int i = 0; i < hp.win_length; ++i) {
        m->window[off + i] = (float) (0.5 - 0.5 * std::cos(2.0 * PK_PI * i / (hp.win_length - 1)));
    }
}

}  // namespace

extern "C" pk_params pk_default_params(void) {
    pk_params p;
    p.n_threads = 0;
    p.kernel = 0;
    p.pin_threads = 1;
    p.use_mmap = 1;
    return p;
}

extern "C" int pk_gguf_is_parakeet(const char * path) {
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) {
        return 0;
    }
    const int64_t id = gguf_find_key(g, "general.architecture");
    const int ok = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
                   std::strcmp(gguf_get_val_str(g, id), "parakeet-tdt") == 0;
    gguf_free(g);
    return ok;
}

static void set_err(char * err, size_t n, const std::string & s) {
    if (err && n > 0) {
        std::snprintf(err, n, "%s", s.c_str());
    }
}

extern "C" pk_model * pk_model_load(const char * path, pk_params params, char * err, size_t err_len) {
    auto m = std::make_unique<pk_model>();
    std::string e;
    if (!m->file.open(path, params.use_mmap != 0, e)) {
        set_err(err, err_len, e);
        return nullptr;
    }
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &m->meta };
    m->gg = gguf_init_from_file(path, ip);
    if (!m->gg) {
        set_err(err, err_len, std::string("not a GGUF file: ") + path);
        return nullptr;
    }
    {
        // gguf_get_val_* abort on a type mismatch, so check the type first
        auto str_kv = [&](const char * key) -> const char * {
            const int64_t kid = gguf_find_key(m->gg, key);
            return kid >= 0 && gguf_get_kv_type(m->gg, kid) == GGUF_TYPE_STRING ? gguf_get_val_str(m->gg, kid) : nullptr;
        };
        const char * arch = str_kv("general.architecture");
        if (!arch || std::strcmp(arch, "parakeet-tdt") != 0) {
            set_err(err, err_len, "general.architecture is not parakeet-tdt");
            return nullptr;
        }
        const char * fmt = str_kv("parakeet.weights.format");
        if (fmt && std::strcmp(fmt, "fv5-b5x3t7-v1") == 0) {
            m->fv5t7 = true;
        } else if (!fmt || std::strcmp(fmt, "fv5-pair16-v1") != 0) {
            set_err(err, err_len,
                    "parakeet.weights.format is neither fv5-b5x3t7-v1 nor fv5-pair16-v1 (re-run convert_phonon2_to_gguf.py)");
            return nullptr;
        }
        if (const char * notice = str_kv("parakeet.notice")) {
            m->notice = notice;
        }
    }
    loader L{ m.get(), {} };
    auto & hp = m->hp;
    hp.d_model = (int) L.u32("d_model", 1024);
    hp.n_layers = (int) L.u32("n_layers", 24);
    hp.n_heads = (int) L.u32("n_heads", 8);
    hp.ff_dim = (int) L.u32("ff_dim", 4096);
    hp.conv_kernel = (int) L.u32("conv_kernel", 9);
    hp.sub_channels = (int) L.u32("sub_channels", 256);
    hp.sub_factor = (int) L.u32("sub_factor", 8);
    hp.n_mels = (int) L.u32("n_mels", 128);
    hp.sample_rate = (int) L.u32("sample_rate", 16000);
    hp.n_fft = (int) L.u32("n_fft", 512);
    hp.win_length = (int) L.u32("win_length", 400);
    hp.hop_length = (int) L.u32("hop_length", 160);
    hp.pred_hidden = (int) L.u32("pred_hidden", 640);
    hp.pred_layers = (int) L.u32("pred_layers", 2);
    hp.joint_hidden = (int) L.u32("joint_hidden", 640);
    hp.vocab = (int) L.u32("vocab", 8192);
    hp.blank_id = (int) L.u32("blank_id", 8192);
    hp.max_symbols = (int) L.u32("max_symbols", 10);
    hp.preemph = L.f32kv("preemph", 0.97f);
    hp.log_guard = L.f32kv("log_guard", 5.9604644775390625e-8f);
    hp.norm_eps = L.f32kv("norm_eps", 1e-5f);
    hp.ln_eps = L.f32kv("ln_eps", 1e-5f);
    if (!L.err.empty()) {
        set_err(err, err_len, L.err);
        return nullptr;
    }
    if (hp.sub_factor != 8 || hp.pred_layers != 2 || hp.n_heads <= 0 || hp.d_model % hp.n_heads ||
        hp.d_model / hp.n_heads != 128 || hp.n_mels != 128 || hp.n_fft != 512 || hp.n_layers <= 0 ||
        hp.vocab <= 0 || hp.blank_id != hp.vocab || hp.max_symbols <= 0 || hp.hop_length <= 0 ||
        hp.win_length <= 1 || hp.win_length > hp.n_fft || hp.sample_rate <= 0 || hp.conv_kernel <= 0 ||
        hp.conv_kernel % 2 == 0) {
        set_err(err, err_len, "unsupported parakeet-tdt hyper-parameters");
        return nullptr;
    }
    {
        const int64_t id = gguf_find_key(m->gg, "parakeet.durations");
        if (id >= 0) {
            if (gguf_get_kv_type(m->gg, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(m->gg, id) != GGUF_TYPE_INT32) {
                set_err(err, err_len, "parakeet.durations: expected an int32 array");
                return nullptr;
            }
            const size_t n = gguf_get_arr_n(m->gg, id);
            hp.durations.clear();
            for (size_t i = 0; i < n; ++i) {
                hp.durations.push_back(((const int32_t *) gguf_get_arr_data(m->gg, id))[i]);
            }
        }
        // the greedy decoder indexes durations and advances by them: a
        // negative duration would loop forever, an empty table read past it
        bool ok = !hp.durations.empty() && hp.durations.size() <= 64;
        for (int d : hp.durations) ok = ok && d >= 0 && d <= 64;
        if (!ok) {
            set_err(err, err_len, "parakeet.durations missing or out of range");
            return nullptr;
        }
    }
    {
        const int64_t id = gguf_find_key(m->gg, "tokenizer.ggml.tokens");
        if (id < 0 || gguf_get_kv_type(m->gg, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(m->gg, id) != GGUF_TYPE_STRING || (int) gguf_get_arr_n(m->gg, id) != hp.vocab) {
            set_err(err, err_len, "tokenizer.ggml.tokens missing or wrong size");
            return nullptr;
        }
        m->pieces.resize(hp.vocab);
        m->is_control.assign(hp.vocab, 0);
        m->word_start.assign(hp.vocab, 0);
        for (int i = 0; i < hp.vocab; ++i) {
            std::string s = gguf_get_arr_str(m->gg, id, i);
            const bool ctrl = s.size() >= 2 && s.front() == '<' && s.back() == '>' && s.find("\xe2\x96\x81") == std::string::npos;
            m->is_control[i] = ctrl;
            std::string o;
            for (size_t k = 0; k < s.size();) {
                if (s.compare(k, 3, "\xe2\x96\x81") == 0) {
                    o += ' ';
                    k += 3;
                } else {
                    o += s[k++];
                }
            }
            m->word_start[i] = !o.empty() && o[0] == ' ';
            m->pieces[i] = o;
        }
    }

    const int D = hp.d_model, C = hp.sub_channels, F = hp.ff_dim, H = hp.pred_hidden, J = hp.joint_hidden;
    const int nb = hp.n_fft / 2 + 1;
    m->mel_fb = L.f32("feat.mel_fb", { nb, hp.n_mels });

    auto transpose9 = [&](const float * w, std::vector<float> & out) {
        out.assign((size_t) 9 * C, 0.0f);
        if (!w) return;
        for (int c = 0; c < C; ++c) {
            for (int k = 0; k < 9; ++k) {
                out[(size_t) k * C + c] = w[(size_t) c * 9 + k];
            }
        }
    };
    transpose9(L.f32("enc.sub.conv0.w", { 9, C }), m->c0_wT);
    m->c0_b = L.f32("enc.sub.conv0.b", { C });
    transpose9(L.f32("enc.sub.dw1.w", { 9, C }), m->dw1_wT);
    m->dw1_b = L.f32("enc.sub.dw1.b", { C });
    transpose9(L.f32("enc.sub.dw2.w", { 9, C }), m->dw2_wT);
    m->dw2_b = L.f32("enc.sub.dw2.b", { C });
    L.intw(m->sub_pw1, "enc.sub.pw1", C, C, L.f32("enc.sub.pw1.b", { C }));
    L.intw(m->sub_pw2, "enc.sub.pw2", C, C, L.f32("enc.sub.pw2.b", { C }));
    {
        // torch flattens (C, F') -> c * F' + f; the encoder keeps channel-last
        // rows (f * C + c), so the linear's columns are permuted at load.
        const int Fq = hp.n_mels / 8;
        std::vector<int> perm((size_t) Fq * C);
        for (int f = 0; f < Fq; ++f) {
            for (int c = 0; c < C; ++c) {
                perm[(size_t) f * C + c] = c * Fq + f;
            }
        }
        L.intw(m->sub_out, "enc.sub.out", D, C * Fq, L.f32("enc.sub.out.b", { D }), perm.data());
    }

    m->layers.resize(hp.n_layers);
    static const char * ln_names[5] = { "norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2",
                                        "norm_out" };
    for (int l = 0; l < hp.n_layers && L.err.empty(); ++l) {
        auto & ly = m->layers[l];
        const std::string p = "enc.l" + std::to_string(l) + ".";
        L.fv5(ly.ff1_w1, p + "ff1.w1", F, D);
        L.fv5(ly.ff1_w2, p + "ff1.w2", D, F);
        L.fv5(ly.qkv, p + "attn.qkv", 3 * D, D);
        L.fv5(ly.out, p + "attn.out", D, D);
        L.fv5(ly.pos, p + "attn.pos", D, D);
        L.fv5(ly.pw1, p + "conv.pw1", 2 * D, D);
        L.fv5(ly.pw2, p + "conv.pw2", D, D);
        L.fv5(ly.ff2_w1, p + "ff2.w1", F, D);
        L.fv5(ly.ff2_w2, p + "ff2.w2", D, F);
        ly.bias_u = L.f32(p + "attn.bias_u", { 128, hp.n_heads });
        ly.bias_v = L.f32(p + "attn.bias_v", { 128, hp.n_heads });
        for (int i = 0; i < 5; ++i) {
            ly.ln_w[i] = L.f32(p + ln_names[i] + ".w", { D });
            ly.ln_b[i] = L.f32(p + ln_names[i] + ".b", { D });
        }
        ly.dw_w = L.f32(p + "conv.dw.w", { hp.conv_kernel, D });
        ly.dw_b = L.f32(p + "conv.dw.b", { D });
        if (ly.dw_w) {
            ly.dw_wT.resize((size_t) hp.conv_kernel * D);
            for (int c = 0; c < D; ++c) {
                for (int k = 0; k < hp.conv_kernel; ++k) {
                    ly.dw_wT[(size_t) k * D + c] = ly.dw_w[(size_t) c * hp.conv_kernel + k];
                }
            }
        }
    }
    L.intw(m->enc_proj, "enc_proj", J, D, L.f32("enc_proj.b", { J }));
    m->embed_q = L.i8("dec.embed.q", { H, hp.vocab + 1 });
    m->embed_s = L.f32("dec.embed.scale", { hp.vocab + 1 });
    for (int l = 0; l < 2; ++l) {
        const std::string p = "dec.lstm" + std::to_string(l);
        m->lstm_ih[l] = L.i8(p + ".ih.q", { H, 4 * H });
        m->lstm_ih_s[l] = L.f32(p + ".ih.scale", { 4 * H });
        m->lstm_hh[l] = L.i8(p + ".hh.q", { H, 4 * H });
        m->lstm_hh_s[l] = L.f32(p + ".hh.scale", { 4 * H });
        m->lstm_b[l] = L.f32(p + ".b", { 4 * H });
    }
    m->proj_q = L.i8("dec.proj.q", { H, J });
    m->proj_s = L.f32("dec.proj.scale", { J });
    m->proj_b = L.f32("dec.proj.b", { J });
    m->head_rows = hp.vocab + 1 + (int) hp.durations.size();
    m->head_q = L.i8("joint.head.q", { J, m->head_rows });
    m->head_s = L.f32("joint.head.scale", { m->head_rows });
    m->head_b = L.f32("joint.head.b", { m->head_rows });
    if (!L.err.empty()) {
        set_err(err, err_len, L.err);
        return nullptr;
    }
    if (H % 16 || J % 16 || D % 16) {
        set_err(err, err_len, "decoder widths must be multiples of 16");
        return nullptr;
    }
    setup_frontend(m.get());

    m->kt = &kernels_for(params.kernel);
    const int nt = params.n_threads > 0 ? params.n_threads : default_threads();
    m->pool = std::make_unique<pool>(nt, params.pin_threads != 0);
    return m.release();
}

extern "C" void pk_model_free(pk_model * m) {
    if (!m) {
        return;
    }
    if (m->gg) {
        gguf_free(m->gg);
    }
    if (m->meta) {
        ggml_free(m->meta);
    }
    delete m;
}

extern "C" const char * pk_model_kernel_name(const pk_model * m) { return m ? m->kt->name : ""; }
extern "C" int pk_model_n_threads(const pk_model * m) { return m ? m->pool->size() : 0; }
extern "C" int pk_model_sample_rate(const pk_model * m) { return m ? m->hp.sample_rate : 0; }
extern "C" const char * pk_model_notice(const pk_model * m) { return m ? m->notice.c_str() : ""; }

// ------------------------------------------------------------ front end

namespace {

// in-place radix-2 FFT of size n (power of two), double precision
void fft(std::vector<std::complex<double>> & a, const std::vector<std::complex<double>> & tw) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t step = n / len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = a[i + k];
                const std::complex<double> v = a[i + k + len / 2] * tw[k * step];
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
        }
    }
}

}  // namespace

bool pk::compute_mel(pk_model * m, const float * pcm, size_t n, std::vector<float> & mel, int & n_frames) {
    const auto & hp = m->hp;
    const int hop = hp.hop_length, nfft = hp.n_fft, nm = hp.n_mels, nb = nfft / 2 + 1;
    n_frames = (int) (n / (size_t) hop);
    if (n_frames < 2) {
        mel.clear();
        n_frames = 0;
        return false;
    }
    std::vector<float> y(n);
    y[0] = pcm[0];
    for (size_t i = 1; i < n; ++i) {
        y[i] = pcm[i] - hp.preemph * pcm[i - 1];
    }
    mel.assign((size_t) n_frames * nm, 0.0f);
    std::vector<std::complex<double>> tw(nfft / 2);
    for (int k = 0; k < nfft / 2; ++k) {
        tw[k] = std::polar(1.0, -2.0 * PK_PI * k / nfft);
    }
    auto frames = [&](int b, int e) {
        std::vector<std::complex<double>> buf(nfft);
        std::vector<float> pw(nb);
        for (int f = b; f < e; ++f) {
            const long c = (long) f * hop - nfft / 2;
            for (int i = 0; i < nfft; ++i) {
                const long s = c + i;
                const float v = (s >= 0 && s < (long) n) ? y[s] : 0.0f;
                buf[i] = std::complex<double>((double) v * (double) m->window[i], 0.0);
            }
            fft(buf, tw);
            for (int k = 0; k < nb; ++k) {
                const float re = (float) buf[k].real(), im = (float) buf[k].imag();
                pw[k] = re * re + im * im;
            }
            float * out = mel.data() + (size_t) f * nm;
            for (int i = 0; i < nm; ++i) {
                const float * fb = m->mel_fb + (size_t) i * nb;
                float s = 0.0f;
                for (int k = m->mel_lo[i]; k < m->mel_hi[i]; ++k) {
                    s += fb[k] * pw[k];
                }
                out[i] = std::log(s + hp.log_guard);
            }
        }
    };
    m->pool->parallel_for(n_frames, 64, frames);
    // per-feature normalisation over the valid frames (unbiased variance)
    for (int i = 0; i < nm; ++i) {
        double mean = 0.0;
        for (int f = 0; f < n_frames; ++f) {
            mean += mel[(size_t) f * nm + i];
        }
        mean /= n_frames;
        double var = 0.0;
        for (int f = 0; f < n_frames; ++f) {
            const double d = mel[(size_t) f * nm + i] - mean;
            var += d * d;
        }
        var /= (n_frames - 1);
        const float sd = (float) std::sqrt(var) + hp.norm_eps;
        const float mf = (float) mean;
        for (int f = 0; f < n_frames; ++f) {
            float & v = mel[(size_t) f * nm + i];
            v = (v - mf) / sd;
        }
    }
    return true;
}

// -------------------------------------------------------------- encoder

namespace {

// conv0 (1 -> C, 3x3, stride 2, pad 1) + ReLU for output time rows [t0, t1),
// channel-last output rows [t][F1][C]
void conv0_rows(const pk_model * m, const float * mel, int Tm, int t0, int t1, float * out) {
    const int C = m->hp.sub_channels, Fin = m->hp.n_mels, F1 = Fin / 2;
    for (int t = t0; t < t1; ++t) {
        for (int f = 0; f < F1; ++f) {
            float * o = out + ((size_t) (t - t0) * F1 + f) * C;
            std::memcpy(o, m->c0_b, sizeof(float) * C);
            for (int dt = 0; dt < 3; ++dt) {
                const int ti = 2 * t + dt - 1;
                if (ti < 0 || ti >= Tm) continue;
                for (int df = 0; df < 3; ++df) {
                    const int fi = 2 * f + df - 1;
                    if (fi < 0 || fi >= Fin) continue;
                    const float v = mel[(size_t) ti * Fin + fi];
                    const float * w = m->c0_wT.data() + (size_t) (dt * 3 + df) * C;
                    int c = 0;
#if PK_NEON
                    const float32x4_t v4 = vdupq_n_f32(v);
                    for (; c + 4 <= C; c += 4) {
                        vst1q_f32(o + c, vfmaq_f32(vld1q_f32(o + c), v4, vld1q_f32(w + c)));
                    }
#endif
                    for (; c < C; ++c) o[c] += v * w[c];
                }
            }
            int c = 0;
#if PK_NEON
            for (; c + 4 <= C; c += 4) vst1q_f32(o + c, vmaxq_f32(vld1q_f32(o + c), vdupq_n_f32(0.0f)));
#endif
            for (; c < C; ++c) o[c] = o[c] > 0.0f ? o[c] : 0.0f;
        }
    }
}

// depthwise 3x3 stride 2 pad 1, channel-last. `in` holds rows [in_t0, ...)
// of a [Tin][Fin][C] map; rows outside [0, Tin) are zero padding.
void dw_rows(const float * in, int in_t0, int in_t1, int Tin, int Fin, int C, const float * wT, const float * b,
             int t0, int t1, float * out) {
    const int Fo = (Fin + 2 - 3) / 2 + 1;
    for (int t = t0; t < t1; ++t) {
        for (int f = 0; f < Fo; ++f) {
            float * o = out + ((size_t) (t - t0) * Fo + f) * C;
            std::memcpy(o, b, sizeof(float) * C);
            for (int dt = 0; dt < 3; ++dt) {
                const int ti = 2 * t + dt - 1;
                if (ti < 0 || ti >= Tin) continue;
                if (ti < in_t0 || ti >= in_t1) continue;  // never hit: callers pass the full halo
                for (int df = 0; df < 3; ++df) {
                    const int fi = 2 * f + df - 1;
                    if (fi < 0 || fi >= Fin) continue;
                    const float * x = in + ((size_t) (ti - in_t0) * Fin + fi) * C;
                    const float * w = wT + (size_t) (dt * 3 + df) * C;
                    int c = 0;
#if PK_NEON
                    for (; c + 4 <= C; c += 4) {
                        vst1q_f32(o + c, vfmaq_f32(vld1q_f32(o + c), vld1q_f32(x + c), vld1q_f32(w + c)));
                    }
#endif
                    for (; c < C; ++c) o[c] += x[c] * w[c];
                }
            }
        }
    }
}

void subsample(pk_model * m, const float * mel, int Tm, std::vector<float> & x, int & T3) {
    const int C = m->hp.sub_channels;
    const int T1 = (Tm - 1) / 2 + 1, T2 = (T1 - 1) / 2 + 1;
    T3 = (T2 - 1) / 2 + 1;
    const int F1 = m->hp.n_mels / 2, F2 = F1 / 2, F3 = F2 / 2;
    const int D = m->hp.d_model;
    auto & P = *m->pool;
    auto & ws = m->ws;
    std::vector<float> s3((size_t) T3 * F3 * C);
    const int CH = 16;  // T3 rows per chunk (1.28 s)
    for (int a3 = 0; a3 < T3; a3 += CH) {
        const int b3 = std::min(T3, a3 + CH);
        const int a2 = std::max(0, 2 * a3 - 1), b2 = std::min(T2, 2 * b3);
        const int a1 = std::max(0, 2 * a2 - 1), b1 = std::min(T1, 2 * b2);
        ws.sub_a.resize((size_t) (b1 - a1) * F1 * C);
        P.parallel_for(b1 - a1, 4, [&](int b, int e) { conv0_rows(m, mel, Tm, a1 + b, a1 + e, ws.sub_a.data() + (size_t) b * F1 * C); });
        ws.sub_b.resize((size_t) (b2 - a2) * F2 * C);
        P.parallel_for(b2 - a2, 2, [&](int b, int e) {
            dw_rows(ws.sub_a.data(), a1, b1, T1, F1, C, m->dw1_wT.data(), m->dw1_b, a2 + b, a2 + e,
                    ws.sub_b.data() + (size_t) b * F2 * C);
        });
        // pointwise 1x1 + ReLU over every (t, f) position
        const int n2 = (b2 - a2) * F2;
        quantize_acts(P, ws.sub_b.data(), n2, C, C, ws.qa);
        ws.sub_c.resize((size_t) n2 * C);
        gemm_q8(P, *m->kt, m->sub_pw1, ws.qa, ws.sub_c.data(), C, epi::relu);
        float * o3 = s3.data() + (size_t) a3 * F3 * C;
        P.parallel_for(b3 - a3, 2, [&](int b, int e) {
            dw_rows(ws.sub_c.data(), a2, b2, T2, F2, C, m->dw2_wT.data(), m->dw2_b, a3 + b, a3 + e,
                    o3 + (size_t) b * F3 * C);
        });
        const int n3 = (b3 - a3) * F3;
        quantize_acts(P, o3, n3, C, C, ws.qa);
        gemm_q8(P, *m->kt, m->sub_pw2, ws.qa, o3, C, epi::relu);
    }
    x.resize((size_t) T3 * D);
    quantize_acts(P, s3.data(), T3, F3 * C, (size_t) F3 * C, ws.qa);
    gemm_q8(P, *m->kt, m->sub_out, ws.qa, x.data(), D, epi::store);
}

inline void f16_to_f32_row(const uint16_t * src, float * dst, int n) {
    int i = 0;
#if PK_NEON
    for (; i + 8 <= n; i += 8) {
        const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src + i));
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_high_f32_f16(h));
    }
#endif
    for (; i < n; ++i) dst[i] = ggml_fp16_to_fp32((ggml_fp16_t) src[i]);
}

// sinusoidal relative-position embedding rows for positions p (the
// ParakeetEncoderRelPositionalEncoding values)
void rel_pos_rows(const std::vector<int> & pos, int D, std::vector<float> & pe) {
    const int n = (int) pos.size();
    pe.resize((size_t) n * D);
    for (int i = 0; i < D / 2; ++i) {
        const float inv = (float) (1.0 / std::pow(10000.0, (2.0 * i) / D));
        for (int r = 0; r < n; ++r) {
            const float a = inv * (float) pos[r];
            pe[(size_t) r * D + 2 * i] = std::sin(a);
            pe[(size_t) r * D + 2 * i + 1] = std::cos(a);
        }
    }
}

// make sure the per-layer position-key cache covers |p| <= need.
// Cache rows c = 0..2P hold position p = P - c. Every row is computed on its
// own (per-row activation scales), so growing the cache only computes the
// new rows at both ends and keeps the old ones: the result is bit-identical
// to a full rebuild, and a stream whose window grows pays for each row once.
void ensure_rpos_cache(pk_model * m, int need) {
    if (need <= m->rc_P) return;
    const int P = ((std::max(need, 1) + 63) / 64) * 64;
    const int D = m->hp.d_model, n = 2 * P + 1;
    const int oldP = m->rc_P;  // -1 when empty
    const int n_top = oldP < 0 ? n : P - oldP;           // positions P .. oldP+1 (or everything)
    const int n_bot = oldP < 0 ? 0 : P - oldP;           // positions -oldP-1 .. -P
    std::vector<int> pos;
    pos.reserve((size_t) n_top + n_bot);
    for (int i = 0; i < n_top; ++i) pos.push_back(P - i);
    for (int i = 0; i < n_bot; ++i) pos.push_back(-oldP - 1 - i);
    const int nn = (int) pos.size();
    std::vector<float> pe, r((size_t) nn * D);
    rel_pos_rows(pos, D, pe);
    // Two-pass (residual) int8 quantisation: pe ~ s1*q1 + s2*q2, so the
    // cached keys carry ~14-bit activations. The keys feed every attention
    // logit, and each row is built once, so the second pass is cheap.
    qact q, q2;
    quantize_acts(*m->pool, pe.data(), nn, D, D, q);
    std::vector<float> res(pe.size());
    for (int t = 0; t < nn; ++t) {
        for (int k = 0; k < D; ++k) {
            const int c = k / 8, j = k % 8;
            const int8_t v = q.buf[(size_t) (t / 2) * q.pair_stride() + (size_t) c * 16 + (t & 1) * 8 + j];
            res[(size_t) t * D + k] = pe[(size_t) t * D + k] - (float) v * q.scale[t];
        }
    }
    quantize_acts(*m->pool, res.data(), nn, D, D, q2);
    // PK_DEBUG_F32_ACTS runs the first pass on the fp32 source already
    const bool f32_debug = std::getenv("PK_DEBUG_F32_ACTS") != nullptr;
    m->rcache.resize(m->hp.n_layers);
    for (int l = 0; l < m->hp.n_layers; ++l) {
        gemm_q8(*m->pool, *m->kt, m->layers[l].pos, q, r.data(), D, epi::store);
        if (!f32_debug) {
            gemm_q8(*m->pool, *m->kt, m->layers[l].pos, q2, r.data(), D, epi::add, 1.0f);
        }
        std::vector<uint16_t> c((size_t) n * D);
        // new rows: [0, n_top) are cache rows 0..n_top-1; [n_top, nn) are the
        // bottom rows n - n_bot .. n - 1; the old cache sits in between
        auto put = [&](int src_row, int dst_row, int count) {
            m->pool->parallel_for(count, 16, [&](int b, int e) {
                for (int i = b; i < e; ++i) {
                    const float * sr = r.data() + (size_t) (src_row + i) * D;
                    uint16_t * dr = c.data() + (size_t) (dst_row + i) * D;
                    for (int k = 0; k < D; ++k) dr[k] = (uint16_t) ggml_fp32_to_fp16(sr[k]);
                }
            });
        };
        put(0, 0, n_top);
        if (oldP >= 0) {
            const auto & old = m->rcache[l];
            std::memcpy(c.data() + (size_t) n_top * D, old.data(), sizeof(uint16_t) * old.size());
            put(n_top, n - n_bot, n_bot);
        }
        m->rcache[l].swap(c);
    }
    m->rc_P = P;
}

// rel-pos multi-head self-attention (Transformer-XL), full context.
void attention(pk_model * m, const layer & ly, int l, int T) {
    auto & P = *m->pool;
    auto & ws = m->ws;
    const int D = m->hp.d_model, NH = m->hp.n_heads, HD = D / NH;
    const int Tp = (T + 3) & ~3;
    const float scale = 1.0f / std::sqrt((float) HD);
    // V^T per head: [NH][HD][Tp]
    ws.v.assign((size_t) NH * HD * Tp, 0.0f);
    P.parallel_for(NH, 1, [&](int b, int e) {
        for (int h = b; h < e; ++h) {
            float * vt = ws.v.data() + (size_t) h * HD * Tp;
            for (int t = 0; t < T; ++t) {
                const float * src = ws.qkv.data() + (size_t) t * 3 * D + 2 * D + h * HD;
                for (int d = 0; d < HD; ++d) {
                    vt[(size_t) d * Tp + t] = src[d];
                }
            }
        }
    });
    const int QB = 16;
    const int nqb = (T + QB - 1) / QB;
    if ((int) ws.att_tls.size() < P.size()) {
        ws.att_tls.resize(P.size());
    }
    ws.att.resize((size_t) T * D);
    std::atomic<int> next{ 0 };
    P.run([&](int tid) {
        auto & s = ws.att_tls[tid];
        const size_t need = (size_t) QB * HD * 2 + (size_t) QB * Tp + (size_t) QB * (T + QB) + (size_t) (T + QB) * HD;
        if (s.size() < need) s.resize(need);
        float * qu = s.data();
        float * qv = qu + QB * HD;
        float * S = qv + QB * HD;
        float * band = S + (size_t) QB * Tp;
        float * rk = band + (size_t) QB * (T + QB);
        const uint16_t * rc = m->rcache[l].data();
        const int rP = m->rc_P;
        for (;;) {
            const int task = next.fetch_add(1);
            if (task >= NH * nqb) break;
            const int h = task / nqb, qb = task % nqb;
            const int i0 = qb * QB, B = std::min(QB, T - i0);
            const float * bu = ly.bias_u + h * HD;
            const float * bv = ly.bias_v + h * HD;
            for (int i = 0; i < B; ++i) {
                const float * q = ws.qkv.data() + (size_t) (i0 + i) * 3 * D + h * HD;
                for (int d = 0; d < HD; ++d) {
                    qu[i * HD + d] = q[d] + bu[d];
                    qv[i * HD + d] = q[d] + bv[d];
                }
            }
            // content term (q + u) . k_j
            gemm_f32_nt(qu, HD, ws.qkv.data() + D + h * HD, (size_t) 3 * D, S, Tp, B, T, HD, 1.0f);
            // position term (q + v) . r_{i-j}: rows T-1-i+j of R, a band
            const int m0 = T - 1 - (i0 + B - 1);
            const int W = T + B - 1;
            // R row m (position T-1-m) is cache row m + rP - T + 1
            for (int w = 0; w < W; ++w) {
                f16_to_f32_row(rc + (size_t) (m0 + w + rP - T + 1) * D + h * HD, rk + (size_t) w * HD, HD);
            }
            gemm_f32_nt(qv, HD, rk, HD, band, W, B, W, HD, 1.0f);
            for (int i = 0; i < B; ++i) {
                float * srow = S + (size_t) i * Tp;
                const float * brow = band + (size_t) i * W + (B - 1 - i);
                float mx = -INFINITY;
                for (int j = 0; j < T; ++j) {
                    srow[j] = (srow[j] + brow[j]) * scale;
                    mx = std::max(mx, srow[j]);
                }
                for (int j = 0; j < T; ++j) srow[j] -= mx;
                exp_inplace(srow, T);
                float sum = 0.0f;
                for (int j = 0; j < T; ++j) sum += srow[j];
                const float inv = 1.0f / sum;
                for (int j = 0; j < T; ++j) srow[j] *= inv;
                for (int j = T; j < Tp; ++j) srow[j] = 0.0f;
            }
            gemm_f32_nt(S, Tp, ws.v.data() + (size_t) h * HD * Tp, Tp, ws.att.data() + (size_t) i0 * D + h * HD, D, B,
                        HD, Tp, 1.0f);
        }
    });
}

void conv_module(pk_model * m, const layer & ly, int T) {
    auto & P = *m->pool;
    auto & ws = m->ws;
    const int D = m->hp.d_model, K = m->hp.conv_kernel, pad = (K - 1) / 2;
    // GLU: u = a * sigmoid(b)
    ws.u.resize((size_t) T * D);
    P.parallel_for(T, 8, [&](int b, int e) {
        std::vector<float> tmp(D);
        for (int t = b; t < e; ++t) {
            const float * g = ws.g.data() + (size_t) t * 2 * D;
            for (int c = 0; c < D; ++c) tmp[c] = -g[D + c];
            exp_inplace(tmp.data(), D);
            float * u = ws.u.data() + (size_t) t * D;
            for (int c = 0; c < D; ++c) u[c] = g[c] / (1.0f + tmp[c]);
        }
    });
    // depthwise conv over time (BatchNorm folded) + SiLU
    ws.xn.resize((size_t) T * D);
    P.parallel_for(T, 8, [&](int b, int e) {
        for (int t = b; t < e; ++t) {
            float * o = ws.xn.data() + (size_t) t * D;
            std::memcpy(o, ly.dw_b, sizeof(float) * D);
            for (int k = 0; k < K; ++k) {
                const int ti = t + k - pad;
                if (ti < 0 || ti >= T) continue;
                const float * x = ws.u.data() + (size_t) ti * D;
                const float * w = ly.dw_wT.data() + (size_t) k * D;
                int c = 0;
#if PK_NEON
                for (; c + 4 <= D; c += 4) {
                    vst1q_f32(o + c, vfmaq_f32(vld1q_f32(o + c), vld1q_f32(w + c), vld1q_f32(x + c)));
                }
#endif
                for (; c < D; ++c) o[c] += w[c] * x[c];
            }
            silu_inplace(o, D);
        }
    });
}

}  // namespace

int pk::encode(pk_model * m, const float * mel, int n_mel, std::vector<float> & encp, pk_cancel_cb cancel, void * cu,
               std::vector<float> * enc_out, std::vector<float> * sub_out, std::vector<float> * l0_out) {
    auto & P = *m->pool;
    auto & ws = m->ws;
    const auto & kt = *m->kt;
    const int D = m->hp.d_model, F = m->hp.ff_dim;
    int T = 0;
    { PKP(0); subsample(m, mel, n_mel, ws.x, T); }
    if (sub_out) *sub_out = ws.x;
    if (T <= 0) return 0;
    { PKP(5); ensure_rpos_cache(m, T - 1); }
    ws.xn.resize((size_t) T * D);
    ws.h.resize((size_t) T * F);
    ws.qkv.resize((size_t) T * 3 * D);
    ws.g.resize((size_t) T * 2 * D);
    float * X = ws.x.data();
    const float eps = m->hp.ln_eps;
    auto ln = [&](const float * w, const float * b, float * out) {
        PKP(1);
        P.parallel_for(T, 8, [&](int bb, int ee) {
            layernorm(X + (size_t) bb * D, out + (size_t) bb * D, ee - bb, D, w, b, eps);
        });
    };
    auto quant = [&](const float * src, int K, qact & q) {
        PKP(2);
        quantize_acts(P, src, T, K, K, q);
    };
    for (int l = 0; l < m->hp.n_layers; ++l) {
        if (cancel && cancel(cu)) return -1;
        const layer & ly = m->layers[l];
        // FF1 (macaron, x0.5)
        ln(ly.ln_w[0], ly.ln_b[0], ws.xn.data());
        quant(ws.xn.data(), D, ws.qa);
        { PKP(3); gemm_q8(P, kt, ly.ff1_w1, ws.qa, ws.h.data(), F, epi::silu); }
        quant(ws.h.data(), F, ws.qh);
        { PKP(3); gemm_q8(P, kt, ly.ff1_w2, ws.qh, X, D, epi::add, 0.5f); }
        // self-attention
        ln(ly.ln_w[1], ly.ln_b[1], ws.xn.data());
        quant(ws.xn.data(), D, ws.qa);
        { PKP(4); gemm_q8(P, kt, ly.qkv, ws.qa, ws.qkv.data(), 3 * D, epi::store); }
        { PKP(6); attention(m, ly, l, T); }
        quant(ws.att.data(), D, ws.qa);
        { PKP(7); gemm_q8(P, kt, ly.out, ws.qa, X, D, epi::add, 1.0f); }
        // convolution module
        ln(ly.ln_w[2], ly.ln_b[2], ws.xn.data());
        quant(ws.xn.data(), D, ws.qa);
        { PKP(8); gemm_q8(P, kt, ly.pw1, ws.qa, ws.g.data(), 2 * D, epi::store); }
        { PKP(9); conv_module(m, ly, T); }
        quant(ws.xn.data(), D, ws.qa);
        { PKP(10); gemm_q8(P, kt, ly.pw2, ws.qa, X, D, epi::add, 1.0f); }
        // FF2 (x0.5)
        ln(ly.ln_w[3], ly.ln_b[3], ws.xn.data());
        quant(ws.xn.data(), D, ws.qa);
        { PKP(3); gemm_q8(P, kt, ly.ff2_w1, ws.qa, ws.h.data(), F, epi::silu); }
        quant(ws.h.data(), F, ws.qh);
        { PKP(3); gemm_q8(P, kt, ly.ff2_w2, ws.qh, X, D, epi::add, 0.5f); }
        // final layer norm, in place
        ln(ly.ln_w[4], ly.ln_b[4], X);
        if (l == 0 && l0_out) *l0_out = ws.x;
    }
    if (enc_out) *enc_out = ws.x;
    const int J = m->hp.joint_hidden;
    encp.resize((size_t) T * J);
    PKP(11);
    quantize_acts(P, X, T, D, D, ws.qa);
    gemm_q8(P, kt, m->enc_proj, ws.qa, encp.data(), J, epi::store);
    return T;
}

// -------------------------------------------------------------- decoder

namespace {

struct qvec {
    std::vector<int8_t> q;
    float s = 0.0f;
};

void quant_vec(const float * x, int n, qvec & out) {
    out.q.resize(n);
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    if (!(amax > 0.0f)) {
        std::fill(out.q.begin(), out.q.end(), 0);
        out.s = 0.0f;
        return;
    }
    const float inv = 127.0f / amax;
    for (int i = 0; i < n; ++i) {
        out.q[i] = (int8_t) std::max(-127.0f, std::min(127.0f, std::nearbyint(x[i] * inv)));
    }
    out.s = amax / 127.0f;
}

struct dec_state {
    std::vector<float> h[2], c[2], out;
};

inline float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void lstm_step(const pk_model * m, int token, dec_state & st, std::vector<int32_t> & a1, std::vector<int32_t> & a2,
               std::vector<float> & gates, qvec & xq, qvec & hq) {
    const int H = m->hp.pred_hidden;
    const auto & kt = *m->kt;
    // layer 0 input: the int8 embedding row is its own quantisation
    const int8_t * x8 = m->embed_q + (size_t) token * H;
    float xs = m->embed_s[token];
    for (int l = 0; l < 2; ++l) {
        if (l == 1) {
            quant_vec(st.h[0].data(), H, xq);
            x8 = xq.q.data();
            xs = xq.s;
        }
        quant_vec(st.h[l].data(), H, hq);
        kt.matvec(m->lstm_ih[l], 4 * H, H, x8, a1.data());
        kt.matvec(m->lstm_hh[l], 4 * H, H, hq.q.data(), a2.data());
        for (int o = 0; o < 4 * H; ++o) {
            gates[o] = (float) a1[o] * m->lstm_ih_s[l][o] * xs + (float) a2[o] * m->lstm_hh_s[l][o] * hq.s +
                       m->lstm_b[l][o];
        }
        float * c = st.c[l].data();
        float * h = st.h[l].data();
        for (int j = 0; j < H; ++j) {
            const float ig = sigm(gates[j]);
            const float fg = sigm(gates[H + j]);
            const float gg = std::tanh(gates[2 * H + j]);
            const float og = sigm(gates[3 * H + j]);
            c[j] = fg * c[j] + ig * gg;
            h[j] = og * std::tanh(c[j]);
        }
    }
    const int J = m->hp.joint_hidden;
    quant_vec(st.h[1].data(), H, hq);
    kt.matvec(m->proj_q, J, H, hq.q.data(), a1.data());
    for (int o = 0; o < J; ++o) {
        st.out[o] = (float) a1[o] * m->proj_s[o] * hq.s + m->proj_b[o];
    }
}

}  // namespace

int pk::tdt_decode(pk_model * m, const float * encp, int T, std::vector<pk_token> & out, pk_cancel_cb cancel,
                   void * cu) {
    const auto & hp = m->hp;
    const int H = hp.pred_hidden, J = hp.joint_hidden, NR = m->head_rows;
    const int nv = hp.vocab + 1;  // tokens + blank
    const int nd = (int) hp.durations.size();
    dec_state st;
    for (int l = 0; l < 2; ++l) {
        st.h[l].assign(H, 0.0f);
        st.c[l].assign(H, 0.0f);
    }
    st.out.assign(J, 0.0f);
    std::vector<int32_t> a1(4 * H), a2(4 * H), acc(NR);
    std::vector<float> gates(4 * H), z(J);
    qvec xq, hq, zq;
    lstm_step(m, hp.blank_id, st, a1, a2, gates, xq, hq);
    out.clear();
    int t = 0, symbols = 0, steps = 0;
    auto & P = *m->pool;
    const auto & kt = *m->kt;
    while (t < T) {
        if (cancel && (++steps & 15) == 0 && cancel(cu)) return -1;
        const float * e = encp + (size_t) t * J;
        for (int j = 0; j < J; ++j) {
            const float v = e[j] + st.out[j];
            z[j] = v > 0.0f ? v : 0.0f;
        }
        quant_vec(z.data(), J, zq);
        const int8_t * zq8 = zq.q.data();
        if (P.size() > 1) {
            P.parallel_for(NR, 1056, [&](int b, int en) {
                kt.matvec(m->head_q + (size_t) b * J, en - b, J, zq8, acc.data() + b);
            });
        } else {
            kt.matvec(m->head_q, NR, J, zq8, acc.data());
        }
        int best = 0;
        float bv = -INFINITY;
        for (int o = 0; o < nv; ++o) {
            const float v = (float) acc[o] * m->head_s[o] * zq.s + m->head_b[o];
            if (v > bv) {
                bv = v;
                best = o;
            }
        }
        int bd = 0;
        float bdv = -INFINITY;
        for (int d = 0; d < nd; ++d) {
            const int o = nv + d;
            const float v = (float) acc[o] * m->head_s[o] * zq.s + m->head_b[o];
            if (v > bdv) {
                bdv = v;
                bd = d;
            }
        }
        int dur = hp.durations[bd];
        if (best == hp.blank_id) {
            if (dur == 0) dur = 1;
            symbols = 0;
        } else {
            out.push_back(pk_token{ best, t, dur });
            lstm_step(m, best, st, a1, a2, gates, xq, hq);
            if (dur == 0) {
                if (++symbols >= hp.max_symbols) {
                    dur = 1;
                    symbols = 0;
                }
            } else {
                symbols = 0;
            }
        }
        t += dur;
    }
    return 0;
}

std::string pk::detokenize(const pk_model * m, const pk_token * toks, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        const int id = toks[i].id;
        if (id < 0 || id >= (int) m->pieces.size() || m->is_control[id]) continue;
        s += m->pieces[id];
    }
    size_t b = 0;
    while (b < s.size() && s[b] == ' ') ++b;
    size_t e = s.size();
    while (e > b && s[e - 1] == ' ') --e;
    return s.substr(b, e - b);
}

// ------------------------------------------------------------------ API

static int pk_max_window_samples(const pk_model * m) {
    const char * env = std::getenv("PK_MAX_WINDOW_S");
    double s = env && *env ? std::atof(env) : 30.0;
    if (s < 4.0) s = 4.0;
    return (int) (s * m->hp.sample_rate);
}

// choose a cut in [lo, hi) at the quietest 100 ms
static size_t quiet_cut(const float * pcm, size_t lo, size_t hi) {
    const size_t win = 1600;
    size_t best = hi;
    double be = 1e300;
    for (size_t s = lo; s + win <= hi; s += 160) {
        double e = 0.0;
        for (size_t i = 0; i < win; ++i) e += (double) pcm[s + i] * pcm[s + i];
        if (e < be) {
            be = e;
            best = s + win / 2;
        }
    }
    return best;
}

extern "C" int pk_transcribe(pk_model * m, const float * pcm, size_t n, pk_cancel_cb cancel, void * cu,
                             pk_result * out) {
    if (!m || !out || (!pcm && n)) return PK_E_INVALID;
    std::memset(out, 0, sizeof(*out));
    std::vector<pk_token> all;
    const size_t maxw = (size_t) pk_max_window_samples(m);
    size_t pos = 0;
    while (pos < n) {
        size_t end = n;
        if (n - pos > maxw) {
            end = quiet_cut(pcm, pos + maxw - 5 * (size_t) m->hp.sample_rate, pos + maxw);
        }
        auto t0 = clk::now();
        std::vector<float> mel;
        int nm = 0;
        if (!compute_mel(m, pcm + pos, end - pos, mel, nm)) {
            pos = end;
            continue;
        }
        out->t_mel_ms += ms_since(t0);
        t0 = clk::now();
        std::vector<float> encp;
        const int T = encode(m, mel.data(), nm, encp, cancel, cu);
        if (T < 0) return PK_E_CANCELLED;
        out->t_encoder_ms += ms_since(t0);
        t0 = clk::now();
        std::vector<pk_token> toks;
        if (tdt_decode(m, encp.data(), T, toks, cancel, cu) < 0) return PK_E_CANCELLED;
        out->t_decoder_ms += ms_since(t0);
        // encoder frames are 80 ms; window starts are rounded to a frame
        const int base = (int) std::lround((double) pos / (m->hp.hop_length * 8));
        for (auto & tk : toks) {
            tk.frame += base;
            all.push_back(tk);
        }
        out->n_frames = base + T;
        pos = end;
    }
    const std::string text = detokenize(m, all.data(), all.size());
    out->text = (char *) std::malloc(text.size() + 1);
    out->tokens = (pk_token *) std::malloc(sizeof(pk_token) * std::max<size_t>(1, all.size()));
    if (!out->text || !out->tokens) return PK_E_NOMEM;
    std::memcpy(out->text, text.c_str(), text.size() + 1);
    if (!all.empty()) std::memcpy(out->tokens, all.data(), sizeof(pk_token) * all.size());
    out->n_tokens = all.size();
    return PK_OK;
}

extern "C" void pk_result_free(pk_result * r) {
    if (!r) return;
    std::free(r->text);
    std::free(r->tokens);
    r->text = nullptr;
    r->tokens = nullptr;
    r->n_tokens = 0;
}

extern "C" int pk_result_words(const pk_model * m, const pk_result * r, int * ws, int * we, size_t * io_n) {
    if (!m || !r || !io_n) return PK_E_INVALID;
    const size_t cap = *io_n;
    size_t n = 0;
    const int fms = m->hp.hop_length * 8 * 1000 / m->hp.sample_rate;  // 80 ms
    int cur_start = -1, cur_end = -1;
    bool has_chars = false;
    auto flush = [&]() {
        if (cur_start >= 0 && has_chars) {
            if (n < cap) {
                if (ws) ws[n] = cur_start;
                if (we) we[n] = cur_end;
            }
            ++n;
        }
        cur_start = cur_end = -1;
        has_chars = false;
    };
    for (size_t i = 0; i < r->n_tokens; ++i) {
        const pk_token & t = r->tokens[i];
        if (t.id < 0 || t.id >= (int) m->pieces.size() || m->is_control[t.id]) continue;
        const std::string & p = m->pieces[t.id];
        // a word starts at a piece with a leading space, unless the current
        // word has no characters yet (a lone space piece)
        if (cur_start < 0 || (m->word_start[t.id] && has_chars)) {
            flush();
            cur_start = t.frame * fms;
        }
        cur_end = (t.frame + std::max(1, t.duration)) * fms;
        if (p.find_first_not_of(' ') != std::string::npos) has_chars = true;
    }
    flush();
    *io_n = std::min(n, cap);
    return PK_OK;
}
