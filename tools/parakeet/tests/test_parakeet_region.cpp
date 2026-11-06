// tools/parakeet/tests/test_parakeet_region.cpp - a parakeet-tdt GGUF stored
// inside a larger file (an uncompressed APK asset), reached through a region
// file (tools/file-region). The region loads mapped and read into memory,
// pk_gguf_is_parakeet sees it, and it gives the same tokens as the plain file.
//
//   test-parakeet-region <phonon-2.gguf> [<16 kHz mono s16 WAV>]
//
// Without a WAV it transcribes a fixed synthetic signal; the tokens still
// have to match exactly.
#include "parakeet.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++failures; } } while (0)

static std::string read_file(const std::string & p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static std::vector<float> wav_pcm(const std::string & path) {
    const std::string b = read_file(path);
    std::vector<float> pcm;
    size_t at = 12;
    while (at + 8 <= b.size()) {
        uint32_t n = 0;
        std::memcpy(&n, b.data() + at + 4, 4);
        if (b.compare(at, 4, "data") == 0) {
            const size_t end = std::min(b.size(), at + 8 + (size_t) n);
            for (size_t i = at + 8; i + 1 < end; i += 2) {
                int16_t s = 0;
                std::memcpy(&s, b.data() + i, 2);
                pcm.push_back(s / 32768.0f);
            }
            break;
        }
        at += 8 + n + (n & 1);
    }
    return pcm;
}

struct result {
    std::string text;
    std::vector<int32_t> ids;
};

static bool transcribe(const std::string & path, int use_mmap, const std::vector<float> & pcm, result & out) {
    pk_params p = pk_default_params();
    p.use_mmap = use_mmap;
    char err[512] = { 0 };
    pk_model * m = pk_model_load(path.c_str(), p, err, sizeof(err));
    if (!m) {
        std::fprintf(stderr, "load %s (mmap %d): %s\n", path.c_str(), use_mmap, err);
        return false;
    }
    pk_result r {};
    const int rc = pk_transcribe(m, pcm.data(), pcm.size(), nullptr, nullptr, &r);
    if (rc == PK_OK) {
        out.text = r.text ? r.text : "";
        for (size_t i = 0; i < r.n_tokens; ++i) out.ids.push_back(r.tokens[i].id);
    }
    pk_result_free(&r);
    pk_model_free(m);
    return rc == PK_OK;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test-parakeet-region <phonon-2.gguf> [<wav>]\n");
        return 2;
    }
#if defined(_WIN32)
    std::printf("test-parakeet-region: region files are mapped with POSIX mmap, skipping\n");
    return 77;
#endif
    const std::string gguf = argv[1];
    std::vector<float> pcm = argc > 2 ? wav_pcm(argv[2]) : std::vector<float>();
    if (pcm.empty()) {
        pcm.resize(16000 * 3);
        for (size_t i = 0; i < pcm.size(); ++i) {
            const double t = i / 16000.0;
            pcm[i] = (float) (0.3 * std::sin(2 * 3.14159265358979 * (220 + 180 * std::sin(3 * t)) * t) * std::sin(3.14159265358979 * t / 3));
        }
    }

    const fs::path dir = fs::temp_directory_path() / ("test-parakeet-region-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const std::string bytes = read_file(gguf);
    const size_t offset = 2 * 4096 + 32;
    const fs::path outer = fs::absolute(dir / "app.apk");
    {
        std::ofstream o(outer, std::ios::binary);
        o << std::string(offset, 'z') << bytes << std::string(77, 'q');
    }
    const fs::path link = dir / "phonon-2.gguf";
    std::ofstream(link) << "eliza-file-region v1\noffset=" << offset << "\nlength=" << bytes.size()
                        << "\npath=" << outer.string() << "\n";

    CHECK(pk_gguf_is_parakeet(link.c_str()) == 1, "region not recognised as parakeet-tdt");
    CHECK(pk_gguf_is_parakeet(outer.c_str()) == 0, "container recognised as parakeet-tdt");

    result want;
    CHECK(transcribe(gguf, 1, pcm, want), "plain transcription");
    for (int use_mmap : { 1, 0 }) {
        result got;
        CHECK(transcribe(link.string(), use_mmap, pcm, got) && got.ids == want.ids && got.text == want.text,
              "region (mmap %d) gave '%s', plain gave '%s'", use_mmap, got.text.c_str(), want.text.c_str());
    }
    std::printf("test-parakeet-region: transcript '%s' (%zu tokens)\n", want.text.c_str(), want.ids.size());
    fs::remove_all(dir);
    if (failures) {
        std::fprintf(stderr, "test-parakeet-region: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-parakeet-region: OK\n");
    return 0;
}
