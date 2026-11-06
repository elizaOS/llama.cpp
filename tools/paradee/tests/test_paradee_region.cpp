// SPDX-License-Identifier: MIT
//
// test-paradee-region — region files (tools/file-region): a model stored
// inside another file, as an uncompressed APK asset is.
//
//   always:        region-file parsing, resolution and mapping at an offset
//                  that is not page aligned (weight-free)
//   PARADEE_GGUF:  the model loads through a region file inside a larger
//                  container, mapped and with PARADEE_USE_MMAP=0, and
//                  synthesizes the same samples as the plain file

#include "eliza-file-region.h"
#include "paradee.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
int main() {
    std::printf("test-paradee-region: region files are mapped with POSIX mmap, skipping\n");
    return 77;
}
#else
using namespace eliza_paradee;
namespace fs = std::filesystem;
namespace fr = eliza_file_region;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++failures; } } while (0)

static void write_file(const fs::path & p, const std::string & bytes) {
    std::ofstream(p, std::ios::binary).write(bytes.data(), (std::streamsize) bytes.size());
}

static std::string read_file(const fs::path & p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static std::string region_text(uint64_t offset, uint64_t length, const fs::path & path) {
    return std::string(fr::kMagic) + "\noffset=" + std::to_string(offset) + "\nlength=" + std::to_string(length) +
           "\npath=" + path.string() + "\n";
}

static std::string junk(size_t n, unsigned seed) {
    std::string s(n, '\0');
    for (size_t i = 0; i < n; ++i) s[i] = (char) ((i * 131u + seed) & 0xff);
    return s;
}

static void test_regions(const fs::path & dir) {
    std::string err;
    // A payload 16-aligned but not page aligned, as in an APK.
    const std::string payload = junk(70000, 7);
    const size_t offset = 3 * 4096 + 48;
    const fs::path outer = fs::absolute(dir / "container.bin");
    write_file(outer, junk(offset, 1) + payload + junk(1000, 3));
    const fs::path link = dir / "model.gguf";
    write_file(link, region_text(offset, payload.size(), outer));

    fr::region r;
    CHECK(fr::resolve(link.string(), r, err) && r.indirect && r.path == outer.string() && r.offset == offset &&
          r.length == payload.size(), "resolve region: %s", err.c_str());
    fr::mapping map;
    CHECK(map.open(r, err) && map.size() == payload.size() &&
          std::memcmp(map.data(), payload.data(), payload.size()) == 0, "mapped region bytes: %s", err.c_str());
    std::vector<uint8_t> heap;
    CHECK(fr::read(r, heap, err) && std::string(heap.begin(), heap.end()) == payload, "read region bytes");

    // Any other file is the whole file, small files included.
    const fs::path plain = dir / "plain.bin";
    write_file(plain, "GGUF small");
    CHECK(fr::resolve(plain.string(), r, err) && !r.indirect && r.offset == 0 && r.length == 10, "plain file");
    CHECK(!fr::resolve((dir / "missing").string(), r, err), "missing file accepted");

    const std::string good = region_text(offset, payload.size(), outer);
    const std::vector<std::string> bad = {
        region_text(offset + 8, payload.size(), outer),                // not 16-aligned
        region_text(offset, payload.size() + 2000, outer),            // past the end
        region_text(offset, 0, outer),                                // empty
        region_text(offset, payload.size(), "relative/container.bin"),
        region_text(offset, payload.size(), dir / "missing.bin"),
        good + "offset=0\n",                                          // repeated key
        good + "extra=1\n",                                           // unknown key
        std::string(fr::kMagic) + "\noffset=-1\nlength=1\npath=" + outer.string() + "\n",
        std::string(fr::kMagic) + "\nlength=1\npath=" + outer.string() + "\n",
    };
    for (const std::string & text : bad) {
        write_file(link, text);
        err.clear();
        CHECK(!fr::resolve(link.string(), r, err) && !err.empty(), "bad region file accepted: %s", text.c_str());
    }
}

static bool synth(const paradee_model * m, std::vector<float> & pcm) {
    std::vector<int32_t> ids;
    std::string err;
    for (const char * phonemes : { "həlˈO wˈɜɹld.", "ˈola mˈundo." }) {
        if (paradee_phonemes_to_ids(m, phonemes, ids, err) == PARADEE_OK) break;
    }
    paradee_synth_options opts;
    paradee_audio audio;
    if (ids.empty() || paradee_synthesize(m, ids, opts, audio, err) != PARADEE_OK) {
        std::fprintf(stderr, "synthesis failed: %s\n", err.c_str());
        return false;
    }
    pcm = audio.samples;
    return !pcm.empty();
}

static void test_model(const fs::path & dir, const std::string & gguf) {
    std::string err;
    const std::string bytes = read_file(gguf);
    const size_t offset = 4096 + 16;
    const fs::path outer = fs::absolute(dir / "app.apk");
    write_file(outer, junk(offset, 5) + bytes + junk(333, 9));
    const fs::path link = dir / "paradee.gguf";
    write_file(link, region_text(offset, bytes.size(), outer));

    auto plain = paradee_load_model(gguf, err);
    if (!plain) { CHECK(false, "plain load: %s", err.c_str()); return; }
    std::vector<float> want;
    CHECK(synth(plain.get(), want), "plain synthesis");

    for (const char * use_mmap : { "1", "0" }) {
        setenv("PARADEE_USE_MMAP", use_mmap, 1);
        auto region = paradee_load_model(link.string(), err);
        CHECK(region != nullptr, "region load (PARADEE_USE_MMAP=%s): %s", use_mmap, err.c_str());
        std::vector<float> got;
        CHECK(region && synth(region.get(), got) && got == want,
              "region synthesis differs (PARADEE_USE_MMAP=%s): %zu vs %zu samples", use_mmap, got.size(), want.size());
    }
    unsetenv("PARADEE_USE_MMAP");

    // A region that does not cover a GGUF fails to load.
    write_file(link, region_text(0, bytes.size(), outer));
    CHECK(!paradee_load_model(link.string(), err) && !err.empty(), "region over junk loaded");
}

int main() {
    const fs::path dir = fs::temp_directory_path() /
        ("test-paradee-region-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    test_regions(dir);
    const char * gguf = std::getenv("PARADEE_GGUF");
    if (gguf && *gguf) {
        test_model(dir, gguf);
    } else {
        std::printf("test-paradee-region: PARADEE_GGUF not set, model checks skipped\n");
    }
    fs::remove_all(dir);
    if (failures) {
        std::fprintf(stderr, "test-paradee-region: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test-paradee-region: OK\n");
    return 0;
}
#endif
