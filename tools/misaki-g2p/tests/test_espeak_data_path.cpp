// SPDX-License-Identifier: Apache-2.0
// eSpeak NG data-path handling in the shared libespeak-ng owner
// (tools/espeak-shared/eliza-espeak-shared.h), through misaki_g2p_open:
//
//   - a missing directory is an error, not exit(1) and not a silent load of
//     libespeak-ng's compiled-in default data;
//   - a path longer than the linked libespeak-ng's N_PATH_HOME buffer is an
//     error naming that limit, or, when the library was built with a larger
//     N_PATH_HOME (the Android builds), works and phonemizes exactly like the
//     short path;
//   - a path whose espeak-ng-data child no longer fits (libespeak-ng would
//     fall back to the parent) is caught the same way;
//   - after any of these failures the short path still opens and phonemizes.
//
// usage: test-espeak-data-path MINI_LEX ESPEAK_NG_DATA_DIR
// ESPEAK_NG_DATA_DIR is the espeak-ng-data directory itself.
// exit 77 (skip) when misaki-g2p was built without libespeak-ng.
#include "misaki-g2p.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

static int g_fail = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);            \
            std::fprintf(stderr, "\n");                   \
            g_fail++;                                     \
        }                                                 \
    } while (0)

struct Opened {
    misaki_g2p * g = nullptr;
    std::string  err;
};

static Opened open_with(const std::string & lex, const std::string & dir) {
    Opened o;
    char * err = nullptr;
    o.g = misaki_g2p_open(lex.c_str(), dir.c_str(), &err);
    if (err) o.err = err;
    misaki_g2p_free(err);
    return o;
}

static std::string phonemize(misaki_g2p * g, const std::string & text) {
    char * out = nullptr;
    char * err = nullptr;
    std::string s;
    if (misaki_g2p_phonemize(g, text.data(), text.size(), &out, &err) == 0 && out) s = out;
    else s = std::string("ERROR: ") + (err ? err : "?");
    misaki_g2p_free(out);
    misaki_g2p_free(err);
    return s;
}

// A directory `root`/aaaa.../aaaa... whose own path is `len` bytes long.
static std::string long_dir(const fs::path & root, size_t len) {
    std::string p = root.string();
    while (p.size() < len) {
        const size_t room = len - p.size() - 1;  // minus the separator
        const size_t seg  = room > 60 ? 60 : room;
        if (seg == 0) {
            p += "/a";  // cannot land exactly; one byte over is fine for callers
            break;
        }
        p += "/" + std::string(seg, 'a');
    }
    std::error_code ec;
    fs::create_directories(p, ec);
    return p;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MINI_LEX ESPEAK_NG_DATA_DIR\n", argv[0]);
        return 2;
    }
    if (!misaki_g2p_espeak_linked()) {
        std::printf("SKIP: misaki-g2p built without libespeak-ng\n");
        return 77;
    }
    const std::string lex  = argv[1];
    const std::string data = fs::absolute(argv[2]).lexically_normal().string();
    if (!fs::is_regular_file(fs::path(data) / "phontab")) {
        std::printf("SKIP: no eSpeak NG data at %s\n", data.c_str());
        return 77;
    }
    const std::string oov = "Zyx quorbled the flimbit, glarf!";

    // Baseline on the short path.
    Opened base = open_with(lex, data);
    if (!base.g) {
        std::printf("SKIP: eSpeak NG data not usable: %s\n", base.err.c_str());
        return 77;
    }
    const std::string want = phonemize(base.g, oov);
    misaki_g2p_close(base.g);
    CHECK(!want.empty() && want.rfind("ERROR", 0) != 0, "baseline phonemize: '%s'", want.c_str());

    std::error_code ec;
    fs::path root = fs::temp_directory_path() / ("espeak-path-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    // 1. Missing directory: an error, the process keeps running.
    {
        Opened o = open_with(lex, (root / "missing").string());
        CHECK(!o.g, "a missing data directory opened");
        CHECK(o.err.find("not found") != std::string::npos, "missing dir error: '%s'", o.err.c_str());
        misaki_g2p_close(o.g);
    }

    // 2./3. Long parents holding an espeak-ng-data link to the real data:
    // 150 bytes (the parent fits a 160-byte buffer, parent/espeak-ng-data
    // does not) and 400 bytes (neither fits).
    for (size_t len : { size_t(150), size_t(400) }) {
        const std::string parent = long_dir(root, len);
        fs::create_directory_symlink(data, fs::path(parent) / "espeak-ng-data", ec);
        CHECK(!ec, "symlink under %zu-byte parent: %s", parent.size(), ec.message().c_str());
        Opened o = open_with(lex, parent);
        if (o.g) {
            const std::string got = phonemize(o.g, oov);
            CHECK(got == want, "%zu-byte path phonemized '%s', short path '%s'", parent.size(), got.c_str(),
                  want.c_str());
            std::printf("%zu-byte data path: accepted (libespeak-ng N_PATH_HOME is larger)\n", parent.size());
        } else {
            CHECK(o.err.find("N_PATH_HOME") != std::string::npos, "%zu-byte path error: '%s'", parent.size(),
                  o.err.c_str());
            std::printf("%zu-byte data path: rejected: %s\n", parent.size(), o.err.c_str());
        }
        misaki_g2p_close(o.g);
    }

    // 4. The short path still works after the failures above.
    {
        Opened o = open_with(lex, data);
        CHECK(o.g, "short path after failures: '%s'", o.err.c_str());
        if (o.g) {
            const std::string got = phonemize(o.g, oov);
            CHECK(got == want, "short path after failures phonemized '%s', want '%s'", got.c_str(), want.c_str());
        }
        misaki_g2p_close(o.g);
    }

    fs::remove_all(root, ec);
    if (g_fail) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
