// SPDX-License-Identifier: Apache-2.0
// Reader for misaki-en-us.lex (written by convert_misaki_lexicon.py).
#include "misaki-internal.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#    define MISAKI_NO_MMAP 1
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace misaki {
namespace {

constexpr size_t kHeader = 128;
constexpr size_t kEntry = 12;
constexpr size_t kVariant = 8;

uint32_t rd32(const uint8_t * p) { return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24); }
uint16_t rd16(const uint8_t * p) { return (uint16_t) (p[0] | (p[1] << 8)); }

}  // namespace

std::unique_ptr<LexFile> LexFile::open(const std::string & path, std::string & err) {
    std::unique_ptr<LexFile> lf(new LexFile());
#if !defined(MISAKI_NO_MMAP)
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        err = "cannot open lexicon " + path + ": " + std::strerror(errno);
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t) kHeader) {
        ::close(fd);
        err = "lexicon " + path + " is too small";
        return nullptr;
    }
    void * p = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        err = "cannot mmap lexicon " + path + ": " + std::strerror(errno);
        return nullptr;
    }
    lf->base_ = (const uint8_t *) p;
    lf->size_ = (size_t) st.st_size;
    lf->mapped_ = true;
#else
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open lexicon " + path;
        return nullptr;
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < (long) kHeader) {
        std::fclose(f);
        err = "lexicon " + path + " is too small";
        return nullptr;
    }
    lf->owned_.resize((size_t) n);
    size_t got = std::fread(lf->owned_.data(), 1, (size_t) n, f);
    std::fclose(f);
    if (got != (size_t) n) {
        err = "short read on lexicon " + path;
        return nullptr;
    }
    lf->base_ = lf->owned_.data();
    lf->size_ = (size_t) n;
#endif
    const uint8_t * h = lf->base_;
    if (std::memcmp(h, "MISAKILX", 8) != 0) {
        err = "lexicon " + path + " has a bad magic (not a misaki-en-us.lex)";
        return nullptr;
    }
    uint32_t version = rd32(h + 8);
    if (version != 1) {
        err = "lexicon " + path + " has format version " + std::to_string(version) + ", expected 1";
        return nullptr;
    }
    lf->n_gold_ = rd32(h + 16);
    lf->n_silver_ = rd32(h + 20);
    lf->n_var_ = rd32(h + 24);
    lf->n_str_ = rd32(h + 28);
    uint32_t off_gold = rd32(h + 96), off_silver = rd32(h + 100), off_var = rd32(h + 104), off_str = rd32(h + 108);
    uint32_t total = rd32(h + 112);
    if (total != lf->size_ || (uint64_t) off_gold + (uint64_t) lf->n_gold_ * kEntry > off_silver ||
        (uint64_t) off_silver + (uint64_t) lf->n_silver_ * kEntry > off_var ||
        (uint64_t) off_var + (uint64_t) lf->n_var_ * kVariant > off_str || (uint64_t) off_str + lf->n_str_ > total) {
        err = "lexicon " + path + " is truncated or inconsistent";
        return nullptr;
    }
    lf->gold_ = h + off_gold;
    lf->silver_ = h + off_silver;
    lf->variants_ = h + off_var;
    lf->strings_ = h + off_str;
    // Validate every reference once so lookups need no bounds checks.
    for (int t = 0; t < 2; t++) {
        const uint8_t * e = t ? lf->silver_ : lf->gold_;
        uint32_t n = t ? lf->n_silver_ : lf->n_gold_;
        for (uint32_t i = 0; i < n; i++, e += kEntry) {
            uint32_t koff = rd32(e), klen = rd16(e + 4), nv = rd16(e + 6), first = rd32(e + 8);
            if ((uint64_t) koff + klen > lf->n_str_ || nv == 0 || (uint64_t) first + nv > lf->n_var_) {
                err = "lexicon " + path + " has a bad entry";
                return nullptr;
            }
        }
    }
    for (uint32_t i = 0; i < lf->n_var_; i++) {
        const uint8_t * v = lf->variants_ + (size_t) i * kVariant;
        if (v[0] >= TAG_COUNT || (uint64_t) rd32(v + 4) + rd16(v + 2) > lf->n_str_) {
            err = "lexicon " + path + " has a bad variant";
            return nullptr;
        }
    }
    return lf;
}

LexFile::~LexFile() {
#if !defined(MISAKI_NO_MMAP)
    if (mapped_ && base_) munmap(const_cast<uint8_t *>(base_), size_);
#endif
}

long LexFile::find(const uint8_t * entries, uint32_t n, const std::string & key) const {
    long lo = 0, hi = (long) n;
    while (lo < hi) {
        long mid = (lo + hi) / 2;
        const uint8_t * e = entries + (size_t) mid * kEntry;
        uint32_t koff = rd32(e), klen = rd16(e + 4);
        int c = std::memcmp(strings_ + koff, key.data(), std::min<size_t>(klen, key.size()));
        if (c == 0) c = (klen < key.size()) ? -1 : (klen > key.size() ? 1 : 0);
        if (c == 0) return mid;
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

LexValue LexFile::get(const uint8_t * entries, uint32_t n, const u32 & key) const {
    LexValue out;
    long i = find(entries, n, to_utf8(key));
    if (i < 0) return out;
    out.found = true;
    const uint8_t * e = entries + (size_t) i * kEntry;
    uint32_t nv = rd16(e + 6), first = rd32(e + 8);
    for (uint32_t k = 0; k < nv; k++) {
        const uint8_t * v = variants_ + (size_t) (first + k) * kVariant;
        uint8_t tag = v[0];
        std::optional<u32> val;
        if (!v[1]) val = from_utf8((const char *) strings_ + rd32(v + 4), rd16(v + 2));
        if (tag == TAG_PLAIN) {
            out.plain = val;
        } else {
            out.is_dict = true;
            out.has_tag[tag] = true;
            out.by_tag[tag] = val;
        }
    }
    return out;
}

LexValue LexFile::gold(const u32 & key) const { return get(gold_, n_gold_, key); }
LexValue LexFile::silver(const u32 & key) const { return get(silver_, n_silver_, key); }
bool LexFile::in_gold(const u32 & key) const { return find(gold_, n_gold_, to_utf8(key)) >= 0; }
bool LexFile::in_silver(const u32 & key) const { return find(silver_, n_silver_, to_utf8(key)) >= 0; }

}  // namespace misaki
