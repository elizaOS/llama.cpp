// SPDX-License-Identifier: MIT
// tools/file-region/eliza-file-region.h
//
// A region file stands in for a model that is stored inside another file,
// such as an uncompressed asset in an Android APK. It is a small text file:
//
//   eliza-file-region v1
//   offset=<first byte of the model in the containing file>
//   length=<model size in bytes>
//   path=<absolute path of the containing file>
//
// The voice loaders (Paradee, Parakeet, the misaki lexicon) resolve every model
// path with resolve(): a region file gives the named bytes of the containing
// file, and any other file gives the whole file. mapping maps a region
// read-only, so the model pages stay clean, file-backed and reclaimable, the
// same as a mapping of a separate model file.
//
// The offset must be a multiple of kAlignment, so the tensors in a mapped
// region keep the alignment they have in a heap copy.
//
// Header only; each loader includes it privately.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace eliza_file_region {

constexpr const char * kMagic = "eliza-file-region v1";
// A region file is a few hundred bytes; anything larger is a model.
constexpr size_t kMaxRegionFile = 4096;
constexpr uint64_t kAlignment = 16;

struct region {
    std::string path;      // file that holds the bytes
    uint64_t offset = 0;   // first byte in `path`
    uint64_t length = 0;   // bytes
    bool indirect = false; // true when resolved from a region file
};

namespace detail {

inline bool file_size(const std::string & path, uint64_t & size) {
#if defined(_WIN32)
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const bool ok = _fseeki64(f, 0, SEEK_END) == 0;
    const long long end = ok ? _ftelli64(f) : -1;
    std::fclose(f);
    if (end < 0) return false;
    size = (uint64_t) end;
    return true;
#else
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) return false;
    size = (uint64_t) st.st_size;
    return true;
#endif
}

inline bool parse_u64(const std::string & text, uint64_t & out) {
    if (text.empty() || text.size() > 20) return false;
    uint64_t v = 0;
    for (char c : text) {
        if (c < '0' || c > '9') return false;
        const uint64_t d = (uint64_t) (c - '0');
        if (v > (UINT64_MAX - d) / 10) return false;
        v = v * 10 + d;
    }
    out = v;
    return true;
}

} // namespace detail

// Parses region-file text. False with `err` when it is not a valid region file.
inline bool parse(const std::string & text, region & out, std::string & err) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        lines.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    if (lines.empty() || lines[0] != kMagic) {
        err = "not a region file";
        return false;
    }
    bool has_offset = false, has_length = false, has_path = false;
    region r;
    for (size_t i = 1; i < lines.size(); ++i) {
        const std::string & line = lines[i];
        if (line.empty()) continue;
        const size_t eq = line.find('=');
        const std::string key = eq == std::string::npos ? line : line.substr(0, eq);
        const std::string value = eq == std::string::npos ? "" : line.substr(eq + 1);
        bool ok = eq != std::string::npos;
        if (key == "offset" && !has_offset) {
            ok = ok && detail::parse_u64(value, r.offset);
            has_offset = true;
        } else if (key == "length" && !has_length) {
            ok = ok && detail::parse_u64(value, r.length);
            has_length = true;
        } else if (key == "path" && !has_path) {
            ok = ok && !value.empty() && value[0] == '/';
            r.path = value;
            has_path = true;
        } else {
            ok = false;
        }
        if (!ok) {
            err = "invalid region file line '" + line + "'";
            return false;
        }
    }
    if (!has_offset || !has_length || !has_path || r.length == 0) {
        err = "region file needs offset, length and an absolute path";
        return false;
    }
    if (r.offset % kAlignment != 0) {
        err = "region offset " + std::to_string(r.offset) + " is not a multiple of " + std::to_string(kAlignment);
        return false;
    }
    r.indirect = true;
    out = r;
    return true;
}

// Resolves a model path. A region file gives its region of the containing
// file, checked against that file's size; any other file gives the whole file.
inline bool resolve(const std::string & path, region & out, std::string & err) {
    uint64_t size = 0;
    if (!detail::file_size(path, size)) {
        err = "cannot open '" + path + "'";
        return false;
    }
    std::string head;
    if (size > 0 && size <= kMaxRegionFile) {
        FILE * f = std::fopen(path.c_str(), "rb");
        if (!f) {
            err = "cannot open '" + path + "'";
            return false;
        }
        head.resize((size_t) size);
        const size_t got = std::fread(&head[0], 1, head.size(), f);
        std::fclose(f);
        head.resize(got);
    }
    const size_t magic = std::strlen(kMagic);
    if (head.size() > magic && head.compare(0, magic, kMagic) == 0 && head[magic] == '\n') {
        region r;
        if (!parse(head, r, err)) {
            err = "'" + path + "': " + err;
            return false;
        }
        uint64_t outer = 0;
        if (!detail::file_size(r.path, outer)) {
            err = "'" + path + "': cannot open '" + r.path + "'";
            return false;
        }
        if (r.offset > outer || r.length > outer - r.offset) {
            err = "'" + path + "': region runs past the end of '" + r.path + "'";
            return false;
        }
        out = r;
        return true;
    }
    out = region{ path, 0, size, false };
    return true;
}

// Reads a region into memory (the no-mmap paths).
inline bool read(const region & r, std::vector<uint8_t> & out, std::string & err) {
    FILE * f = std::fopen(r.path.c_str(), "rb");
    if (!f) {
        err = "cannot open '" + r.path + "'";
        return false;
    }
#if defined(_WIN32)
    const bool seek = _fseeki64(f, (long long) r.offset, SEEK_SET) == 0;
#else
    const bool seek = fseeko(f, (off_t) r.offset, SEEK_SET) == 0;
#endif
    out.resize((size_t) r.length);
    const bool ok = seek && std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    if (!ok) err = "short read on '" + r.path + "'";
    return ok;
}

// Read-only private mapping of a region. The mapping starts at the page that
// holds the region's first byte; data() points at that byte.
class mapping {
public:
    mapping() = default;
    mapping(const mapping &) = delete;
    mapping & operator=(const mapping &) = delete;
    ~mapping() { reset(); }

    bool open(const region & r, std::string & err) {
        reset();
#if defined(_WIN32)
        (void) r;
        err = "memory mapping is not supported on this platform";
        return false;
#else
        if (r.length == 0) {
            err = "cannot map an empty file '" + r.path + "'";
            return false;
        }
        const int fd = ::open(r.path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            err = "cannot open '" + r.path + "': " + std::strerror(errno);
            return false;
        }
        const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
        const uint64_t base = r.offset - r.offset % page;
        const size_t delta = (size_t) (r.offset - base);
        const size_t bytes = (size_t) r.length + delta;
        void * p = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, (off_t) base);
        ::close(fd);
        if (p == MAP_FAILED) {
            err = "mmap failed for '" + r.path + "': " + std::strerror(errno);
            return false;
        }
        base_ = p;
        mapped_ = bytes;
        data_ = (const uint8_t *) p + delta;
        size_ = (size_t) r.length;
        return true;
#endif
    }

    void reset() {
#if !defined(_WIN32)
        if (base_) munmap(base_, mapped_);
#endif
        base_ = nullptr;
        mapped_ = 0;
        data_ = nullptr;
        size_ = 0;
    }

    const uint8_t * data() const { return data_; }
    size_t size() const { return size_; }

private:
    void * base_ = nullptr;
    size_t mapped_ = 0;
    const uint8_t * data_ = nullptr;
    size_t size_ = 0;
};

} // namespace eliza_file_region
