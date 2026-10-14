// tools/parakeet/tools/pk-wav.h - minimal RIFF/WAVE reader for the CLI tools
// (PCM16 / PCM32 / float32, any channel count, mixed to mono) plus a
// windowed-sinc resampler to 16 kHz.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace pkwav {

constexpr double kPi = 3.14159265358979323846;

inline bool read_wav(const std::string & path, std::vector<float> & out, int & sr, std::string & err) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> d;
    uint8_t buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) d.insert(d.end(), buf, buf + n);
    std::fclose(f);
    auto u16 = [&](size_t o) { return (uint32_t) d[o] | ((uint32_t) d[o + 1] << 8); };
    auto u32 = [&](size_t o) { return u16(o) | (u16(o + 2) << 16); };
    if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4)) {
        err = path + ": not a RIFF/WAVE file";
        return false;
    }
    int fmt = 0, ch = 0, bits = 0;
    size_t pos = 12, data_off = 0, data_len = 0;
    while (pos + 8 <= d.size()) {
        const uint32_t len = u32(pos + 4);
        if (!std::memcmp(d.data() + pos, "fmt ", 4) && pos + 24 <= d.size()) {
            fmt = (int) u16(pos + 8);
            ch = (int) u16(pos + 10);
            sr = (int) u32(pos + 12);
            bits = (int) u16(pos + 22);
            if (fmt == 0xFFFE && len >= 26) fmt = (int) u16(pos + 32);
        } else if (!std::memcmp(d.data() + pos, "data", 4)) {
            data_off = pos + 8;
            data_len = std::min<size_t>(len, d.size() - data_off);
            break;
        }
        pos += 8 + len + (len & 1);
    }
    if (!data_off || ch <= 0) {
        err = path + ": no fmt/data chunk";
        return false;
    }
    const int bps = bits / 8;
    const size_t frames = data_len / (size_t) (bps * ch);
    out.assign(frames, 0.0f);
    for (size_t i = 0; i < frames; ++i) {
        double s = 0.0;
        for (int c = 0; c < ch; ++c) {
            const uint8_t * p = d.data() + data_off + (i * ch + c) * bps;
            if (fmt == 1 && bits == 16) s += (int16_t) (p[0] | (p[1] << 8)) / 32768.0;
            else if (fmt == 1 && bits == 32) s += (int32_t) (p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24)) / 2147483648.0;
            else if (fmt == 1 && bits == 24) s += ((int32_t) ((p[0] << 8) | (p[1] << 16) | ((uint32_t) p[2] << 24)) >> 8) / 8388608.0;
            else if (fmt == 3 && bits == 32) { float v; std::memcpy(&v, p, 4); s += v; }
            else { err = path + ": unsupported sample format"; return false; }
        }
        out[i] = (float) (s / ch);
    }
    return true;
}

// Kaiser-windowed sinc, 32 zero crossings; adequate for test tools.
inline std::vector<float> resample(const std::vector<float> & x, int sr_in, int sr_out) {
    if (sr_in == sr_out) return x;
    const double ratio = (double) sr_out / sr_in;
    const double cutoff = std::min(1.0, ratio) * 0.97;
    const int half = 32;
    const size_t n_out = (size_t) std::floor(x.size() * ratio);
    std::vector<float> y(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double t = i / ratio;
        const long c = (long) std::floor(t);
        double acc = 0.0, wsum = 0.0;
        for (long k = c - half / cutoff; k <= c + half / cutoff; ++k) {
            if (k < 0 || k >= (long) x.size()) continue;
            const double u = (t - k) * cutoff;
            const double sinc = u == 0.0 ? 1.0 : std::sin(kPi * u) / (kPi * u);
            const double r = u / half;
            const double w = std::fabs(r) >= 1.0 ? 0.0 : 0.5 + 0.5 * std::cos(kPi * r);
            acc += x[k] * sinc * w * cutoff;
            wsum += sinc * w * cutoff;
        }
        y[i] = (float) (wsum != 0.0 ? acc : 0.0);
    }
    return y;
}

}  // namespace pkwav
