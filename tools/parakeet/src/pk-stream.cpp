// tools/parakeet/src/pk-stream.cpp - streaming on a full-context model.
//
// Parakeet-TDT-v3 (and Phonon-2) attends over the whole input and has no
// cache-aware mode, so a stream re-decodes a window. Windows must not start
// in the middle of speech: the model (the dense fp32 reference included)
// drops or invents words when its input begins mid-word. So the stream works
// on pause-delimited segments:
//
//   * every decode covers [segment start, newest sample);
//   * tokens are committed when two consecutive decodes agree on them and
//     they lie at least `right_context` before the newest sample, on word
//     boundaries only;
//   * when a pause (>= 150 ms of low energy) lies in committed audio, the next
//     segment starts in it (preferably between the last committed token and
//     the first uncommitted one), so later windows begin in silence and stay
//     short; committed tokens a new window re-decodes are skipped by count, or
//     by frame and id when the re-decode differs;
//   * a segment longer than `max_window` without a pause is cut at the
//     quietest 100 ms of its last 2 s of committed audio;
//   * the decode interval grows with the window (at least min_hop, at most
//     a sixth of the window), bounding the re-decode cost;
//   * finish() decodes the open segment with full right context.
//
// Committed text never changes. Partials are committed + tentative tokens.
#include "parakeet.h"

#include "pk-model.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace pk;

struct pk_stream {
    pk_model * m = nullptr;
    pk_stream_params p{};
    std::vector<float> audio;  // samples [audio_base, total)
    size_t audio_base = 0;
    size_t total = 0;
    size_t last_decode_total = 0;
    size_t seg_start = 0;      // absolute sample, multiple of one encoder frame
    size_t commit_sample = 0;  // all committed tokens lie before this
    std::vector<pk_token> committed;  // absolute frames
    std::vector<pk_token> tentative;  // absolute frames
    bool done = false;
    float speech_level = 0.0f;  // loud (90th percentile) 10 ms energy of the last window with tokens
    int n_decodes = 0;
    int n_skipped = 0;          // decodes skipped on quiet audio
    size_t committed_in_segment(int f0) const {
        size_t n = 0;
        while (n < committed.size() && committed[committed.size() - 1 - n].frame >= f0) ++n;
        return n;
    }
    double decode_ms = 0.0;
    double first_ms = -1.0;
};

extern "C" pk_stream_params pk_stream_default_params(void) {
    pk_stream_params p;
    p.right_context_ms = 800;
    p.min_hop_ms = 400;
    p.max_window_ms = 12000;
    return p;
}

extern "C" pk_stream * pk_stream_open(pk_model * m, pk_stream_params p) {
    if (!m) return nullptr;
    auto * s = new pk_stream();
    s->m = m;
    s->p = p;
    return s;
}

extern "C" int pk_stream_feed(pk_stream * s, const float * pcm, size_t n) {
    if (!s || (!pcm && n)) return PK_E_INVALID;
    if (s->done) return PK_E_INVALID;
    s->audio.insert(s->audio.end(), pcm, pcm + n);
    s->total += n;
    return (int) n;
}

// token identity for re-decode matching: the piece without its word-start
// space, lower-cased (a segment's first word is often re-decoded capitalised)
static bool same_piece(const pk_model * m, int a, int b) {
    if (a == b) return true;
    const std::string & x = m->pieces[a];
    const std::string & y = m->pieces[b];
    size_t i = x.find_first_not_of(' '), j = y.find_first_not_of(' ');
    if (i == std::string::npos || j == std::string::npos) return false;
    if (x.size() - i != y.size() - j) return false;
    for (; i < x.size(); ++i, ++j) {
        if (std::tolower((unsigned char) x[i]) != std::tolower((unsigned char) y[j])) return false;
    }
    return true;
}

static size_t ms_to_samples(const pk_stream * s, int ms) {
    return (size_t) ((long long) std::max(0, ms) * s->m->hp.sample_rate / 1000);
}

static const float * at(const pk_stream * s, size_t abs_sample) { return s->audio.data() + (abs_sample - s->audio_base); }

// 10 ms frame energies over [a, b)
static std::vector<float> frame_energy(const pk_stream * s, size_t a, size_t b) {
    const size_t hop = 160;
    std::vector<float> e;
    for (size_t x = a; x + hop <= b; x += hop) {
        const float * p = at(s, x);
        double acc = 0.0;
        for (size_t i = 0; i < hop; ++i) acc += (double) p[i] * p[i];
        e.push_back((float) (acc / hop));
    }
    return e;
}

// 90th percentile of the 10 ms frame energies over [a, b)
static float loud_level(const pk_stream * s, size_t a, size_t b) {
    std::vector<float> e = frame_energy(s, a, b);
    if (e.empty()) return 0.0f;
    std::nth_element(e.begin(), e.begin() + e.size() * 9 / 10, e.end());
    return e[e.size() * 9 / 10];
}

// Quiet: every 10 ms frame below -60 dBFS, or below -30 dB of the speech
// level of the last window that held tokens.
static bool is_quiet(const pk_stream * s, size_t a, size_t b) {
    const float thr = std::max(1e-6f, 0.03f * 0.03f * s->speech_level);
    for (float v : frame_energy(s, a, b)) {
        if (v >= thr) return false;
    }
    return true;
}

static void drop_before(pk_stream * s, size_t abs_sample) {
    if (abs_sample > s->audio_base) {
        s->audio.erase(s->audio.begin(), s->audio.begin() + (abs_sample - s->audio_base));
        s->audio_base = abs_sample;
    }
}

// true when sample x falls in no committed token's span [frame, frame + dur)
static bool between_tokens(const pk_stream * s, size_t x, size_t fs) {
    for (auto it = s->committed.rbegin(); it != s->committed.rend(); ++it) {
        const size_t b = (size_t) it->frame * fs;
        const size_t e = (size_t) (it->frame + std::max(1, it->duration)) * fs;
        if (e <= x) return true;   // tokens are in time order
        if (b <= x) return false;  // x inside this token
    }
    return true;
}

// Latest pause midpoint in [lo, hi): >= 15 consecutive 10 ms frames below
// 5% (amplitude) of the segment's loud level (90th percentile energy), and
// not inside a committed token. 0 when there is none.
static size_t find_pause(const pk_stream * s, size_t lo, size_t hi) {
    if (hi <= lo + 3200) return 0;
    const std::vector<float> seg = frame_energy(s, s->seg_start, std::min(hi, s->total));
    if (seg.size() < 40) return 0;
    std::vector<float> sorted(seg);
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() * 9 / 10, sorted.end());
    const float loud = sorted[sorted.size() * 9 / 10];
    const float thr = std::max(1e-7f, 0.05f * 0.05f * loud);
    size_t best = 0;
    size_t run = 0;
    for (size_t i = 0; i < seg.size(); ++i) {
        run = seg[i] < thr ? run + 1 : 0;
        if (run >= 15) {
            const size_t x = s->seg_start + (i + 1) * 160;  // end of the run so far
            const size_t mid = x - run * 80;
            const size_t fs = (size_t) s->m->hp.hop_length * 8;
            const size_t cut = (mid / fs) * fs;  // where the segment would start
            if (mid >= lo && mid < hi && between_tokens(s, cut, fs)) best = mid;
        }
    }
    return best;
}

// centre of the quietest 100 ms in [lo, hi)
static size_t quietest(const pk_stream * s, size_t lo, size_t hi) {
    const std::vector<float> e = frame_energy(s, lo, hi);
    if (e.size() < 10) return hi;
    size_t best = 0;
    double be = 1e300;
    for (size_t i = 0; i + 10 <= e.size(); ++i) {
        double acc = 0;
        for (size_t j = 0; j < 10; ++j) acc += e[i + j];
        if (acc < be) {
            be = acc;
            best = i;
        }
    }
    return lo + best * 160 + 800;
}

static int decode_window(pk_stream * s, bool final, pk_cancel_cb cancel, void * cu) {
    pk_model * m = s->m;
    const size_t fs = (size_t) m->hp.hop_length * 8;  // samples per encoder frame (1280)
    const size_t ws = s->seg_start;
    if (s->total <= ws) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> mel, encp;
    int nm = 0;
    std::vector<pk_token> toks;
    if (compute_mel(m, at(s, ws), s->total - ws, mel, nm)) {
        const int T = encode(m, mel.data(), nm, encp, cancel, cu);
        if (T < 0) return PK_E_CANCELLED;
        if (tdt_decode(m, encp.data(), T, toks, cancel, cu) < 0) return PK_E_CANCELLED;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    s->n_decodes++;
    s->decode_ms += ms;
    if (s->first_ms < 0) s->first_ms = ms;
    s->last_decode_total = s->total;

    const int f0 = (int) (ws / fs);
    for (auto & t : toks) t.frame += f0;
    // PK_STREAM_DEBUG: window geometry and token counts only. Transcript
    // text and token ids never go to a log.
    const bool dbg = std::getenv("PK_STREAM_DEBUG") != nullptr;
    if (dbg) {
        std::fprintf(stderr, "[pk-stream] window %.2f-%.2fs: %zu tokens, %.1f ms\n", ws / 16000.0,
                     s->total / 16000.0, toks.size(), ms);
    }

    // The window re-decodes this segment's committed tokens too. Skip them by
    // count when the ids agree, otherwise by the commit point.
    size_t seg_committed = 0;
    while (seg_committed < s->committed.size() &&
           s->committed[s->committed.size() - 1 - seg_committed].frame >= f0) {
        ++seg_committed;
    }
    std::vector<pk_token> hyp;
    {
        const size_t c0 = s->committed.size() - seg_committed;
        bool same = toks.size() >= seg_committed;
        for (size_t i = 0; same && i < seg_committed; ++i) same = same_piece(m, toks[i].id, s->committed[c0 + i].id);
        size_t start = 0;
        if (same) {
            start = seg_committed;
        } else {
            // the segment's committed tokens re-decoded differently: keep what
            // starts after the commit point, minus re-emissions of the
            // committed tail (same ids within 3 frames)
            const int boundary = (int) (s->commit_sample / fs);
            while (start < toks.size() && toks[start].frame < boundary - 3) ++start;
            size_t drop = 0;
            const size_t avail = toks.size() - start;
            const size_t kmax = std::min<size_t>(6, std::min(s->committed.size(), avail));
            for (size_t kk = kmax; kk >= 1 && !drop; --kk) {
                bool ok = true;
                for (size_t i = 0; i < kk && ok; ++i) {
                    const pk_token & c = s->committed[s->committed.size() - kk + i];
                    ok = same_piece(m, c.id, toks[start + i].id) && std::abs(c.frame - toks[start + i].frame) <= 3;
                }
                if (ok) drop = kk;
            }
            start += drop;
            while (start < toks.size() && toks[start].frame < boundary) ++start;
        }
        hyp.assign(toks.begin() + start, toks.end());
    }
    if (final) {
        s->committed.insert(s->committed.end(), hyp.begin(), hyp.end());
        s->tentative.clear();
        s->commit_sample = s->total;
        return 0;
    }
    const size_t rc = ms_to_samples(s, s->p.right_context_ms);
    const int stable = (int) ((s->total > rc ? s->total - rc : 0) / fs);
    size_t k = 0;
    while (k < hyp.size() && k < s->tentative.size() && hyp[k].id == s->tentative[k].id && hyp[k].frame < stable) ++k;
    const bool force = s->total - ws > ms_to_samples(s, s->p.max_window_ms);
    if (force) {
        while (k < hyp.size() && hyp[k].frame < stable) ++k;
    }
    // whole words only: the next token must start a word. The newest word
    // also counts as finished when every token of the hypothesis ends before
    // the stable point (right_context of audio with no further token):
    // otherwise the last word before a pause would wait for the next word.
    const bool tail_done = !hyp.empty() && hyp.back().frame + std::max(1, hyp.back().duration) <= stable;
    while (k > 0 && !(k < hyp.size() ? m->word_start[hyp[k].id] != 0 : tail_done)) --k;
    if (k > 0) {
        s->committed.insert(s->committed.end(), hyp.begin(), hyp.begin() + k);
        // commit point: the start frame of the first uncommitted token, or
        // the end of the last committed one when nothing is left
        const size_t cp = k < hyp.size() ? (size_t) hyp[k].frame * fs
                                         : (size_t) (hyp.back().frame + std::max(1, hyp.back().duration)) * fs;
        s->commit_sample = std::max(s->commit_sample, cp);
    }
    s->tentative.assign(hyp.begin() + k, hyp.end());
    if (!toks.empty()) s->speech_level = loud_level(s, ws, s->total);

    // Start a new segment in a pause, so the next window begins in silence.
    // Preferred: a pause between the last committed token and the first
    // uncommitted one (the new window then holds no committed token). Else
    // the latest pause in the last 3 s of committed audio (its committed
    // tokens are re-decoded and skipped above). Past max_window with no
    // pause at all: the quietest 100 ms of the last 2 s of committed audio.
    // Past max_window with no token at all (noise, no speech): keep the
    // newest half window, so cost and memory stay bounded and the window is
    // still long enough for the per-window feature normalisation.
    if (hyp.empty() && s->committed_in_segment(f0) == 0 && force) {
        const size_t keep = ms_to_samples(s, s->p.max_window_ms / 2);
        const size_t cut = ((s->total - keep) / fs) * fs;
        if (cut > ws) {
            s->seg_start = cut;
            s->commit_sample = std::max(s->commit_sample, cut);
            drop_before(s, cut);
        }
    } else if (!s->committed.empty() && s->commit_sample > ws + fs) {
        const pk_token & lc = s->committed.back();
        const size_t gap_lo = std::max(ws + fs, (size_t) (lc.frame + std::max(1, lc.duration)) * fs);
        size_t cut = gap_lo < s->commit_sample ? find_pause(s, gap_lo, s->commit_sample) : 0;
        if (!cut) {
            const size_t lo3 = std::max(ws + fs, s->commit_sample > 48000 ? s->commit_sample - 48000 : 0);
            cut = find_pause(s, lo3, s->commit_sample);
        }
        if (!cut && force) {
            const size_t lo2 = std::max(ws + fs, s->commit_sample > 32000 ? s->commit_sample - 32000 : 0);
            if (s->commit_sample > lo2 + 1600) cut = quietest(s, lo2, s->commit_sample);
        }
        cut = (cut / fs) * fs;
        if (cut > ws && cut <= s->commit_sample) {
            s->seg_start = cut;
            drop_before(s, cut);
        }
    }
    if (dbg) {
        std::fprintf(stderr, "[pk-stream]   commit+%zu -> %.2fs, %zu tentative, segment from %.2fs%s\n", k,
                     s->commit_sample / 16000.0, s->tentative.size(), s->seg_start / 16000.0,
                     force ? " (forced)" : "");
    }
    return 0;
}

static int write_text(pk_stream * s, char * out, size_t cap) {
    std::vector<pk_token> all = s->committed;
    all.insert(all.end(), s->tentative.begin(), s->tentative.end());
    const std::string t = detokenize(s->m, all.data(), all.size());
    if (!out || cap == 0) return (int) t.size();
    if (t.size() >= cap) {
        out[0] = '\0';
        return PK_E_INVALID;
    }
    std::memcpy(out, t.data(), t.size());
    out[t.size()] = '\0';
    return (int) t.size();
}

extern "C" int pk_stream_partial(pk_stream * s, char * out, size_t cap, pk_cancel_cb cancel, void * cu) {
    if (!s) return PK_E_INVALID;
    // decode interval: at least min_hop, growing to a sixth of the window
    const size_t win = s->total - s->seg_start;
    const size_t hop = std::max(ms_to_samples(s, s->p.min_hop_ms), win / 6);
    if (!s->done && s->total - s->last_decode_total >= hop) {
        if (s->tentative.empty() && s->last_decode_total > s->seg_start &&
            is_quiet(s, s->last_decode_total, s->total)) {
            // Nothing pending and only quiet audio since the last decode: a
            // decode could only re-read silence (and short noise windows are
            // where the model invents words). Skip it, and keep at most the
            // newest 1.5 s of a quiet stretch, so a long pause neither grows
            // the next window nor the buffer.
            s->n_skipped++;
            s->last_decode_total = s->total;
            const size_t fs = (size_t) s->m->hp.hop_length * 8;
            size_t lo = s->seg_start + fs;
            if (!s->committed.empty()) {
                const pk_token & lc = s->committed.back();
                lo = std::max(lo, (size_t) (lc.frame + std::max(1, lc.duration)) * fs);
            }
            const size_t keep = ms_to_samples(s, 1500);
            if (s->total > lo + keep + fs) {
                const size_t cut = ((s->total - keep) / fs) * fs;
                if (cut > lo && is_quiet(s, std::max(lo, cut - fs), s->total)) {
                    s->seg_start = cut;
                    s->commit_sample = std::max(s->commit_sample, cut);
                    drop_before(s, cut);
                }
            }
        } else {
            const int rc = decode_window(s, false, cancel, cu);
            if (rc < 0) return rc;
        }
    }
    return write_text(s, out, cap);
}

extern "C" int pk_stream_finish(pk_stream * s, char * out, size_t cap, pk_cancel_cb cancel, void * cu) {
    if (!s) return PK_E_INVALID;
    if (!s->done) {
        const int rc = decode_window(s, true, cancel, cu);
        if (rc < 0) return rc;
        s->done = true;
    }
    return write_text(s, out, cap);
}

extern "C" void pk_stream_stats(const pk_stream * s, int * n, double * ms, double * first) {
    if (!s) return;
    if (n) *n = s->n_decodes;
    if (ms) *ms = s->decode_ms;
    if (first) *first = s->first_ms;
}

extern "C" void pk_stream_close(pk_stream * s) { delete s; }
