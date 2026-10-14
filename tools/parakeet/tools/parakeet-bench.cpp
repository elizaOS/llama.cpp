// tools/parakeet/tools/parakeet-bench.cpp - speed harness.
//
//   parakeet-bench -m model.gguf -t N --list utts.tsv [--prefix-ms 1000]
//                  [--stream] [--chunk-ms 80] [--warmup 1] [--repeat R] > results.jsonl
//
// utts.tsv: one "<id>\t<wav path>" per line. For every utterance it prints a
// JSON line with the transcript, wall time, process CPU time (getrusage, so
// worker threads are included), the time to decode the first prefix-ms of
// audio (the first-window partial of a stream) and, with --stream, the
// streaming result fed in chunk-ms pieces. The last line is a summary with
// the peak RSS. With --repeat R every timing is the minimum of R runs, which
// filters out preemption on a shared host. Times are measured on the host this runs on; an Apple
// Silicon Mac is an ARM CPU proxy, not phone evidence.
#include "parakeet.h"
#include "pk-wav.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/time.h>
#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

static double cpu_ms() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1e3 + (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e3;
}

static double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return ru.ru_maxrss / (1024.0 * 1024.0);
#else
    return ru.ru_maxrss / 1024.0;
#endif
}

static std::string jesc(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char) c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

int main(int argc, char ** argv) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    std::string model, list;
    pk_params pp = pk_default_params();
    int prefix_ms = 1000, chunk_ms = 80, warmup = 1, repeat = 1;
    bool stream = false;
    pk_stream_params sp = pk_stream_default_params();
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "-m") model = next();
        else if (a == "-t") pp.n_threads = std::atoi(next());
        else if (a == "--kernel") { const std::string k = next(); pp.kernel = k == "scalar" ? 1 : k == "dotprod" ? 2 : k == "i8mm" ? 3 : k == "avx2" ? 4 : 0; }
        else if (a == "--no-pin") pp.pin_threads = 0;
        else if (a == "--list") list = next();
        else if (a == "--prefix-ms") prefix_ms = std::atoi(next());
        else if (a == "--stream") stream = true;
        else if (a == "--chunk-ms") chunk_ms = std::atoi(next());
        else if (a == "--warmup") warmup = std::atoi(next());
        else if (a == "--repeat") repeat = std::max(1, std::atoi(next()));
        else if (a == "--right-ms") sp.right_context_ms = std::atoi(next());
        else if (a == "--hop-ms") sp.min_hop_ms = std::atoi(next());
        else if (a == "--max-window-ms") sp.max_window_ms = std::atoi(next());
    }
    if (model.empty() || list.empty()) {
        std::fprintf(stderr, "usage: %s -m model.gguf -t N --list utts.tsv [--stream]\n", argv[0]);
        return 2;
    }
    char err[512] = { 0 };
    const auto tl = std::chrono::steady_clock::now();
    const double rss0 = peak_rss_mb();
    pk_model * m = pk_model_load(model.c_str(), pp, err, sizeof(err));
    if (!m) { std::fprintf(stderr, "load failed: %s\n", err); return 1; }
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl).count();
    std::vector<std::pair<std::string, std::string>> utts;
    {
        std::ifstream in(list);
        std::string line;
        while (std::getline(in, line)) {
            const size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            utts.emplace_back(line.substr(0, tab), line.substr(tab + 1));
        }
    }
    std::vector<std::vector<float>> audio(utts.size());
    for (size_t i = 0; i < utts.size(); ++i) {
        int sr = 0;
        std::string e;
        if (!pkwav::read_wav(utts[i].second, audio[i], sr, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        audio[i] = pkwav::resample(audio[i], sr, 16000);
    }
    for (int w = 0; w < warmup && !audio.empty(); ++w) {
        pk_result r;
        pk_transcribe(m, audio[0].data(), audio[0].size(), nullptr, nullptr, &r);
        pk_result_free(&r);
    }
    double tot_audio = 0, tot_wall = 0, tot_cpu = 0;
    for (size_t i = 0; i < utts.size(); ++i) {
        const auto & pcm = audio[i];
        const double audio_s = pcm.size() / 16000.0;
        pk_result r{};
        double wall = 1e30, cpu = 1e30, pwall = 1e30, pcpu = 1e30;
        double c0;
        std::chrono::steady_clock::time_point t0;
        const size_t np = std::min(pcm.size(), (size_t) prefix_ms * 16);
        for (int rep = 0; rep < repeat; ++rep) {
            if (rep) pk_result_free(&r);
            c0 = cpu_ms();
            t0 = std::chrono::steady_clock::now();
            pk_transcribe(m, pcm.data(), pcm.size(), nullptr, nullptr, &r);
            wall = std::min(wall, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            cpu = std::min(cpu, cpu_ms() - c0);
            // first-window partial: decode the first prefix_ms of audio
            pk_result rp;
            c0 = cpu_ms();
            t0 = std::chrono::steady_clock::now();
            pk_transcribe(m, pcm.data(), np, nullptr, nullptr, &rp);
            pwall = std::min(pwall, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            pcpu = std::min(pcpu, cpu_ms() - c0);
            pk_result_free(&rp);
        }
        tot_audio += audio_s; tot_wall += wall; tot_cpu += cpu;
        std::ostringstream js;
        js << "{\"id\":\"" << jesc(utts[i].first) << "\",\"audio_s\":" << audio_s << ",\"wall_ms\":" << wall
           << ",\"cpu_ms\":" << cpu << ",\"mel_ms\":" << r.t_mel_ms << ",\"enc_ms\":" << r.t_encoder_ms
           << ",\"dec_ms\":" << r.t_decoder_ms << ",\"prefix_wall_ms\":" << pwall << ",\"prefix_cpu_ms\":" << pcpu
           << ",\"text\":\"" << jesc(r.text ? r.text : "") << "\"";
        pk_result_free(&r);
        if (stream) {
            pk_stream * s = pk_stream_open(m, sp);
            const size_t step = (size_t) chunk_ms * 16;
            std::vector<char> buf(1 << 16);
            c0 = cpu_ms();
            double first_wall = -1;
            for (size_t p = 0; p < pcm.size(); p += step) {
                pk_stream_feed(s, pcm.data() + p, std::min(step, pcm.size() - p));
                t0 = std::chrono::steady_clock::now();
                pk_stream_partial(s, buf.data(), buf.size(), nullptr, nullptr);
                int nd = 0;
                pk_stream_stats(s, &nd, nullptr, nullptr);
                if (first_wall < 0 && nd > 0) {
                    first_wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                }
            }
            t0 = std::chrono::steady_clock::now();
            pk_stream_finish(s, buf.data(), buf.size(), nullptr, nullptr);
            const double fin = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const double scpu = cpu_ms() - c0;
            int nd = 0;
            double dms = 0, first = 0;
            pk_stream_stats(s, &nd, &dms, &first);
            js << ",\"stream_text\":\"" << jesc(buf.data()) << "\",\"stream_decodes\":" << nd
               << ",\"stream_decode_ms\":" << dms << ",\"stream_cpu_ms\":" << scpu
               << ",\"stream_first_partial_ms\":" << first_wall << ",\"stream_finish_ms\":" << fin;
            pk_stream_close(s);
        }
        js << "}";
        std::printf("%s\n", js.str().c_str());
        std::fflush(stdout);
    }
    double la[3] = { 0, 0, 0 };
#if defined(__APPLE__) || defined(__linux__)
    getloadavg(la, 3);
#endif
    std::printf("{\"summary\":true,\"kernel\":\"%s\",\"threads\":%d,\"n\":%zu,\"audio_s\":%.2f,\"wall_s\":%.3f,"
                "\"cpu_s\":%.3f,\"rtfx_wall\":%.2f,\"rtfx_cpu\":%.2f,\"load_ms\":%.0f,\"peak_rss_mb\":%.0f,"
                "\"rss_before_load_mb\":%.0f,\"loadavg\":%.1f}\n",
                pk_model_kernel_name(m), pk_model_n_threads(m), utts.size(), tot_audio, tot_wall / 1e3, tot_cpu / 1e3,
                tot_audio * 1e3 / tot_wall, tot_audio * 1e3 / tot_cpu, load_ms, peak_rss_mb(), rss0, la[0]);
    pk_model_free(m);
    return 0;
}
