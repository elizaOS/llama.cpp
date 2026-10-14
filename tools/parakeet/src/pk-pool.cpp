// tools/parakeet/src/pk-pool.cpp
#include "pk-pool.h"

#include <algorithm>
#include <cstdio>
#include <string>

#if defined(__linux__)
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_ARM64)
#define PK_RELAX() __yield()
#else
#define PK_RELAX() _mm_pause()
#endif
#elif defined(__aarch64__)
#define PK_RELAX() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__) || defined(__i386__)
#define PK_RELAX() __builtin_ia32_pause()
#else
#define PK_RELAX() ((void) 0)
#endif

namespace pk {

std::vector<int> fast_cpus() {
    std::vector<int> out;
#if defined(__linux__)
    std::vector<std::pair<int, long>> f;
    for (int c = 0; c < 64; ++c) {
        const std::string p = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/cpufreq/cpuinfo_max_freq";
        FILE * fp = std::fopen(p.c_str(), "r");
        if (!fp) {
            continue;
        }
        long v = 0;
        if (std::fscanf(fp, "%ld", &v) == 1) {
            f.emplace_back(c, v);
        }
        std::fclose(fp);
    }
    long mx = 0;
    for (auto & e : f) {
        mx = std::max(mx, e.second);
    }
    for (auto & e : f) {
        if (mx > 0 && e.second * 10 >= mx * 8) {
            out.push_back(e.first);
        }
    }
#endif
    return out;
}

int default_threads() {
    int n = 0;
#if defined(__APPLE__)
    int v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &v, &len, nullptr, 0) == 0) {
        n = v;
    }
#elif defined(__linux__)
    n = (int) fast_cpus().size();
#endif
    if (n <= 0) {
        n = (int) std::thread::hardware_concurrency() / 2;
    }
    return std::max(1, std::min(4, n));
}

pool::pool(int n_threads, bool pin) : n_(std::max(1, n_threads)) {
    std::vector<int> cpus;
    if (pin) {
        cpus = fast_cpus();
    }
    for (int t = 1; t < n_; ++t) {
        threads_.emplace_back([this, t, cpus] {
#if defined(__linux__)
            if (!cpus.empty()) {
                cpu_set_t set;
                CPU_ZERO(&set);
                for (int c : cpus) {
                    CPU_SET(c, &set);
                }
                sched_setaffinity((pid_t) syscall(SYS_gettid), sizeof(set), &set);
            }
#else
            (void) cpus;
#endif
#if defined(__APPLE__)
            // keep workers on performance cores (no affinity API on Darwin)
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
            worker(t);
        });
    }
}

pool::~pool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_.store(true);
        gen_.fetch_add(1);
    }
    cv_.notify_all();
    for (auto & th : threads_) {
        th.join();
    }
}

void pool::worker(int tid) {
    uint64_t seen = 0;
    for (;;) {
        uint64_t g = gen_.load(std::memory_order_acquire);
        if (g == seen) {
            // spin briefly, then yield (other threads may need this core when
            // the CPU is oversubscribed), then sleep
            for (int i = 0; i < 20000 && g == seen; ++i) {
                if (i < 2000) {
                    PK_RELAX();
                } else {
                    std::this_thread::yield();
                }
                g = gen_.load(std::memory_order_acquire);
            }
            if (g == seen) {
                std::unique_lock<std::mutex> lk(m_);
                sleeping_.fetch_add(1);
                cv_.wait(lk, [&] { return gen_.load() != seen; });
                sleeping_.fetch_sub(1);
                g = gen_.load();
            }
        }
        seen = g;
        if (stop_.load()) {
            return;
        }
        (*job_)(tid);
        remaining_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void pool::run(const std::function<void(int)> & f) {
    if (n_ == 1) {
        f(0);
        return;
    }
    job_ = &f;
    remaining_.store(n_ - 1, std::memory_order_relaxed);
    gen_.fetch_add(1);
    if (sleeping_.load() > 0) {
        std::lock_guard<std::mutex> lk(m_);
        cv_.notify_all();
    }
    f(0);
    for (int i = 0; remaining_.load(std::memory_order_acquire) > 0; ++i) {
        if (i < 2000) {
            PK_RELAX();
        } else {
            std::this_thread::yield();  // a worker was preempted; let it run
        }
    }
}

}  // namespace pk
