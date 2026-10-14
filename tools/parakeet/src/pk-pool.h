// tools/parakeet/src/pk-pool.h - a small persistent worker pool. Workers spin
// briefly between jobs (the encoder issues a few hundred short parallel ops
// per second of audio), then sleep on a condition variable when idle.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace pk {

class pool {
  public:
    pool(int n_threads, bool pin);
    ~pool();
    pool(const pool &) = delete;
    pool & operator=(const pool &) = delete;

    int size() const { return n_; }

    // Run f(tid) on every thread; the caller runs tid 0. Returns when all
    // threads finished.
    void run(const std::function<void(int)> & f);

    // Hand out [0, n) in chunks of `grain` to all threads; f(begin, end).
    template <class F> void parallel_for(int n, int grain, F && f) {
        if (n <= 0) {
            return;
        }
        if (n_ == 1 || n <= grain) {
            f(0, n);
            return;
        }
        std::atomic<int> next{0};
        run([&](int) {
            for (;;) {
                const int b = next.fetch_add(grain, std::memory_order_relaxed);
                if (b >= n) {
                    break;
                }
                f(b, b + grain < n ? b + grain : n);
            }
        });
    }

  private:
    void worker(int tid);

    int n_;
    std::vector<std::thread> threads_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> remaining_{0};
    std::atomic<int> sleeping_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int)> * job_ = nullptr;
    std::mutex m_;
    std::condition_variable cv_;
};

// Threads to use when the caller passes n_threads <= 0: the fast cores
// (Android big/prime cores, Apple performance cores), at most 4.
int default_threads();
// Linux/Android: CPU ids whose cpuinfo_max_freq is within 20% of the fastest.
std::vector<int> fast_cpus();

}  // namespace pk
