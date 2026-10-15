#pragma once
#include <algorithm>
#include <cstddef>

// Callback lifetime and backpressure apply to each bounded PCM window.
template<class Callback, class Cancelled>
bool eliza_paradee_stream_output(const float * pcm, size_t count,
        Callback callback, void * user, Cancelled cancelled) {
    constexpr size_t max_samples = 24000;
    for (size_t offset = 0; offset < count;) {
        if (cancelled()) return false;
        const size_t size = std::min(max_samples, count - offset);
        if (callback(pcm + offset, size, 0, user) != 0 || cancelled()) return false;
        offset += size;
    }
    return !cancelled();
}
