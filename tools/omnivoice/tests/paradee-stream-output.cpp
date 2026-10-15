#include "../src/paradee-stream-output.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <vector>

int main() {
    std::vector<float> pcm(48007);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = float(i);
    std::vector<float> delivered;
    std::vector<size_t> sizes;
    bool cancelled = false;
    auto consume = [&](const float * data, size_t n, int final, void *) {
        assert(!final && n <= 24000);
        sizes.push_back(n);
        delivered.insert(delivered.end(), data, data + n);
        return 0;
    };
    assert(eliza_paradee_stream_output(pcm.data(), pcm.size(), consume, nullptr, [&] { return cancelled; }));
    assert(delivered == pcm && sizes == std::vector<size_t>({24000, 24000, 7}));
    int calls = 0;
    assert(!eliza_paradee_stream_output(pcm.data(), pcm.size(),
        [&](const float *, size_t, int, void *) { ++calls; return 1; }, nullptr, [] { return false; }));
    assert(calls == 1);
    calls = 0;
    assert(!eliza_paradee_stream_output(pcm.data(), pcm.size(),
        [&](const float *, size_t, int, void *) { ++calls; cancelled = true; return 0; }, nullptr, [&] { return cancelled; }));
    assert(calls == 1);
    assert(!eliza_paradee_stream_output(pcm.data(), pcm.size(), consume, nullptr, [] { return true; }));
    assert(eliza_paradee_stream_output(nullptr, 0, consume, nullptr, [] { return false; }));
}
