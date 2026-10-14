// Scripting + counters for the Paradee / misaki test doubles in
// fake-paradee.cpp. Only the FFI unit test uses this.
#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

struct fake_paradee_state {
    std::atomic<int>  model_loads{0};
    std::atomic<int>  live_models{0};
    std::atomic<int>  lexicon_loads{0};
    std::atomic<int>  live_lexicons{0};
    std::atomic<int>  synth_calls{0};
    std::atomic<bool> in_flight{false};
    // > 0: each paradee_synthesize call runs this long, polling the cancel
    // callback every millisecond (stand-in for a slow forward pass).
    std::atomic<int>  block_ms{0};
    std::mutex          sizes_mutex;
    std::vector<size_t> input_sizes;   // input_ids.size() per synth call
};

fake_paradee_state & fake_paradee();
void fake_paradee_reset();

// Samples the fake emits per non-pad input id.
constexpr int FAKE_PARADEE_SAMPLES_PER_ID = 100;
