#pragma once
// streaming-opts.h — W3-3 streaming-path optimizations.
//
// Three concrete optimizations are wired here (per VOICE_WAVE_3.md §4
// W3-3):
//
//   1. madvise(MADV_DONTNEED) for transient MaskGIT scratch.
//      The 28L Qwen3 bidirectional forward holds a large attention-mask
//      F16 buffer (B' * S * S * sizeof(f16)) plus position-id +
//      mask buffers across the 32 MaskGIT steps. These are NOT touched
//      between chunks; releasing the dirty pages back to the kernel
//      lifts the documented ~1.17 GB transient peak (voice-budget.ts).
//
//      The helper takes a (ptr, bytes) pair and:
//        * On Linux:  posix_madvise(MADV_DONTNEED) — the kernel zeroes
//          the page on next touch.
//        * On macOS:  madvise(MADV_FREE_REUSABLE) — equivalent semantic.
//        * On Windows: best effort VirtualUnlock; subsequent access
//          remains valid (Win32 has no exact equivalent of DONTNEED).
//        * Other / unsupported: no-op.
//
//      A single failure mode is a non-zero return; callers may log but
//      must NEVER swallow this into a fatal error path (the next-touch
//      semantics are best-effort; correctness does not depend on
//      success).
//
//   2. Mel codec streaming buffer flush hook.
//      The DAC vocoder upsamples codebook tokens to 24 kHz PCM at a
//      hop. When streaming, the runtime wants to emit fully-decoded
//      hops as fast as the codec produces them rather than accumulate.
//      `omnivoice_streaming_flush_hops()` is the well-known name the
//      pipeline-tts streaming path calls when N hops have buffered.
//      The pipeline-codec module owns the underlying buffer; this is a
//      thin facade so the codec module can be patched without changing
//      the call site signature.
//
//   3. Slot-based KV layout sentinel.
//      The shipped MaskGIT decoder has NO KV cache across steps (every
//      forward is stateless / bidirectional). The "slot-based KV"
//      optimization in W3-3 only applies to the LM-conditioning
//      *prefix* (instruct tokens + ref-audio tokens) — these stay
//      constant across calls in a frozen-voice path, so multiple
//      concurrent TTS streams can share the prefix forward output
//      without re-running it.
//
//      The sentinel function `omnivoice_streaming_prefix_slot()`
//      returns the slot index a given preset id maps to. A slot pool
//      with LRU eviction is documented for future implementation.
//      Today the function returns -1 to mean "no slot cached", and
//      callers fall through to the normal forward. The hook exists so
//      a slot pool can land without breaking the calling convention.

#include <atomic>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#if defined(__linux__) || defined(__APPLE__)
#  include <sys/mman.h>
#  include <unistd.h>
#endif

namespace omnivoice_streaming {

// Drop transient scratch pages back to the kernel without unmap'ing the
// underlying allocation. Safe on NULL/zero. Returns 0 on success or the
// errno from the syscall on failure (no-op platforms return 0).
inline int release_scratch(void * ptr, std::size_t bytes) {
    if (!ptr || bytes == 0) return 0;
#if defined(__linux__)
    // POSIX MADV_DONTNEED on Linux releases the mapped pages immediately;
    // next access faults them back in zeroed. Aligned to page boundaries
    // by the kernel, so passing unaligned ranges is safe but partially
    // ineffective (only fully-covered pages are released).
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return 0;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned = (addr + (uintptr_t) page - 1) & ~((uintptr_t) page - 1);
    if (aligned >= addr + bytes) return 0;
    const std::size_t aligned_bytes = (bytes - (aligned - addr)) & ~((std::size_t) page - 1);
    if (aligned_bytes == 0) return 0;
    return ::madvise(reinterpret_cast<void *>(aligned), aligned_bytes, MADV_DONTNEED);
#elif defined(__APPLE__)
    // MADV_FREE_REUSABLE tells the kernel the pages can be reclaimed AND
    // re-mapped without rezeroing if the process re-uses them quickly.
    // Best Darwin equivalent of Linux MADV_DONTNEED for transient scratch.
    return ::madvise(ptr, bytes, MADV_FREE_REUSABLE);
#else
    // Windows / unsupported: no-op. The transient peak stays — same as
    // the pre-W3-3 path — but correctness is unaffected.
    (void) ptr;
    (void) bytes;
    return 0;
#endif
}

// Mel codec streaming buffer flush hook. The underlying codec module
// (pipeline-codec.cpp) owns the buffer and emits hops to the on_chunk
// callback registered on ov_tts_params. This is a no-op today and
// exists so the streaming-path callers can land a `flush_hops(N)`
// invocation without depending on a specific codec-internal API; the
// implementation can be filled in inside pipeline-codec.cpp when the
// per-hop emission lands.
inline void flush_hops(int /*hops*/) {
    // Intentionally empty in the merged tree — the codec emits via the
    // on_chunk callback set on ov_tts_params, which already provides
    // fully-decoded hop-sized PCM chunks. Documented here so the call
    // site stays stable when an internal buffered path is added later.
}

// LM-conditioning prefix slot pool — real implementation (H2.c).
//
// Multiple concurrent TTS streams that target the same frozen-voice preset
// share an identical LM-conditioning prefix (instruct tokens + ref-audio
// tokens). Re-running the prefix forward on every call wastes ~28L of
// bidirectional attention; the slot pool caches a per-preset slot id that
// the pipeline can key into when it stores its computed prefix activations
// elsewhere (e.g. an external KV cache or per-slot tensor pool).
//
// The pool itself is intentionally minimal: it owns slot ids and the
// `preset_id` → `slot` mapping, plus LRU bookkeeping for eviction. The
// upstream caller (`pipeline-tts.cpp`) decides what to do with the slot
// number — what to store in it, when to invalidate, etc. The pool's only
// promise is "stable slot id per (alive) preset" with LRU eviction.
class PrefixSlotPool {
public:
    static constexpr int kDefaultCapacity = 8;

    explicit PrefixSlotPool(int capacity = kDefaultCapacity)
        : capacity_(capacity > 0 ? capacity : kDefaultCapacity)
        , entries_(new Entry[(std::size_t) (capacity > 0 ? capacity : kDefaultCapacity)]()) {}

    // Look up the slot for `preset_id`. If a slot is already reserved for
    // this preset, return its index and bump LRU. Otherwise, allocate a
    // fresh slot (evicting the LRU entry when the pool is full) and
    // return that. Empty/null preset ids return -1 — the bypass path.
    int acquire(const char * preset_id) {
        if (!preset_id || preset_id[0] == '\0') return -1;
        std::lock_guard<std::mutex> lk(mu_);
        const std::uint64_t now = tick();
        // 1. Reuse path.
        for (int i = 0; i < capacity_; ++i) {
            if (entries_[i].used && entries_[i].preset_id == preset_id) {
                entries_[i].last_used = now;
                ++entries_[i].refcount;
                return i;
            }
        }
        // 2. Allocate path. Look for an unused slot first, then evict the
        //    LRU entry with refcount==0. If every slot is pinned, return
        //    -1 (callers fall through to a non-slotted forward).
        int free_idx = -1;
        int evict_idx = -1;
        std::uint64_t oldest_tick = UINT64_MAX;
        for (int i = 0; i < capacity_; ++i) {
            if (!entries_[i].used) {
                free_idx = i;
                break;
            }
            if (entries_[i].refcount == 0 && entries_[i].last_used < oldest_tick) {
                oldest_tick = entries_[i].last_used;
                evict_idx = i;
            }
        }
        const int target = (free_idx >= 0) ? free_idx : evict_idx;
        if (target < 0) return -1;
        entries_[target].preset_id = preset_id;
        entries_[target].refcount = 1;
        entries_[target].last_used = now;
        entries_[target].used = true;
        return target;
    }

    // Release a slot acquired via `acquire`. Decrements the refcount;
    // when it reaches 0 the slot becomes evictable but its contents (the
    // caller's cached prefix activations) survive until another preset
    // takes the slot. Safe to call with a stale/invalid index — it's a
    // no-op when the slot id doesn't match.
    void release(int slot, const char * preset_id) {
        if (slot < 0 || slot >= capacity_) return;
        std::lock_guard<std::mutex> lk(mu_);
        Entry & e = entries_[slot];
        if (!e.used) return;
        if (preset_id && e.preset_id != preset_id) return;
        if (e.refcount > 0) --e.refcount;
    }

    // Invalidate the entry mapped to `preset_id`. The slot returns to the
    // free list on the next `acquire`. Returns true when an entry was
    // dropped.
    bool invalidate(const char * preset_id) {
        if (!preset_id) return false;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < capacity_; ++i) {
            if (entries_[i].used && entries_[i].preset_id == preset_id) {
                entries_[i].used = false;
                entries_[i].preset_id.clear();
                entries_[i].refcount = 0;
                entries_[i].last_used = 0;
                return true;
            }
        }
        return false;
    }

    // Drop every slot. Pinned slots (refcount > 0) are forced out — the
    // caller is responsible for not racing this against in-flight uses.
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < capacity_; ++i) {
            entries_[i].used = false;
            entries_[i].preset_id.clear();
            entries_[i].refcount = 0;
            entries_[i].last_used = 0;
        }
    }

    int capacity() const { return capacity_; }

    // Number of slots that currently hold a preset binding.
    int size() const {
        std::lock_guard<std::mutex> lk(mu_);
        int n = 0;
        for (int i = 0; i < capacity_; ++i) {
            if (entries_[i].used) ++n;
        }
        return n;
    }

private:
    struct Entry {
        std::string preset_id;
        int refcount = 0;
        std::uint64_t last_used = 0;
        bool used = false;
    };

    std::uint64_t tick() {
        // Monotonic logical clock — independent of wall time so eviction
        // works in lock-step regardless of clock skew. Atomic to keep
        // multi-acquire ordering stable.
        return ++clock_;
    }

    const int capacity_;
    std::unique_ptr<Entry[]> entries_;
    mutable std::mutex mu_;
    std::atomic<std::uint64_t> clock_{0};
};

// Process-wide default pool. The previous stub used a free function with
// no state; preserve the function for source compatibility and route it
// through the singleton pool.
inline PrefixSlotPool & default_prefix_slot_pool() {
    static PrefixSlotPool instance(PrefixSlotPool::kDefaultCapacity);
    return instance;
}

// Convenience facade: acquire a slot for `preset_id` from the process-wide
// pool. Returns -1 for empty/null preset ids or when every slot is pinned.
// Callers that want explicit lifecycle control should use the pool API
// directly via `default_prefix_slot_pool()`.
inline int prefix_slot(const char * preset_id) {
    return default_prefix_slot_pool().acquire(preset_id);
}

}  // namespace omnivoice_streaming
