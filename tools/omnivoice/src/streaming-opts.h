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

#include <cstddef>
#include <cstdint>

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

// LM-conditioning prefix slot lookup. Returns the slot index the given
// frozen-voice preset id has been pinned to, or -1 when no slot is
// reserved (callers fall through to the normal forward).
//
// Today the slot pool is not yet allocated — the function always
// returns -1. This keeps the calling convention stable so the slot
// pool can land without touching every call site in pipeline-tts.cpp.
inline int prefix_slot(const char * /*preset_id*/) {
    return -1;
}

}  // namespace omnivoice_streaming
