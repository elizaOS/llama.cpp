# Session-op backend seam — design (NOT implemented)

The per-op backend seam (`backend-registry.h` + `<mod>-backend.h` +
`<mod>-backend-selector.cpp` + a chokepoint at the top of the FFI fn) is now in
place for the **one-shot** ops:

| modality | FFI fn                          | header / selector            | env key               | artifact dir       |
|----------|---------------------------------|------------------------------|-----------------------|--------------------|
| embed    | `eliza_inference_embed`         | `embed-backend.*`            | `ELIZA_EMBED_BACKEND` | `<bundle>/embedding/` |
| vision   | `eliza_inference_describe_image`| `vision-backend.*`           | `ELIZA_VISION_BACKEND`| `<bundle>/vision/` |
| asr      | `eliza_inference_asr_transcribe`| `asr-backend.*`              | `ELIZA_ASR_BACKEND`   | `<bundle>/asr/`    |
| tts      | `eliza_inference_tts_synthesize`| `tts-backend.*`              | `ELIZA_TTS_BACKEND`   | `<bundle>/tts/`    |
| eot      | `eliza_inference_llm_eot_score` | `eot-backend.*`              | `ELIZA_EOT_BACKEND`   | `<bundle>/eot/`    |

A one-shot op is stateless across calls: select → (delegate | fall through to
ggml) on every call. There is nothing to keep alive between calls, so the seam
is a single chokepoint at the top of the fn.

The **session** ops are different: `vad`, `wakeword`, `speaker`, `diariz` each
`_open` a native handle (`EliVad *`, `EliWakeword *`, `EliSpeaker *`,
`EliDiariz *`) that persists across many `_segment`/`_detect`/`_embed` calls and
is torn down with `_close`/`_reset`. The seam has to follow that lifecycle, not
re-select per call. This file records HOW to extend the seam to them. **None of
the below is implemented yet.**

## The shape of a session op (today, in-tree only)

Each session modality exposes, e.g. for VAD:

```c
EliVad * eliza_inference_vad_open(EliInferenceContext * ctx, /* params */, char ** out_error);
int      eliza_inference_vad_segment(EliVad * vad, const float * pcm, size_t n, /* out */, char ** out_error);
int      eliza_inference_vad_reset(EliVad * vad, char ** out_error);
void     eliza_inference_vad_close(EliVad * vad);
```

`EliVad` (and the wakeword/speaker/diariz equivalents) is the in-tree handle
struct defined in `eliza-inference-ffi.cpp`. Its in-tree fields stay exactly as
they are; the seam is **additive** — one extra pointer.

## Extending the seam to a session op

For each session modality `<mod>` (vad | wakeword | speaker | diariz):

### 1. A session factory interface — `<mod>-backend.h`

Mirror the one-shot factory's four common probes, but the forward methods mirror
the **session** ABI 1:1 instead of a single one-shot fn. The factory does NOT
own the handle struct; it produces and operates on an opaque backend-session:

```cpp
struct VadBackendFactory {
    virtual ~VadBackendFactory() = default;
    virtual const char * name() const = 0;
    virtual bool         available() const = 0;
    virtual bool         can_serve(const char * bundle_dir) const = 0;  // probes <bundle>/vad/
    virtual int          preference_rank() const { return 0; }

    // Lifecycle, mirroring the FFI session ABI 1:1. The factory returns an
    // opaque backend-session pointer it owns; the FFI stashes it on the Eli*
    // handle. A NULL return + *out_error is a hard open failure.
    virtual void * open(EliInferenceContext * ctx, /* same params as eliza_inference_vad_open */,
                        char ** out_error) = 0;
    virtual int   segment(void * session, const float * pcm, size_t n, /* out */, char ** out_error) = 0;
    virtual int   reset(void * session, char ** out_error) = 0;
    virtual void  close(void * session) = 0;
};
```

Plus the same free-functions as the one-shot seam:
`vad_backend_register`, `vad_backend_register_builtins` (EMPTY for now — no
LiteRT session backend exists), `vad_backend_select(bundle_dir, out_error)`,
backed by a `eliza_backend::Registry<VadBackendFactory>` in
`<mod>-backend-selector.cpp` with env keys `ELIZA_VAD_BACKEND` → `ELIZA_BACKEND`
and modality `"vad"`. Artifact probe dir `<bundle>/vad/` (resp. `wakeword/`,
`speaker/`, `diariz/`).

### 2. A backend-session pointer on the Eli* handle

The selection happens ONCE, at `_open`, not per call. Add one field to the
in-tree handle struct:

```cpp
struct EliVad {
    /* ... existing in-tree fields, unchanged ... */

    /* Backend seam (additive). When non-null, this handle is served by an
     * accelerator backend and every op delegates to it; the in-tree fields
     * above are then unused. When null, the in-tree ggml path owns the handle. */
    VadBackendFactory * be         = nullptr;  // the factory that opened be_session
    void *              be_session = nullptr;  // factory-owned backend session
};
```

### 3. Select at `_open`

In `eliza_inference_vad_open`, after the existing arg validation and before the
in-tree handle is built:

```cpp
char * be_error = nullptr;
VadBackendFactory * be = vad_backend_select(llm_backend_context_bundle_dir(ctx), &be_error);
if (be_error) { eliza_set_error(out_error, std::string(be_error)); std::free(be_error);
                return /* NULL handle */; }
if (be) {
    void * sess = be->open(ctx, /* params */, out_error);
    if (!sess) return /* NULL handle — open failed, out_error already set */;
    EliVad * h = new EliVad();
    h->be = be;
    h->be_session = sess;
    return h;
}
/* else: fall through and build the in-tree handle exactly as today. */
```

### 4. A guard at the TOP of each `_segment` / `_reset` / `_close`

Each per-call op checks the backend pointer and delegates before touching any
in-tree state:

```cpp
int eliza_inference_vad_segment(EliVad * vad, const float * pcm, size_t n, /* out */, char ** out_error) {
    if (!vad) { /* invalid-arg as today */ }
    if (vad->be) {                                   // <-- guard
        return vad->be->segment(vad->be_session, pcm, n, /* out */, out_error);
    }
    /* ... existing in-tree ggml segment body, unchanged ... */
}

void eliza_inference_vad_close(EliVad * vad) {
    if (!vad) return;
    if (vad->be) { vad->be->close(vad->be_session); delete vad; return; }  // <-- guard
    /* ... existing in-tree teardown, then delete vad ... */
}
```

`_reset` follows the same guard pattern.

## Why this shape (vs. re-selecting per call)

- **Selection is per-session, not per-call.** A session's backend is fixed at
  `_open`; you cannot have `_segment` cross from the ggml path to LiteRT mid
  session because the KV/feature state lives in the (in-tree OR backend)
  session, not on the FFI boundary. The one pointer captures that binding.
- **Hard-fail localizes to `_open`.** A bundle-invalid override surfaces once,
  where the caller is already prepared to handle a NULL handle, instead of on
  every `_segment`.
- **Additive + inert.** With no session backend registered (the case today),
  `_open`'s `select()` returns nullptr, `be`/`be_session` stay null, and every
  guard is a no-op — the in-tree path is byte-for-byte unchanged. Same inert-by
  -default contract as the one-shot seam.

## Status

- One-shot seam: embed (with a LiteRT builtin), vision/asr/tts/eot (inert,
  no builtin) — **done**.
- Session seam (vad/wakeword/speaker/diariz): **not implemented.** No
  `<mod>-backend.{h,cpp}`, no handle field, no `_open` select, no per-call
  guards exist yet. This file is the spec for when a session backend lands.
