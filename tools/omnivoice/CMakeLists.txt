# tools/omnivoice/CMakeLists.txt
#
# Merged omnivoice.cpp subtree, hosted inside the elizaOS llama.cpp fork.
# Built only when LLAMA_BUILD_OMNIVOICE=ON (see the top-level CMakeLists.txt
# guard). The intent is to share llama.cpp's already-built ggml/ggml-base
# /ggml-<backend> targets — DO NOT add_subdirectory(ggml) from here.
#
# Patch series (see ../../llama.cpp-omnivoice-merge/PLAN.md):
#   0001  add LLAMA_BUILD_OMNIVOICE option       (landed)
#   0002  vendor src/, tools/, include/, tests/  (this patch)
#   0003  replace backend.h wedge to share the
#         llama_context backend pair             (deferred)
#
# Backend wiring in this patch is intentionally minimal: omnivoice_lib
# links PUBLIC against llama, which transitively brings in ggml +
# whichever ggml-<backend> targets the parent build selected. Patch 0003
# is where backend.h is rewritten to consume the llama_context backend
# pair ; until then, the wedge here is "compile and link against llama,
# trust the parent build's backend selection".

# version.h: embed the submodule's git commit hash into every binary.
# Mirrors the original omnivoice.cpp behaviour ; the cmake script only
# rewrites version.h when the hash actually changes, so a clean build is
# the only one that pays this cost.
set(OMNIVOICE_VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(omnivoice-version ALL
    COMMAND "${CMAKE_COMMAND}"
            "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DOUTPUT=${OMNIVOICE_VERSION_OUTPUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${OMNIVOICE_VERSION_OUTPUT}"
    COMMENT "omnivoice: refreshing version.h"
)

# Audio tokenizer tensor names can exceed the default GGML_MAX_NAME of 64.
# Mirrors the original omnivoice.cpp CMakeLists.txt. Apply GGML_MAX_NAME to the
# linked ggml/llama targets too; ggml parses tensor names during GGUF load, so
# keeping this only on omnivoice translation units leaves tokenizer GGUFs
# unreadable.
set(OMNIVOICE_GGML_COMPILE_DEFINITIONS GGML_MAX_NAME=128)
foreach(_omnivoice_ggml_name_target
        ggml ggml-base ggml-cpu ggml-blas ggml-metal ggml-vulkan ggml-cuda
        llama common)
    if(TARGET ${_omnivoice_ggml_name_target})
        target_compile_definitions(${_omnivoice_ggml_name_target}
            PUBLIC ${OMNIVOICE_GGML_COMPILE_DEFINITIONS})
    endif()
endforeach()
# LLAMA_BUILD_OMNIVOICE and ELIZA_FUSE_OMNIVOICE stay scoped to omnivoice
# targets and llama-server. The legacy ELIZA_FUSE_OMNIVOICE source guard remains
# as a deprecation runway while the canonical merged path is LLAMA_BUILD_OMNIVOICE.
#
# W3-3 — OmniVoice → llama.cpp literal merge (canonical path):
#   * Both LLAMA_BUILD_OMNIVOICE and ELIZA_FUSE_OMNIVOICE are defined on
#     every target so source-level `#ifdef ELIZA_FUSE_OMNIVOICE` blocks
#     (e.g. `tools/server/server.cpp`'s /v1/audio/speech route handler)
#     continue to compile under the canonical merged path. The legacy
#     out-of-tree graft (`omnivoice/` at fork root) is being removed in
#     this wave; ELIZA_FUSE_OMNIVOICE stays as a source-level alias for
#     one release as a deprecation runway.
set(OMNIVOICE_COMPILE_DEFINITIONS
    GGML_MAX_NAME=128
    LLAMA_BUILD_OMNIVOICE
    ELIZA_FUSE_OMNIVOICE)
set(OMNIVOICE_LLAMA_PRIVATE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

# Sources shared by both the static core and the shared ABI-export library.
set(OMNIVOICE_CORE_SOURCES
    src/omnivoice.cpp
    src/pipeline-codec.cpp
    src/pipeline-tts.cpp
)

# Eliza FFI bridge sources — produce the `eliza_inference_*` symbol table
# consumed by JS/Bun (`plugin-local-inference/services/voice/ffi-bindings.ts`),
# Capacitor (mobile), Electrobun (desktop). The bridge wraps omnivoice +
# llama + mtmd into a single ABI-stable C surface.
set(OMNIVOICE_FFI_SOURCES
    src/eliza-inference-ffi.cpp
    # Multi-runtime streaming-LLM backend seam (cutover plan M3). The selector
    # is always compiled; it is inert until a -DELIZA_ENABLE_* accelerator
    # backend below registers itself, so the default build keeps the in-tree
    # llama.cpp path.
    src/llm-backend-selector.cpp
    # Per-op backend seams (cutover M3+). Each modality's selector reuses the
    # shared eliza_backend::Registry (backend-registry.h) and is inert until a
    # gated backend registers — so the default build keeps the ggml path per-op.
    src/embed-backend-selector.cpp
    src/vision-backend-selector.cpp
    src/asr-backend-selector.cpp
    src/tts-backend-selector.cpp
    src/eot-backend-selector.cpp
)

# Vendored standalone voice-classifier forward graphs (pure scalar C, no
# ggml/llama dependency; they carry their own GGUF reader). These back
# the fused wake-word (ABI v5), speaker-encoder, diarizer (ABI v6), and
# Silero VAD (ABI v7) surfaces so the whole voice pipeline runs through
# one libelizainference handle instead of separate
# libvoice_classifier / libwakeword / libsilero_vad .so libs. Vendored
# from packages/native/plugins/{wakeword-cpp,voice-classifier-cpp,
# silero-vad-cpp} by the Phase-1 fused-voice consolidation.
set(OMNIVOICE_VOICE_CLASSIFIER_SOURCES
    src/voice-classifiers/wakeword/wakeword_runtime.c
    src/voice-classifiers/wakeword/wakeword_melspec.c
    src/voice-classifiers/wakeword/wakeword_window.c
    src/voice-classifiers/voice_classifier/voice_speaker.c
    src/voice-classifiers/voice_classifier/voice_speaker_distance.c
    src/voice-classifiers/voice_classifier/voice_diarizer.c
    src/voice-classifiers/voice_classifier/voice_mel_features.c
    src/voice-classifiers/voice_classifier/voice_gguf_loader.c
    src/voice-classifiers/voice_classifier/voice_gguf_tensors.c
    src/voice-classifiers/vad/silero_vad_runtime.c
    src/voice-classifiers/vad/silero_vad_state.c
    src/voice-classifiers/vad/silero_vad_resample.c
)

# Include roots so the vendored TUs resolve their own headers
# (`voice_classifier/voice_classifier.h`, `voice_gguf_loader.h`,
# `wakeword/wakeword.h`, `wakeword_internal.h`, `silero_vad/silero_vad.h`,
# `silero_vad_state.h`) and so the FFI bridge can #include the public ABI
# headers via their vendored relative paths.
set(OMNIVOICE_VOICE_CLASSIFIER_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voice-classifiers/wakeword
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voice-classifiers/wakeword/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voice-classifiers/voice_classifier
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voice-classifiers/voice_classifier/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voice-classifiers/vad
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voice-classifiers/vad/include
)

# Static archive of the vendored scalar-C forward graphs. Plain C11, no
# llama/ggml link. PIC so it folds into the elizainference SHARED lib.
add_library(eliza_voice_classifiers STATIC ${OMNIVOICE_VOICE_CLASSIFIER_SOURCES})
set_target_properties(eliza_voice_classifiers PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(eliza_voice_classifiers PUBLIC
    ${OMNIVOICE_VOICE_CLASSIFIER_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(eliza_voice_classifiers PRIVATE /W3 /wd4100)
    # The vendored voice-classifier C sources include <sys/mman.h> and
    # <unistd.h> (POSIX) to mmap their GGUF files read-only and use
    # open/close/read. MSVC ships neither header; this compat dir provides
    # thin shims (mmap over Win32 file mapping; unistd over <io.h>) so the
    # static lib — folded into the fused elizainference DLL — builds on
    # Windows. Prepended via BEFORE so <sys/mman.h>/<unistd.h> resolve here.
    target_include_directories(eliza_voice_classifiers BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/compat/msvc)
else()
    target_compile_options(eliza_voice_classifiers PRIVATE
        -O3 -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(eliza_voice_classifiers PUBLIC m)
endif()

# omnivoice_lib (STATIC) — the core. Tools link against this to pull
# every pipeline_*/backend_* symbol without going through the public ABI.
# Public include directories let consumers (omnivoice-tts, omnivoice-codec,
# tests/abi-c) #include "omnivoice.h" / "pipeline-tts.h" etc. directly.
add_library(omnivoice_lib STATIC ${OMNIVOICE_CORE_SOURCES})
target_compile_features(omnivoice_lib PUBLIC cxx_std_17)
target_compile_definitions(omnivoice_lib
    PUBLIC  OMNIVOICE_STATIC
    PRIVATE ${OMNIVOICE_COMPILE_DEFINITIONS})
target_include_directories(omnivoice_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OMNIVOICE_LLAMA_PRIVATE_INCLUDE_DIRS})
# llama transitively pulls ggml + the backend pair chosen at parent
# configure time. The omnivoice tree never touches backend selection
# directly ; patch 0003 will rewrite backend.h to consume the
# llama_context backend pair explicitly.
target_link_libraries(omnivoice_lib PUBLIC llama)
add_dependencies(omnivoice_lib omnivoice-version)

if(MSVC)
    target_compile_options(omnivoice_lib PRIVATE
        /W4 /wd4100 /wd4505)
else()
    # -Wshadow / -Wconversion intentionally omitted: omnivoice sources
    # include llama's private headers (src/llama-arch.h, llama-adapter.h)
    # and the public ggml-backend.h, none of which are shadow- or
    # conversion-clean upstream. Under -DLLAMA_FATAL_WARNINGS=ON those
    # warnings escalate to -Werror and break the 3rd-party CI lane.
    target_compile_options(omnivoice_lib PRIVATE
        -Wall -Wextra
        -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
endif()

# omnivoice (SHARED) — opt-in ABI consumer library. Hidden visibility,
# exports only the OV_API-marked symbols. Separate target name from
# omnivoice_lib so the static path used by the bundled tools is never
# affected.
if(OMNIVOICE_SHARED)
    add_library(omnivoice SHARED ${OMNIVOICE_CORE_SOURCES})
    target_compile_features(omnivoice PUBLIC cxx_std_17)
    target_compile_definitions(omnivoice
        PRIVATE OMNIVOICE_BUILD ${OMNIVOICE_COMPILE_DEFINITIONS})
    target_include_directories(omnivoice PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${OMNIVOICE_LLAMA_PRIVATE_INCLUDE_DIRS})
    target_link_libraries(omnivoice PUBLIC llama)
    set_target_properties(omnivoice PROPERTIES
        C_VISIBILITY_PRESET       hidden
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)
    add_dependencies(omnivoice omnivoice-version)
endif()

# elizainference (SHARED) — the canonical fused FFI library consumed by
# the Eliza runtime (JS / Bun via bun:ffi, Capacitor on mobile, Electrobun
# on desktop). Exports BOTH `eliza_inference_*` (the FFI ABI declared in
# include/eliza-inference-ffi.h) AND `omnivoice` / `llama` symbol families
# so the same `dlopen()`-ed handle serves text inference + speech
# synthesis + ASR + VAD.
#
# Built when LLAMA_BUILD_OMNIVOICE=ON AND the mtmd target exists. The FFI
# wraps mtmd_* for the ASR (audio→text) pipeline, so it cannot be built
# without it. iOS XCFramework configurations (LLAMA_BUILD_TOOLS=OFF,
# LLAMA_BUILD_MTMD=OFF) skip this target — the Eliza iOS runtime loads
# llama / omnivoice without ASR until a build with mtmd is enabled.
#
# Library file name: `libelizainference.{so,dylib,dll}` (no version
# suffix on Linux to keep dlopen() happy on the runtime side). Default
# visibility is `default` (not hidden) because every `eliza_inference_*`
# entry the JS loader needs is reachable through dlsym.
# ELIZA_ENABLE_VISION — ABI v9 mmproj vision describe
# (`eliza_inference_describe_image` / `eliza_inference_vision_supported`). When
# ON, the FFI bridge compiles the mtmd vision-describe path (reusing the mtmd
# machinery already linked for ASR) and `_vision_supported()` returns 1. OFF
# (default) leaves it stubbed (`_vision_supported() == 0`,
# `_describe_image` -> ELIZA_ERR_NOT_IMPLEMENTED) so the IMAGE_DESCRIPTION
# handler falls back to the libllama mtmd path. Mirrors the shim's
# `-DELIZA_ENABLE_VISION=1` env flag. Defaults ON when LLAMA_BUILD_MTMD is set
# (the fused build links mtmd anyway), opt out with -DELIZA_ENABLE_VISION=OFF.
option(ELIZA_ENABLE_VISION "Build the fused mmproj vision-describe ABI (v9)" ON)

# ELIZA_ENABLE_LITERT — compile the LiteRT-LM in-process streaming-LLM backend
# (cutover plan M4 — Android NPU: Tensor / Qualcomm QNN / MediaTek NeuroPilot).
# OFF by default: the selector registers no LiteRT backend and the streaming-LLM
# pipe keeps the in-tree llama.cpp path. ON requires the LiteRT-LM SDK
# (ELIZA_LITERT_SDK_DIR) — a host/device cross-build concern, not the Linux CI
# default. See docs/multi-backend-ffi-seam.md.
option(ELIZA_ENABLE_LITERT "Build the LiteRT C-API per-op backends, e.g. embed (M4)" OFF)

# ELIZA_ENABLE_LITERT_LM — the streaming-LLM backend on the heavier LiteRT-LM
# Engine SDK (litert::lm), separate from the LiteRT C runtime above. OFF until
# that SDK is built; point -DELIZA_LITERT_LM_SDK_DIR / -DELIZA_LITERT_LM_LIBS at it.
option(ELIZA_ENABLE_LITERT_LM "Build the LiteRT-LM in-process streaming-LLM backend" OFF)

# ELIZA_ENABLE_MLX — compile the CoreML/MLX in-process streaming-LLM backend
# (cutover plan M5 — Apple Silicon). OFF by default; ON is Apple-only and
# requires the MLX / CoreML toolchain. See docs/multi-backend-ffi-seam.md.
option(ELIZA_ENABLE_MLX "Build the CoreML/MLX in-process LLM backend (M5)" OFF)

if(TARGET mtmd)
    add_library(elizainference SHARED
        ${OMNIVOICE_CORE_SOURCES}
        ${OMNIVOICE_FFI_SOURCES})
    target_compile_features(elizainference PUBLIC cxx_std_17)
    target_compile_definitions(elizainference
        PRIVATE OMNIVOICE_BUILD ${OMNIVOICE_COMPILE_DEFINITIONS})
    if(ELIZA_ENABLE_VISION)
        target_compile_definitions(elizainference PRIVATE ELIZA_ENABLE_VISION)
    endif()
    target_include_directories(elizainference PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}
        ${OMNIVOICE_LLAMA_PRIVATE_INCLUDE_DIRS})
    target_include_directories(elizainference PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        # common/ — speculative.h, sampling.h, common.h for the same-file MTP
        # speculative-decode engine wired into the streaming-LLM text path
        # (ABI v8). The DRAFT_MTP driver reuses common/speculative.cpp.
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
    # Link llama + mtmd. The transitive link pulls ggml + the parent
    # build's backend selection in; we never call ggml_backend_*_init()
    # from here directly (patch 0003 — backend_init_auto() consumes the
    # shared pair or constructs its own as needed).
    #
    # eliza_voice_classifiers folds the vendored scalar-C wake-word /
    # speaker / diarizer forward graphs into the fused .so so the
    # `eliza_inference_{wakeword,speaker,diariz}_*` ABIs resolve from
    # the same dlopen()-ed handle. PRIVATE: the standalone C ABI
    # (`wakeword_*` / `voice_speaker_*` / `voice_diarizer_*`) is an
    # implementation detail behind the `eliza_inference_*` surface and
    # must NOT be re-exported.
    target_link_libraries(elizainference PUBLIC llama mtmd)
    # llama-common (PRIVATE): the same-file MTP speculative-decode engine
    # in the streaming-LLM text path (ABI v8) reuses common/speculative.cpp,
    # common/sampling.cpp, and common/common.cpp. PRIVATE so the common ABI
    # is not re-exported from the fused .so (only eliza_inference_* + llama).
    target_link_libraries(elizainference PRIVATE eliza_voice_classifiers llama-common)
    # Kokoro-82M TTS (ABI v10): fold kokoro_lib into the fused .so so the
    # mobile Kokoro path synthesizes in-process (iOS / Google Play forbid the
    # local-TCP `llama-server /v1/audio/speech` route). kokoro_lib carries its
    # own GGUF reader + iSTFT decoder. PRIVATE — the `eliza_kokoro::*` C++ API
    # is an implementation detail behind the `eliza_inference_kokoro_*` surface.
    if(TARGET kokoro_lib)
        target_link_libraries(elizainference PRIVATE kokoro_lib)
        target_include_directories(elizainference PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../kokoro/include)
        target_compile_definitions(elizainference PRIVATE ELIZA_ENABLE_KOKORO)
    endif()
    # ── Multi-runtime streaming-LLM accelerator backends (cutover M4/M5) ────
    # The M3 selector (src/llm-backend-selector.cpp) is always compiled in via
    # OMNIVOICE_FFI_SOURCES. These accelerator backends each link an external
    # SDK, so they are opt-in. When a gate is OFF its source is not compiled,
    # the selector's `#ifdef`-guarded factory declaration + registration drop
    # out, and the streaming-LLM pipe keeps the in-tree llama.cpp path — so the
    # default desktop/CI build is byte-for-byte the pre-seam behavior.
    if(ELIZA_ENABLE_LITERT)
        # LiteRT C-API per-op backends (embed today; vision/etc. as artifacts
        # ship). SDK = the LiteRT C runtime (github.com/google-ai-edge/LiteRT,
        # libLiteRt.so + the GPU/NPU delegate). Point at a built SDK with
        # -DELIZA_LITERT_SDK_DIR=<dir> and link with -DELIZA_LITERT_LIBS=LiteRt.
        target_sources(elizainference PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/litert-embed-backend.cpp)
        target_compile_definitions(elizainference PRIVATE ELIZA_ENABLE_LITERT)
        if(ELIZA_LITERT_SDK_DIR)
            target_include_directories(elizainference PRIVATE ${ELIZA_LITERT_SDK_DIR}/include)
            target_link_directories(elizainference PRIVATE ${ELIZA_LITERT_SDK_DIR}/lib)
        endif()
        if(ELIZA_LITERT_LIBS)
            target_link_libraries(elizainference PRIVATE ${ELIZA_LITERT_LIBS})
        endif()
    endif()
    if(ELIZA_ENABLE_LITERT_LM)
        # The streaming-LLM backend needs the heavier LiteRT-LM Engine SDK
        # (litert::lm, github.com/google-ai-edge/LiteRT-LM) — separate from the
        # LiteRT C runtime above. Point at it with -DELIZA_LITERT_LM_SDK_DIR /
        # -DELIZA_LITERT_LM_LIBS.
        target_sources(elizainference PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/litert-backend.cpp)
        target_compile_definitions(elizainference PRIVATE ELIZA_ENABLE_LITERT_LM)
        if(ELIZA_LITERT_LM_SDK_DIR)
            target_include_directories(elizainference PRIVATE ${ELIZA_LITERT_LM_SDK_DIR}/include)
            target_link_directories(elizainference PRIVATE ${ELIZA_LITERT_LM_SDK_DIR}/lib)
        endif()
        if(ELIZA_LITERT_LM_LIBS)
            target_link_libraries(elizainference PRIVATE ${ELIZA_LITERT_LM_LIBS})
        endif()
    endif()
    if(ELIZA_ENABLE_MLX)
        if(NOT APPLE)
            message(FATAL_ERROR
                "ELIZA_ENABLE_MLX requires an Apple host (CoreML/MLX are Apple-only).")
        endif()
        target_sources(elizainference PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/mlx-coreml-backend.mm)
        target_compile_definitions(elizainference PRIVATE ELIZA_ENABLE_MLX)
        # MLX C API (ml-explore/mlx-c) via -DELIZA_MLX_C_DIR / -DELIZA_MLX_LIBS,
        # plus the system CoreML / Metal / Foundation frameworks.
        if(ELIZA_MLX_C_DIR)
            target_include_directories(elizainference PRIVATE ${ELIZA_MLX_C_DIR}/include)
            target_link_directories(elizainference PRIVATE ${ELIZA_MLX_C_DIR}/lib)
        endif()
        if(ELIZA_MLX_LIBS)
            target_link_libraries(elizainference PRIVATE ${ELIZA_MLX_LIBS})
        endif()
        target_link_libraries(elizainference PRIVATE
            "-framework Foundation" "-framework CoreML" "-framework Metal")
    endif()
    set_target_properties(elizainference PROPERTIES
        OUTPUT_NAME              elizainference
        POSITION_INDEPENDENT_CODE ON
        # On Windows/MSVC symbols are NOT exported by default (unlike the Unix
        # `default` visibility this target relies on), so the extern "C"
        # eliza_inference_* / ov_* FFI surface the JS loader dlsym()s would be
        # absent from the DLL (bun:ffi: "Symbol eliza_inference_abi_version not
        # found"). Auto-generate an export table for every symbol so Windows
        # matches the Unix default-export behavior the runtime assumes.
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # Apple re-exports llama symbols so the fused .dylib carries everything
    # JS will dlopen() without a second handle. Mirrors the legacy fork-root
    # graft block.
    if(APPLE)
        target_link_options(elizainference PRIVATE
            "LINKER:-reexport_library,$<TARGET_FILE:llama>")
    endif()
    add_dependencies(elizainference omnivoice-version)
endif()

# llama-server wiring — mount the OmniVoice POST /v1/audio/speech route
# onto the same process that serves /v1/chat/completions + the DFlash
# speculative loop. Source-level guard is `#ifdef ELIZA_FUSE_OMNIVOICE` in
# `tools/server/server.cpp` (kept for the deprecation runway — both
# defines are set above).
#
# This call is the merged equivalent of the pre-W3-3 root CMakeLists.txt
# block (`ELIZA-OMNIVOICE-FUSION-GRAFT-V1`) that ran at the build-script
# graft layer. The target llama-server is declared by the regular
# add_subdirectory(tools) — by the time we get here it already exists.
if(TARGET llama-server)
    target_link_libraries(llama-server PRIVATE omnivoice_lib)
    target_include_directories(llama-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OMNIVOICE_LLAMA_PRIVATE_INCLUDE_DIRS})
    target_compile_definitions(llama-server PRIVATE ${OMNIVOICE_COMPILE_DEFINITIONS})
    # H2.c — the OmniVoice /v1/audio/speech preset-aware route used to live
    # inline in tools/server/server.cpp. It now lives in committed source at
    # tools/omnivoice/src/server-mount.cpp; compile it directly into the
    # llama-server target so the eliza_omnivoice::* symbols resolve at link
    # time without needing a separate static library.
    target_sources(llama-server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/server-mount.cpp)
endif()

# omnivoice-tts : full TTS pipeline CLI.
add_executable(omnivoice-tts tools/omnivoice-tts.cpp)
target_compile_definitions(omnivoice-tts PRIVATE ${OMNIVOICE_COMPILE_DEFINITIONS})
target_link_libraries(omnivoice-tts PRIVATE omnivoice_lib)

# omnivoice-codec : standalone codec CLI (codes <-> WAV).
add_executable(omnivoice-codec tools/omnivoice-codec.cpp)
target_compile_definitions(omnivoice-codec PRIVATE ${OMNIVOICE_COMPILE_DEFINITIONS})
target_link_libraries(omnivoice-codec PRIVATE omnivoice_lib)

# omnivoice-test-abi-c : pure C99 smoke test that locks in the public ABI
# contract. Compiles omnivoice.h with a C compiler under -Wall -Werror
# -pedantic and links against the static lib. Never loads a model ; a
# failure means the public API regressed. Built by default so a regression
# breaks the main build, not just an opt-in target.
add_executable(omnivoice-test-abi-c tests/abi-c.c)
set_target_properties(omnivoice-test-abi-c PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)
if(MSVC)
    target_compile_options(omnivoice-test-abi-c PRIVATE /W4 /WX)
else()
    target_compile_options(omnivoice-test-abi-c PRIVATE -Wall -Werror -pedantic)
endif()
target_include_directories(omnivoice-test-abi-c PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(omnivoice-test-abi-c PRIVATE omnivoice_lib)

# omnivoice-dac-parity : focused CPU numerical parity harness for
# dac_conv_t1d's ggml_conv_transpose_1d migration (elizaOS/eliza#7660).
# The default run uses deterministic synthetic DAC block shapes; pass
# `--gguf <omnivoice-tokenizer.gguf>` to exercise real conv_t1 weights.
#
# The parity harness calls ggml_backend_cpu_init / ggml_backend_cpu_set_n_threads
# directly, so it can only be built when the ggml-cpu CMake target is present.
# On some cross-compile configurations (Android NDK in particular) the ggml-cpu
# target is not defined, and the executable would fail to link. Gate the whole
# target on ggml-cpu availability so those builds skip it cleanly.
if(TARGET ggml-cpu)
    add_executable(omnivoice-dac-parity tests/dac-parity.cpp)
    target_compile_features(omnivoice-dac-parity PRIVATE cxx_std_17)
    target_include_directories(omnivoice-dac-parity PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
        ${CMAKE_SOURCE_DIR}/ggml/src)
    target_link_libraries(omnivoice-dac-parity PRIVATE ggml ggml-cpu)
    if(TARGET ggml-base)
        target_link_libraries(omnivoice-dac-parity PRIVATE ggml-base)
    endif()

    if(BUILD_TESTING)
        add_test(NAME omnivoice-dac-parity COMMAND omnivoice-dac-parity --no-real)
    endif()
endif()
