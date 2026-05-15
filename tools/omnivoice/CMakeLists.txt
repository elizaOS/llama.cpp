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
)

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
    target_compile_options(omnivoice_lib PRIVATE
        -Wall -Wextra -Wshadow -Wconversion
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
# Built when LLAMA_BUILD_OMNIVOICE=ON (always, when the merged tree is
# enabled). The pre-W3-3 build script (`build-llama-cpp-dflash.mjs`) used
# to graft + generate this via `omnivoice-fuse/prepare.mjs`; the merged
# path produces it directly from in-fork source.
#
# Library file name: `libelizainference.{so,dylib,dll}` (no version
# suffix on Linux to keep dlopen() happy on the runtime side). Default
# visibility is `default` (not hidden) because every `eliza_inference_*`
# entry the JS loader needs is reachable through dlsym.
add_library(elizainference SHARED
    ${OMNIVOICE_CORE_SOURCES}
    ${OMNIVOICE_FFI_SOURCES})
target_compile_features(elizainference PUBLIC cxx_std_17)
target_compile_definitions(elizainference
    PRIVATE OMNIVOICE_BUILD ${OMNIVOICE_COMPILE_DEFINITIONS})
target_include_directories(elizainference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${OMNIVOICE_LLAMA_PRIVATE_INCLUDE_DIRS})
target_include_directories(elizainference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
# Link llama + mtmd. The transitive link pulls ggml + the parent
# build's backend selection in; we never call ggml_backend_*_init()
# from here directly (patch 0003 — backend_init_auto() consumes the
# shared pair or constructs its own as needed).
target_link_libraries(elizainference PUBLIC llama)
if(TARGET mtmd)
    target_link_libraries(elizainference PUBLIC mtmd)
endif()
set_target_properties(elizainference PROPERTIES
    OUTPUT_NAME              elizainference
    POSITION_INDEPENDENT_CODE ON)
# Apple re-exports llama symbols so the fused .dylib carries everything
# JS will dlopen() without a second handle. Mirrors the legacy fork-root
# graft block.
if(APPLE)
    target_link_options(elizainference PRIVATE
        "LINKER:-reexport_library,$<TARGET_FILE:llama>")
endif()
add_dependencies(elizainference omnivoice-version)

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
add_executable(omnivoice-dac-parity tests/dac-parity.cpp)
target_compile_features(omnivoice-dac-parity PRIVATE cxx_std_17)
target_include_directories(omnivoice-dac-parity PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src)
target_link_libraries(omnivoice-dac-parity PRIVATE ggml ggml-cpu ggml-base)

if(BUILD_TESTING)
    add_test(NAME omnivoice-dac-parity COMMAND omnivoice-dac-parity --no-real)
endif()
