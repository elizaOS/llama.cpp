# tools/kokoro/CMakeLists.txt
#
# Kokoro-82M (StyleTTS-2 + iSTFTNet) standalone TTS subtree, hosted inside
# the elizaOS llama.cpp fork. Built only when LLAMA_BUILD_KOKORO=ON
# (see the top-level CMakeLists.txt guard).
#
# Produces:
#   kokoro_lib            — static library with the inference, iSTFT, phoneme,
#                           and server-mount handler translation units.
#   kokoro-tts            — standalone CLI harness (J2 verification target).
#
# The library transitively gets ggml + llama from the parent build's
# target set. Do not add_subdirectory(ggml) here.

set(KOKORO_PUBLIC_HEADERS
    include/kokoro.h
    include/kokoro-istft.h
    include/kokoro-phonemes.h
    include/kokoro-server-mount.h
    include/kokoro-tensor-names.h
    include/kokoro-layers.h
    include/kokoro-predictor.h)

set(KOKORO_CORE_SOURCES
    src/kokoro.cpp
    src/kokoro-istft.cpp
    src/kokoro-phonemes.cpp
    src/kokoro-predictor.cpp
    src/kokoro-generator.cpp
    src/kokoro-decoder.cpp)

add_library(kokoro_lib STATIC
    ${KOKORO_CORE_SOURCES}
    ${KOKORO_PUBLIC_HEADERS})

target_include_directories(kokoro_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ggml + llama are already configured by the parent build; pulling them via
# target_link_libraries gives us the include paths and the link line.
target_link_libraries(kokoro_lib PUBLIC ggml)
if(TARGET ggml-base)
    target_link_libraries(kokoro_lib PUBLIC ggml-base)
endif()
if(TARGET ggml-cpu)
    target_link_libraries(kokoro_lib PUBLIC ggml-cpu)
endif()

target_compile_features(kokoro_lib PUBLIC cxx_std_17)

# kokoro-layers.h routes the Conv1d/ConvTranspose1d/Linear/LSTM hot loops
# through the Accelerate BLAS (im2col + sgemm/sgemv) on Apple platforms —
# the scalar loops are ~2 orders of magnitude too slow for real-time TTS.
# On every other platform the header's portable-fast path (internal
# std::thread pool + NEON on aarch64) is used, so it needs the threads lib.
if(APPLE)
    find_library(KOKORO_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(kokoro_lib PUBLIC ${KOKORO_ACCELERATE_FRAMEWORK})
else()
    find_package(Threads REQUIRED)
    target_link_libraries(kokoro_lib PUBLIC Threads::Threads)
endif()

# Real G2P via libespeak-ng. When present, kokoro_phonemize() drives
# espeak_TextToPhonemes() (en-us IPA) and maps codepoints to Kokoro vocab ids,
# reproducing the reference token sequence. When absent, the text path falls
# back to the degraded ASCII grapheme mapping — so an espeak-less build reports
# `eliza_inference_kokoro_g2p_kind() == ELIZA_KOKORO_G2P_ASCII` and the TS voice
# layer supplies espeak-ng (WASM) IPA through
# `eliza_inference_kokoro_synthesize_ipa()` (ABI v14, #11776) to stay
# intelligible. Override the search with -DKOKORO_ESPEAK_ROOT=<prefix> (Homebrew).
option(KOKORO_ENABLE_ESPEAK "Link libespeak-ng for real Kokoro G2P" ON)
if(KOKORO_ENABLE_ESPEAK)
    find_path(ESPEAK_NG_INCLUDE_DIR espeak-ng/speak_lib.h
        HINTS ${KOKORO_ESPEAK_ROOT} /opt/homebrew /usr/local /usr
        PATH_SUFFIXES include)
    find_library(ESPEAK_NG_LIBRARY NAMES espeak-ng
        HINTS ${KOKORO_ESPEAK_ROOT} /opt/homebrew /usr/local /usr
        PATH_SUFFIXES lib lib64)
    if(ESPEAK_NG_INCLUDE_DIR AND ESPEAK_NG_LIBRARY)
        target_include_directories(kokoro_lib PRIVATE ${ESPEAK_NG_INCLUDE_DIR})
        target_link_libraries(kokoro_lib PRIVATE ${ESPEAK_NG_LIBRARY})
        target_compile_definitions(kokoro_lib PRIVATE KOKORO_USE_ESPEAK)
        message(STATUS "Kokoro G2P: libespeak-ng found (${ESPEAK_NG_LIBRARY}) — real IPA path enabled")
    else()
        message(STATUS "Kokoro G2P: libespeak-ng not found — ASCII grapheme fallback; g2p_kind() reports ASCII so the TS layer feeds IPA via kokoro_synthesize_ipa (#11776)")
    endif()
endif()

# Standalone CLI harness — host-only (required by J2 verification,
# tools/voice-kokoro/). Skipped on iOS: there an executable is implicitly a
# MACOSX_BUNDLE app, so `install(TARGETS ... RUNTIME)` fails configure with
# "no BUNDLE DESTINATION". The fused mobile slice only needs kokoro_lib folded
# into libelizainference, never the CLI.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    add_executable(kokoro-tts tools/kokoro-tts.cpp)
    target_link_libraries(kokoro-tts PRIVATE kokoro_lib)
    install(TARGETS kokoro-tts RUNTIME)

    add_executable(kokoro-stage-dump tools/kokoro-stage-dump.cpp)
    target_link_libraries(kokoro-stage-dump PRIVATE kokoro_lib)

    add_executable(kokoro-decoder-test tools/kokoro-decoder-test.cpp)
    target_link_libraries(kokoro-decoder-test PRIVATE kokoro_lib)
endif()

# Server-mount handler: compiled into kokoro_lib only when the server target
# exists. The handler is guarded by `#ifdef LLAMA_BUILD_KOKORO` and pulls in
# the same `server-http.h` interface that the omnivoice handler uses, plus
# the vendored nlohmann/json.hpp from common's vendor directory.
if(LLAMA_BUILD_SERVER)
    target_sources(kokoro_lib PRIVATE src/kokoro-server-mount.cpp)
    target_compile_definitions(kokoro_lib PUBLIC LLAMA_BUILD_KOKORO)
    target_include_directories(kokoro_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../server
        ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor)
endif()

# Tests (built only when the parent build's tests are enabled).
if(LLAMA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
