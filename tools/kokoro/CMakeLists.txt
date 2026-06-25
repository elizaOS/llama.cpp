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

# kokoro_lib is folded into the fused libelizainference SHARED library
# (tools/omnivoice/CMakeLists.txt links it PRIVATE). When the parent build is
# configured with BUILD_SHARED_LIBS=ON (e.g. the OpenVINO Linux CI job), the
# static archive's objects must be position-independent or ld refuses to fold
# them into a -shared object ("relocation R_X86_64_PC32 ... can not be used
# when making a shared object; recompile with -fPIC"). PIC is not transitive
# from the SHARED consumer, so set it on the static target itself — mirroring
# eliza_voice_classifiers / omnivoice_lib in the sibling omnivoice subtree.
set_target_properties(kokoro_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

# Real G2P via libespeak-ng. When present, kokoro_phonemize() drives
# espeak_TextToPhonemes() (en-us IPA) and maps codepoints to Kokoro vocab ids,
# reproducing the reference token sequence. When absent, the build falls back
# to the degraded ASCII grapheme mapping and the TS layer must supply IPA.
# Override the search with -DKOKORO_ESPEAK_ROOT=<prefix> (e.g. Homebrew).
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
        message(STATUS "Kokoro G2P: libespeak-ng not found — falling back to ASCII grapheme mapping (TS layer must supply IPA)")
    endif()
endif()

# Standalone CLI harnesses (required by J2 verification + Kokoro decoder dev).
# Force MACOSX_BUNDLE OFF: CMake defaults Apple executables to bundles, and
# install(TARGETS ... RUNTIME) on a bundle target fails configure with
# "no BUNDLE DESTINATION" on every ios/tvos/visionos/macos build.
add_executable(kokoro-tts tools/kokoro-tts.cpp)
target_link_libraries(kokoro-tts PRIVATE kokoro_lib)
set_target_properties(kokoro-tts PROPERTIES MACOSX_BUNDLE OFF)
install(TARGETS kokoro-tts RUNTIME)

add_executable(kokoro-stage-dump tools/kokoro-stage-dump.cpp)
target_link_libraries(kokoro-stage-dump PRIVATE kokoro_lib)
set_target_properties(kokoro-stage-dump PROPERTIES MACOSX_BUNDLE OFF)

add_executable(kokoro-decoder-test tools/kokoro-decoder-test.cpp)
target_link_libraries(kokoro-decoder-test PRIVATE kokoro_lib)
set_target_properties(kokoro-decoder-test PROPERTIES MACOSX_BUNDLE OFF)

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
