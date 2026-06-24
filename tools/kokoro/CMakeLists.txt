# tools/kokoro/CMakeLists.txt
#
# Kokoro-82M (StyleTTS-2 + iSTFTNet) standalone TTS subtree, hosted inside
# the elizaOS llama.cpp fork. Built only when LLAMA_BUILD_KOKORO=ON
# (see the top-level CMakeLists.txt guard).
#
# Produces:
#   kokoro_lib            — static library: the vendored CrispStrobe/CrispASR
#                           ggml StyleTTS2 forward (kokoro-crispasr.cpp), the
#                           eliza_kokoro:: adapter facade (kokoro-adapter.cpp),
#                           the phoneme tokenizer, and the GGUF loader.
#   kokoro-tts            — standalone CLI harness (verification target).
#
# The forward (kokoro-crispasr.cpp) is the real iSTFTNet synthesis; the
# eliza_kokoro:: surface in kokoro.h is a thin adapter over it so the fused
# FFI (tools/omnivoice) and this CLI never change. The synthesis entry takes
# already-IPA phonemes, so the espeak / G2P front-end is compiled out — no
# CRISPASR_HAVE_ESPEAK_NG / CRISPASR_ESPEAK_DLOPEN define, no GPL link dep.
#
# The library transitively gets ggml + llama from the parent build's
# target set. Do not add_subdirectory(ggml) here.

set(KOKORO_PUBLIC_HEADERS
    include/kokoro.h
    include/kokoro-crispasr.h
    include/phonemizer.h
    include/kokoro-server-mount.h)

set(KOKORO_CORE_SOURCES
    src/kokoro-crispasr.cpp
    src/kokoro-adapter.cpp
    src/phonemizer.cpp
    src/core/gguf_loader.cpp
    src/core/audio_resample.cpp)

add_library(kokoro_lib STATIC
    ${KOKORO_CORE_SOURCES}
    ${KOKORO_PUBLIC_HEADERS})

# PIC so the static archive folds into the elizainference SHARED lib (same
# requirement as eliza_voice_classifiers / omnivoice_lib in tools/omnivoice).
set_target_properties(kokoro_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(kokoro_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# core/*.h (attention, conv, hifigan, gguf_loader, …) and the private ggml
# header the loader needs (ggml-backend-impl.h is not on the public include
# path — same access omnivoice/CMakeLists.txt takes via ${CMAKE_SOURCE_DIR}/ggml/src).
target_include_directories(kokoro_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_SOURCE_DIR}/ggml/src)

# ggml + llama are already configured by the parent build; pulling them via
# target_link_libraries gives us the include paths and the link line. CrispASR
# uses ggml_backend_init_best / ggml_backend_sched_* (in ggml-base) and
# ggml_col2im_1d (the fork-added op, CPU + CUDA + Vulkan).
target_link_libraries(kokoro_lib PUBLIC ggml)
if(TARGET ggml-base)
    target_link_libraries(kokoro_lib PUBLIC ggml-base)
endif()
if(TARGET ggml-cpu)
    target_link_libraries(kokoro_lib PUBLIC ggml-cpu)
endif()

target_compile_features(kokoro_lib PUBLIC cxx_std_17)

# Standalone CLI harness — synthesis verification target.
add_executable(kokoro-tts tools/kokoro-tts.cpp)
target_link_libraries(kokoro-tts PRIVATE kokoro_lib)
install(TARGETS kokoro-tts RUNTIME)

# Server-mount handler: compiled into kokoro_lib only when the server target
# exists. The handler drives the eliza_kokoro:: adapter surface (unchanged),
# and pulls in the same `server-http.h` interface that the omnivoice handler
# uses, plus the vendored nlohmann/json.hpp from common's vendor directory.
if(LLAMA_BUILD_SERVER)
    target_sources(kokoro_lib PRIVATE src/kokoro-server-mount.cpp)
    target_compile_definitions(kokoro_lib PUBLIC LLAMA_BUILD_KOKORO)
    target_include_directories(kokoro_lib PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../server
        ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor)
endif()
