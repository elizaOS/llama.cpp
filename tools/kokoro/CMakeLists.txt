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
    include/kokoro-layers.h
    include/kokoro-predictor.h)

set(KOKORO_CORE_SOURCES
    src/kokoro.cpp
    src/kokoro-istft.cpp
    src/kokoro-phonemes.cpp
    src/kokoro-predictor.cpp)

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

# Standalone CLI harness — required by J2 verification (tools/voice-kokoro/).
add_executable(kokoro-tts tools/kokoro-tts.cpp)
target_link_libraries(kokoro-tts PRIVATE kokoro_lib)
install(TARGETS kokoro-tts RUNTIME)

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
