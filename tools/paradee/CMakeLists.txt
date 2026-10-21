# tools/paradee/CMakeLists.txt
#
# Paradee-8M TTS (sahilmahendrakar/Paradee-8M-v1.0 and -v1.1, Apache-2.0): a
# width-scaled, single-voice distillation of Kokoro-82M. Built when
# LLAMA_BUILD_PARADEE=ON (see the top-level CMakeLists.txt).
#
# Produces:
#   paradee_lib   static library (loader, synthesis, phase-locking filter,
#                 v1.1 source-filter head).
#                 Reuses kokoro_lib's StyleTTS-2 / iSTFTNet layers.
#   paradee-tts   CLI: misaki phonemes -> 24 kHz WAV (host only).
#   paradee-stage-dump   writes every stage for the PyTorch parity gate.

if (NOT TARGET kokoro_lib)
    message(FATAL_ERROR "LLAMA_BUILD_PARADEE=ON needs kokoro_lib (LLAMA_BUILD_KOKORO=ON): "
                        "Paradee runs on the tools/kokoro layers.")
endif()

add_library(paradee_lib STATIC
    src/paradee.cpp
    src/paradee-phase-lock.cpp
    src/paradee-head.cpp
    src/paradee-head.h
    include/paradee.h)
set_target_properties(paradee_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(paradee_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(paradee_lib PUBLIC kokoro_lib ggml)
if (TARGET ggml-base)
    target_link_libraries(paradee_lib PUBLIC ggml-base)
endif()
target_compile_features(paradee_lib PUBLIC cxx_std_17)
# Same layer configuration as kokoro_lib, so the inline kokoro-layers.h
# definitions agree across the two archives.
target_compile_definitions(paradee_lib PRIVATE KOKORO_USE_GGML_BACKEND)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    include(GNUInstallDirs)
    add_executable(paradee-tts tools/paradee-tts.cpp)
    target_link_libraries(paradee-tts PRIVATE paradee_lib)
    install(TARGETS paradee-tts
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        BUNDLE DESTINATION ${CMAKE_INSTALL_BINDIR})

    add_executable(paradee-stage-dump tools/paradee-stage-dump.cpp)
    target_link_libraries(paradee-stage-dump PRIVATE paradee_lib)
endif()

if (LLAMA_BUILD_TESTS)
    add_subdirectory(tests)
endif()
