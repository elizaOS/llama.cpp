# tools/parakeet/CMakeLists.txt
#
# Native Parakeet-TDT ASR (FastConformer encoder + TDT decoder) for
# FermionResearch/Phonon-2, the five-value QAT re-quantisation of
# nvidia/parakeet-tdt-0.6b-v3 (weights CC-BY-4.0). Built when
# LLAMA_BUILD_PARAKEET=ON (see the top-level CMakeLists.txt).
#
# Produces:
#   parakeet_lib     static library (GGUF loader, log-mel, encoder, TDT
#                    decoder, streaming). Uses ggml's gguf reader only; the
#                    compute is in pk-*.cpp with runtime-dispatched NEON
#                    i8mm / dotprod kernels, x86_64 AVX2 kernels and a
#                    scalar reference.
#   parakeet-cli     transcribe WAV files (host tools only).
#   parakeet-bench   RTFx / first-partial / RSS harness.

add_library(parakeet_lib STATIC
    include/parakeet.h
    src/parakeet.cpp
    src/pk-stream.cpp
    src/pk-ops.cpp
    src/pk-pool.cpp
    src/pk-kernels.cpp
    src/pk-kernels-dotprod.cpp
    src/pk-kernels-i8mm.cpp
    src/pk-kernels-avx2.cpp)
# hidden: folded PRIVATE into libelizainference, so pk_* must not be exported
set_target_properties(parakeet_lib PROPERTIES POSITION_INDEPENDENT_CODE ON
                                              CXX_VISIBILITY_PRESET hidden
                                              VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(parakeet_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(parakeet_lib PUBLIC ggml)
if (TARGET ggml-base)
    target_link_libraries(parakeet_lib PUBLIC ggml-base)
endif()
target_compile_features(parakeet_lib PUBLIC cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(parakeet_lib PUBLIC Threads::Threads)

# The dotprod and i8mm tiers are separate translation units compiled for
# their extension and only entered after a runtime CPU check (HWCAP /
# HWCAP2 on Linux/Android, sysctl on Apple), so the baseline stays armv8-a.
set(PARAKEET_ARM64 OFF)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$" OR CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(PARAKEET_ARM64 ON)
endif()
if (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")  # x86_64 build on an Apple silicon host
    set(PARAKEET_ARM64 OFF)
endif()
list(LENGTH CMAKE_OSX_ARCHITECTURES PARAKEET_N_OSX_ARCHS)
if (PARAKEET_ARM64 AND NOT MSVC AND PARAKEET_N_OSX_ARCHS LESS_EQUAL 1)
    set_source_files_properties(src/pk-kernels-dotprod.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
    set_source_files_properties(src/pk-kernels-i8mm.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod+i8mm")
elseif (PARAKEET_ARM64)
    message(WARNING "parakeet: dotprod/i8mm tiers need per-file -march; this configuration builds the scalar tier only")
    target_compile_definitions(parakeet_lib PRIVATE PK_NO_ARM_EXT=1)
endif()

# Likewise the AVX2 tier on x86_64 (Android x86_64 emulator images, desktop
# hosts): its own translation unit with -mavx2, entered after CPUID + XGETBV,
# so the baseline stays x86-64 (SSE2).
set(PARAKEET_X86_64 OFF)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" OR CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64" OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(PARAKEET_X86_64 ON)
endif()
if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(PARAKEET_X86_64 OFF)
endif()
if (PARAKEET_X86_64 AND MSVC)
    set_source_files_properties(src/pk-kernels-avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif (PARAKEET_X86_64 AND PARAKEET_N_OSX_ARCHS LESS_EQUAL 1)
    set_source_files_properties(src/pk-kernels-avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
else()
    if (PARAKEET_X86_64)
        message(WARNING "parakeet: the avx2 tier needs per-file -mavx2; this configuration builds the scalar tier only on x86_64")
    endif()
    target_compile_definitions(parakeet_lib PRIVATE PK_NO_X86_EXT=1)
endif()
if (NOT MSVC)
    target_compile_options(parakeet_lib PRIVATE -O3)
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    include(GNUInstallDirs)
    add_executable(parakeet-cli tools/parakeet-cli.cpp)
    target_link_libraries(parakeet-cli PRIVATE parakeet_lib)
    target_include_directories(parakeet-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    install(TARGETS parakeet-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    if (NOT WIN32)  # getrusage / getloadavg
        add_executable(parakeet-bench tools/parakeet-bench.cpp)
        target_link_libraries(parakeet-bench PRIVATE parakeet_lib)
        target_include_directories(parakeet-bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools)
    endif()
endif()

if (LLAMA_BUILD_TESTS OR PARAKEET_BUILD_TESTS)
    add_subdirectory(tests)
endif()
