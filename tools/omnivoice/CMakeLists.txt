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
# Mirrors the original omnivoice.cpp CMakeLists.txt. Attached PRIVATE on
# every omnivoice target rather than redefined globally — keeps the
# LLAMA_BUILD_OMNIVOICE wedge self-contained and additive.
set(OMNIVOICE_COMPILE_DEFINITIONS GGML_MAX_NAME=128)

# Sources shared by both the static core and the shared ABI-export library.
set(OMNIVOICE_CORE_SOURCES
    src/omnivoice.cpp
    src/pipeline-codec.cpp
    src/pipeline-tts.cpp
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
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR})
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
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(omnivoice PUBLIC llama)
    set_target_properties(omnivoice PROPERTIES
        C_VISIBILITY_PRESET       hidden
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)
    add_dependencies(omnivoice omnivoice-version)
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
