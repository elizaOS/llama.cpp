# misaki-g2p: native misaki 0.9.4 American English G2P for Paradee-8M.
#
# Pure C++17, no ggml dependency: usable from the llama.cpp tree
# (LLAMA_BUILD_MISAKI_G2P) or standalone (cmake -S tools/misaki-g2p).
#
# Options
#   MISAKI_G2P_ESPEAK        link libespeak-ng for misaki's eSpeak fallback
#                            (GPL-3.0-or-later; flagged, not required).
#                            Uses an existing `espeak-ng` CMake target when the
#                            parent project defines one, else find_library with
#                            MISAKI_G2P_ESPEAK_ROOT as a hint.
#   MISAKI_G2P_BUILD_TESTS   unit tests and the misaki agreement gate.
#   MISAKI_G2P_LEXICON       prebuilt misaki-en-us.lex for the tests; when
#                            empty the tests build one with
#                            convert_misaki_lexicon.py --fetch (Python 3 and
#                            network access to raw.githubusercontent.com).
#   MISAKI_G2P_ESPEAK_DATA   espeak-ng-data directory for the gate test.
cmake_minimum_required(VERSION 3.14)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(misaki_g2p CXX)
    set(MISAKI_G2P_STANDALONE ON)
    include(CTest)
else()
    set(MISAKI_G2P_STANDALONE OFF)
endif()

option(MISAKI_G2P_ESPEAK "Link libespeak-ng for the misaki OOV fallback (GPL-3.0-or-later)" ON)
set(MISAKI_G2P_ESPEAK_ROOT "" CACHE PATH "Prefix to search for libespeak-ng")
if (MISAKI_G2P_STANDALONE)
    option(MISAKI_G2P_BUILD_TESTS "Build misaki-g2p tests" ON)
else()
    option(MISAKI_G2P_BUILD_TESTS "Build misaki-g2p tests" ${LLAMA_BUILD_TESTS})
endif()
set(MISAKI_G2P_LEXICON "" CACHE FILEPATH "Prebuilt misaki-en-us.lex for the tests")
set(MISAKI_G2P_ESPEAK_DATA "" CACHE PATH "espeak-ng-data directory for the agreement gate")

add_library(misaki_g2p STATIC
    src/espeak.cpp
    src/espeak-g2p.cpp
    src/g2p.cpp
    src/lexicon.cpp
    src/misaki-g2p.cpp
    src/numbers.cpp
    src/tagger.cpp
    src/tokenizer.cpp
    src/unicode.cpp
)
target_include_directories(misaki_g2p PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(misaki_g2p PUBLIC cxx_std_17)
set_target_properties(misaki_g2p PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
target_compile_definitions(misaki_g2p PRIVATE MISAKI_G2P_BUILD)

set(MISAKI_G2P_HAS_ESPEAK OFF)
if (MISAKI_G2P_ESPEAK)
    if (TARGET espeak-ng)
        target_link_libraries(misaki_g2p PRIVATE espeak-ng)
        set(MISAKI_G2P_HAS_ESPEAK ON)
    else()
        find_path(MISAKI_ESPEAK_INCLUDE_DIR espeak-ng/speak_lib.h
            HINTS ${MISAKI_G2P_ESPEAK_ROOT} /opt/homebrew /usr/local /usr PATH_SUFFIXES include)
        find_library(MISAKI_ESPEAK_LIBRARY NAMES espeak-ng
            HINTS ${MISAKI_G2P_ESPEAK_ROOT} /opt/homebrew /usr/local /usr PATH_SUFFIXES lib lib64)
        if (MISAKI_ESPEAK_INCLUDE_DIR AND MISAKI_ESPEAK_LIBRARY)
            target_include_directories(misaki_g2p PRIVATE ${MISAKI_ESPEAK_INCLUDE_DIR})
            target_link_libraries(misaki_g2p PRIVATE ${MISAKI_ESPEAK_LIBRARY})
            set(MISAKI_G2P_HAS_ESPEAK ON)
        endif()
    endif()
    if (MISAKI_G2P_HAS_ESPEAK)
        target_compile_definitions(misaki_g2p PRIVATE MISAKI_G2P_USE_ESPEAK)
        # One lock + init owner for the process-global libespeak-ng, shared
        # with tools/kokoro (header-only; see tools/espeak-shared).
        target_include_directories(misaki_g2p PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../espeak-shared)
        message(STATUS "misaki-g2p: libespeak-ng linked (GPL-3.0-or-later): eSpeak + E2M fallback, as in Paradee training")
    else()
        message(STATUS "misaki-g2p: libespeak-ng not found: out-of-lexicon words are spelled out (fallback kind 0)")
    endif()
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
    add_executable(misaki-g2p-cli tools/misaki-g2p-cli.cpp)
    target_include_directories(misaki-g2p-cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(misaki-g2p-cli PRIVATE misaki_g2p)
endif()

if (MISAKI_G2P_BUILD_TESTS AND NOT CMAKE_CROSSCOMPILING)
    enable_testing()
    add_subdirectory(tests)
endif()
