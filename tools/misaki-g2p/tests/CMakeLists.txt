# misaki-g2p tests
#
#   misaki_g2p_unit       pure functions + C API on a small lexicon built from
#                         tests/data (no network, no eSpeak needed)
#   misaki_g2p_agreement  the agreement gate: native output against the pinned
#                         Python misaki 0.9.4 reference in fixtures/reference.txt
#                         (>= 99.5% of words and >= 98% of sentences identical);
#                         needs the full lexicon and libespeak-ng 1.52.0 data
#   misaki_g2p_agent      the same gate on assistant-style sentences
#                         (fixtures/agent_style.txt)
#
# The full lexicon comes from MISAKI_G2P_LEXICON, or is built by the
# misaki_g2p_lexicon fixture (convert_misaki_lexicon.py --fetch: Python 3 and
# network access to raw.githubusercontent.com, sha256-checked).

find_package(Python3 COMPONENTS Interpreter)

set(MISAKI_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(MISAKI_TEST_DATA ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_executable(test-misaki-g2p test_misaki_g2p.cpp)
target_include_directories(test-misaki-g2p PRIVATE ${MISAKI_ROOT}/src)
target_link_libraries(test-misaki-g2p PRIVATE misaki_g2p)

add_executable(test-misaki-g2p-agreement test_agreement.cpp)
target_include_directories(test-misaki-g2p-agreement PRIVATE ${MISAKI_ROOT}/src)
target_link_libraries(test-misaki-g2p-agreement PRIVATE misaki_g2p)

if (NOT Python3_Interpreter_FOUND)
    message(STATUS "misaki-g2p: Python 3 not found, lexicon-based tests are not registered")
    return()
endif()

set(MINI_LEX ${CMAKE_CURRENT_BINARY_DIR}/mini.lex)
add_test(NAME misaki_g2p_mini_lexicon
         COMMAND ${Python3_EXECUTABLE} ${MISAKI_ROOT}/convert_misaki_lexicon.py --allow-unpinned
                 --gold ${MISAKI_TEST_DATA}/mini_gold.json --silver ${MISAKI_TEST_DATA}/mini_silver.json -o ${MINI_LEX})
set_tests_properties(misaki_g2p_mini_lexicon PROPERTIES FIXTURES_SETUP misaki_mini_lex)

add_test(NAME misaki_g2p_unit COMMAND test-misaki-g2p ${MINI_LEX} ${MISAKI_TEST_DATA})
set_tests_properties(misaki_g2p_unit PROPERTIES FIXTURES_REQUIRED misaki_mini_lex)
# "main" is the label CI runs (ctest -L main): both are offline (committed
# mini lexicon data) and need neither eSpeak nor model weights.
set_tests_properties(misaki_g2p_mini_lexicon misaki_g2p_unit PROPERTIES LABELS main)

if (MISAKI_G2P_LEXICON)
    set(FULL_LEX ${MISAKI_G2P_LEXICON})
else()
    set(FULL_LEX ${CMAKE_CURRENT_BINARY_DIR}/misaki-en-us.lex)
    add_test(NAME misaki_g2p_lexicon
             COMMAND ${Python3_EXECUTABLE} ${MISAKI_ROOT}/convert_misaki_lexicon.py
                     --fetch ${CMAKE_CURRENT_BINARY_DIR}/misaki-data -o ${FULL_LEX})
    set_tests_properties(misaki_g2p_lexicon PROPERTIES FIXTURES_SETUP misaki_full_lex TIMEOUT 600)
endif()

# eSpeak NG data for the gate: explicit, else next to the linked library
set(GATE_ESPEAK_DATA "${MISAKI_G2P_ESPEAK_DATA}")
if (NOT GATE_ESPEAK_DATA AND MISAKI_ESPEAK_LIBRARY)
    get_filename_component(_espeak_libdir ${MISAKI_ESPEAK_LIBRARY} DIRECTORY)
    if (EXISTS ${_espeak_libdir}/../share/espeak-ng-data/phontab)
        get_filename_component(GATE_ESPEAK_DATA ${_espeak_libdir}/../share/espeak-ng-data ABSOLUTE)
    endif()
endif()

# misaki and Kokoro phonemizing concurrently through the shared libespeak-ng
# owner (tools/espeak-shared). Needs kokoro_lib, which is configured before
# this directory in the llama.cpp tree; skips (77) without libespeak-ng.
if (TARGET kokoro_lib)
    find_package(Threads REQUIRED)
    add_executable(test-espeak-shared test_espeak_shared.cpp)
    target_link_libraries(test-espeak-shared PRIVATE misaki_g2p kokoro_lib Threads::Threads)
    if (GATE_ESPEAK_DATA)
        set(_espeak_shared_args ${GATE_ESPEAK_DATA})
    else()
        set(_espeak_shared_args "")
    endif()
    add_test(NAME misaki_g2p_espeak_shared COMMAND test-espeak-shared ${MINI_LEX} ${_espeak_shared_args})
    set_tests_properties(misaki_g2p_espeak_shared PROPERTIES
        FIXTURES_REQUIRED misaki_mini_lex SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()

# eSpeak NG data-path handling: missing directory, paths longer than the
# linked libespeak-ng's N_PATH_HOME, recovery afterwards. Skips (77) without
# libespeak-ng or its data.
add_executable(test-espeak-data-path test_espeak_data_path.cpp)
target_link_libraries(test-espeak-data-path PRIVATE misaki_g2p)
if (GATE_ESPEAK_DATA)
    add_test(NAME misaki_g2p_espeak_data_path COMMAND test-espeak-data-path ${MINI_LEX} ${GATE_ESPEAK_DATA})
    set_tests_properties(misaki_g2p_espeak_data_path PROPERTIES
        FIXTURES_REQUIRED misaki_mini_lex SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()

foreach(gate agreement agent)
    if (gate STREQUAL "agreement")
        set(fixture ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/reference.txt)
    else()
        set(fixture ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/agent_style.txt)
    endif()
    add_test(NAME misaki_g2p_${gate}
             COMMAND test-misaki-g2p-agreement --lex ${FULL_LEX} --fixture ${fixture}
                     --espeak "${GATE_ESPEAK_DATA}" --report ${CMAKE_CURRENT_BINARY_DIR}/${gate}-report.txt)
    set_tests_properties(misaki_g2p_${gate} PROPERTIES SKIP_RETURN_CODE 77)
    if (NOT MISAKI_G2P_LEXICON)
        set_tests_properties(misaki_g2p_${gate} PROPERTIES FIXTURES_REQUIRED misaki_full_lex)
    endif()
endforeach()
