# tools/paradee/tests/CMakeLists.txt
#
# test-paradee-phase-lock   self-contained (committed fixture), always runs.
# test-paradee-api          needs the Paradee GGUF.
# test-paradee-parity       needs the GGUF and a reference dump from
#                           tools/dump_paradee_reference.py.
#
# The model-backed tests read PARADEE_GGUF / PARADEE_REF_DIR from the
# environment, or from the cache variables of the same names, and report
# SKIP (exit 77) when they are not set. See tools/paradee/README.md.

set(PARADEE_GGUF "" CACHE FILEPATH "Paradee GGUF for test-paradee-api / test-paradee-parity")
set(PARADEE_REF_DIR "" CACHE PATH "dump_paradee_reference.py output for test-paradee-parity")

set(PARADEE_FIXTURE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

add_executable(test-paradee-phase-lock test_paradee_phase_lock.cpp)
target_link_libraries(test-paradee-phase-lock PRIVATE paradee_lib)
target_compile_definitions(test-paradee-phase-lock PRIVATE PARADEE_FIXTURE_DIR="${PARADEE_FIXTURE_DIR}")
add_test(NAME test-paradee-phase-lock COMMAND test-paradee-phase-lock)

add_executable(test-paradee-api test_paradee_api.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test-paradee-api PRIVATE paradee_lib Threads::Threads)
add_test(NAME test-paradee-api COMMAND test-paradee-api)

add_executable(test-paradee-parity test_paradee_parity.cpp)
target_link_libraries(test-paradee-parity PRIVATE paradee_lib)
target_include_directories(test-paradee-parity PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
target_compile_definitions(test-paradee-parity PRIVATE PARADEE_FIXTURE_DIR="${PARADEE_FIXTURE_DIR}")
add_test(NAME test-paradee-parity COMMAND test-paradee-parity)

set(_paradee_env "")
if (PARADEE_GGUF)
    list(APPEND _paradee_env "PARADEE_GGUF=${PARADEE_GGUF}")
endif()
if (PARADEE_REF_DIR)
    list(APPEND _paradee_env "PARADEE_REF_DIR=${PARADEE_REF_DIR}")
endif()
foreach(t test-paradee-api test-paradee-parity)
    set_tests_properties(${t} PROPERTIES SKIP_RETURN_CODE 77)
    if (_paradee_env)
        set_tests_properties(${t} PROPERTIES ENVIRONMENT "${_paradee_env}")
    endif()
endforeach()
set_tests_properties(test-paradee-parity PROPERTIES TIMEOUT 900)
