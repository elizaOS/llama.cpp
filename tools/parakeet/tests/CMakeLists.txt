# tools/parakeet/tests/CMakeLists.txt

add_executable(test-parakeet-kernels test_parakeet_kernels.cpp)
target_link_libraries(test-parakeet-kernels PRIVATE parakeet_lib)
target_include_directories(test-parakeet-kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME test-parakeet-kernels COMMAND test-parakeet-kernels)

# Model parity gate against reference_transformers.py dumps
# (tools/dump_parakeet_reference.py). Needs the converted GGUF and the
# fixture directory, so it only runs when both are given:
#   -DPARAKEET_TEST_MODEL=/path/phonon-2-fv5.gguf -DPARAKEET_TEST_FIXTURES=/path/fixtures
add_executable(test-parakeet-parity test_parakeet_parity.cpp)
target_link_libraries(test-parakeet-parity PRIVATE parakeet_lib)
target_include_directories(test-parakeet-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
if (PARAKEET_TEST_MODEL AND PARAKEET_TEST_FIXTURES)
    add_test(NAME test-parakeet-parity
             COMMAND test-parakeet-parity ${PARAKEET_TEST_MODEL} ${PARAKEET_TEST_FIXTURES})
endif()
