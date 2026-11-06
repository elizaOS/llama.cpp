# tools/parakeet/tests/CMakeLists.txt

add_executable(test-parakeet-kernels test_parakeet_kernels.cpp)
target_link_libraries(test-parakeet-kernels PRIVATE parakeet_lib)
target_include_directories(test-parakeet-kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME test-parakeet-kernels COMMAND test-parakeet-kernels)
# weight-free and fast: runs in CI (ctest -L main)
set_property(TEST test-parakeet-kernels PROPERTY LABELS main)

# Model parity gate against reference_transformers.py dumps
# (tools/dump_parakeet_reference.py). Needs the converted GGUF and the
# fixture directory, so it only runs when both are given:
#   -DPARAKEET_TEST_MODEL=/path/phonon-2-fv5.gguf -DPARAKEET_TEST_FIXTURES=/path/fixtures
add_executable(test-parakeet-parity test_parakeet_parity.cpp)
target_link_libraries(test-parakeet-parity PRIVATE parakeet_lib)
target_include_directories(test-parakeet-parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src ${CMAKE_CURRENT_SOURCE_DIR}/../../file-region)
if (PARAKEET_TEST_MODEL AND PARAKEET_TEST_FIXTURES)
    add_test(NAME test-parakeet-parity
             COMMAND test-parakeet-parity ${PARAKEET_TEST_MODEL} ${PARAKEET_TEST_FIXTURES})
endif()

# Region files (an APK asset): the GGUF inside a larger file loads mapped and
# read into memory and transcribes the same tokens. Needs the GGUF:
#   -DPARAKEET_TEST_MODEL=/path/phonon-2-fv5.gguf [-DPARAKEET_TEST_WAV=/path/16k.wav]
add_executable(test-parakeet-region test_parakeet_region.cpp)
target_link_libraries(test-parakeet-region PRIVATE parakeet_lib)
if (PARAKEET_TEST_MODEL)
    add_test(NAME test-parakeet-region COMMAND test-parakeet-region ${PARAKEET_TEST_MODEL} ${PARAKEET_TEST_WAV})
endif()
