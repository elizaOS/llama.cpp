# tools/kokoro/tests/CMakeLists.txt
#
# Kokoro inference unit tests. Each test is a small standalone executable
# linked against kokoro_lib + (where needed) llama / ggml.

add_executable(test-kokoro-phonemes test_kokoro_phonemes.cpp)
target_link_libraries(test-kokoro-phonemes PRIVATE kokoro_lib)
add_test(NAME test-kokoro-phonemes COMMAND test-kokoro-phonemes)

add_executable(test-kokoro-istft test_kokoro_istft.cpp)
target_link_libraries(test-kokoro-istft PRIVATE kokoro_lib)
add_test(NAME test-kokoro-istft COMMAND test-kokoro-istft)
