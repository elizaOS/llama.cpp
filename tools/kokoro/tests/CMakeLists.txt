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

add_executable(test-kokoro-tensor-names test_kokoro_tensor_names.cpp)
target_link_libraries(test-kokoro-tensor-names PRIVATE kokoro_lib)
add_test(NAME test-kokoro-tensor-names COMMAND test-kokoro-tensor-names)

# G2P: text → espeak IPA → Kokoro vocab ids (reproduces the reference token
# sequence). Self-checking; no-ops gracefully when espeak is not linked.
add_executable(test-kokoro-g2p-espeak test_kokoro_g2p_espeak.cpp)
target_link_libraries(test-kokoro-g2p-espeak PRIVATE kokoro_lib)
add_test(NAME test-kokoro-g2p-espeak COMMAND test-kokoro-g2p-espeak)

# Portable-fast (non-Apple) layer path: parity vs the scalar reference on the
# four hot primitives + a microbench on the real generator shapes. Header-only
# — deliberately NOT linked against kokoro_lib (it instantiates the header
# twice under renamed namespaces; see the *_impl.cpp files).
add_executable(test-kokoro-layers-portable
    test_kokoro_layers_portable.cpp
    test_kokoro_layers_ref_impl.cpp
    test_kokoro_layers_fast_impl.cpp)
target_include_directories(test-kokoro-layers-portable PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
find_package(Threads REQUIRED)
target_link_libraries(test-kokoro-layers-portable PRIVATE Threads::Threads)
add_test(NAME test-kokoro-layers-portable COMMAND test-kokoro-layers-portable)
