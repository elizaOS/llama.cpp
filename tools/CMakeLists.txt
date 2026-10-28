# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (LLAMA_BUILD_KOKORO)
        # Kokoro-82M (StyleTTS-2 + iSTFTNet) standalone TTS — J2 fork-side
        # replacement for the ONNX runtime path. Produces kokoro_lib +
        # the kokoro-tts CLI; when LLAMA_BUILD_SERVER is also on, the
        # `/v1/audio/speech` handler is mounted via server.cpp.
        #
        # MUST precede omnivoice below: the fused `elizainference` target folds
        # kokoro in via `if(TARGET kokoro_lib)` (ABI v10), so kokoro_lib has to
        # exist by the time omnivoice configures. The root CMakeLists mirrors
        # this ordering for the embed-as-library (LLAMA_BUILD_TOOLS=OFF) case.
        add_subdirectory(kokoro)
    endif()
    if (LLAMA_BUILD_MISAKI_G2P)
        # Native misaki G2P (the phoneme spelling Paradee and Kokoro were
        # trained on). Added when the subtree is present.
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/misaki-g2p/CMakeLists.txt")
            add_subdirectory(misaki-g2p)
        else()
            message(STATUS "LLAMA_BUILD_MISAKI_G2P: tools/misaki-g2p/ is not in this checkout; skipped")
        endif()
    endif()
    if (LLAMA_BUILD_PARADEE)
        # Paradee-8M (width-scaled Kokoro student, single voice). Needs
        # kokoro_lib; like kokoro it precedes omnivoice so the fused
        # `elizainference` target can fold paradee_lib in.
        # A build with Kokoro off skips Paradee rather than failing.
        if (LLAMA_BUILD_KOKORO)
            add_subdirectory(paradee)
        else()
            message(STATUS "LLAMA_BUILD_PARADEE: skipped because LLAMA_BUILD_KOKORO=OFF")
        endif()
    endif()
    if (LLAMA_BUILD_PARAKEET)
        # Parakeet-TDT ASR (Phonon-2 five-value weights). Like kokoro it
        # precedes omnivoice so the fused `elizainference` target can fold
        # parakeet_lib in.
        add_subdirectory(parakeet)
    endif()
    if (LLAMA_BUILD_OMNIVOICE)
        # The merged omnivoice subtree lives under tools/omnivoice/. The
        # top-level CMakeLists.txt has a parallel add_subdirectory hook for
        # the embed-as-library case (LLAMA_BUILD_TOOLS=OFF) ; this gate is
        # the one that fires for the normal standalone build.
        add_subdirectory(omnivoice)
    endif()
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (LLAMA_CONNECT)
        add_subdirectory(connect)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    if (GGML_METAL)
        add_subdirectory(tuning)
    endif()
    add_subdirectory(results)
endif()
