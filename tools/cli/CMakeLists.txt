set(TARGET llama-cli)
add_executable(${TARGET} cli.cpp)
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

include_directories(../server)

# llama-cli rejects `--no-conversation` and prints
# "please use llama-completion instead" (see cli.cpp). Build the redirect
# target alongside llama-cli so the documented escape hatch is always
# present whenever someone explicitly asks for --target llama-cli.
if(TARGET llama-completion)
    add_dependencies(${TARGET} llama-completion)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
