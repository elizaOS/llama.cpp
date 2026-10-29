find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK    Metal      REQUIRED)
find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)

message(STATUS "Metal framework found")

ggml_add_backend_library(ggml-metal
                         ggml-metal.cpp
                         ggml-metal-device.m
                         ggml-metal-device.cpp
                         ggml-metal-common.cpp
                         ggml-metal-context.m
                         ggml-metal-fusion.cpp
                         ggml-metal-ops.cpp
                         ggml-metal-tuning.cpp
                        )

target_link_libraries(ggml-metal PRIVATE
                      ${FOUNDATION_LIBRARY}
                      ${METAL_FRAMEWORK}
                      ${METALKIT_FRAMEWORK}
                      )

if (GGML_METAL_NDEBUG)
    add_compile_definitions(GGML_METAL_NDEBUG)
endif()

set(METALLIB_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-common.h")
set(METALLIB_KERNELS_COMMON     "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common.h")
set(METALLIB_KERNELS_DEQUANTIZE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/dequantize.h")
set(METALLIB_KERNELS_QUANTIZE   "${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantize.h")
set(METALLIB_KERNELS_FA_COMMON     "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fa_common.metal")
set(METALLIB_KERNELS_FA_VEC_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/kernels/fa_vec_common.metal")
set(METALLIB_KERNELS_FA_SHARED
    ${METALLIB_KERNELS_FA_COMMON}
    ${METALLIB_KERNELS_FA_VEC_COMMON}
)

set(METALLIB_KERNEL_SOURCES
    kernels/fa_aux.metal
    kernels/fa_f16.metal
    kernels/fa_f32.metal
    kernels/fa_q4_0.metal
    kernels/fa_q4_1.metal
    kernels/fa_q5_0.metal
    kernels/fa_q5_1.metal
    kernels/fa_q8_0.metal
    kernels/fa_tensor.metal
    kernels/fa_vec_f16.metal
    kernels/fa_vec_f32.metal
    kernels/fa_vec_q4_0.metal
    kernels/fa_vec_q4_1.metal
    kernels/fa_vec_q5_0.metal
    kernels/fa_vec_q5_1.metal
    kernels/fa_vec_q8_0.metal
    kernels/mul_mv.metal
    kernels/mul_mv_mma.metal
    kernels/mul_mm.metal
    kernels/quantize.metal
    kernels/softmax.metal
    kernels/norm.metal
    kernels/unary.metal
    kernels/binbcast.metal
    kernels/reduce.metal
    kernels/tri.metal
    kernels/ssm.metal
    kernels/wkv.metal
    kernels/gated_delta_net.metal
    kernels/solve_tri.metal
    kernels/rope.metal
    kernels/conv.metal
    kernels/upscale.metal
    kernels/argsort.metal
    kernels/pool.metal
    kernels/misc.metal
)

# Eliza standalone shaders. Each one is a self-contained TU (only
# #include <metal_stdlib>) that redeclares its own block structs and constants,
# so it is compiled to its own .air and never shares a TU with kernels/.
# Keep in sync with GGML_METAL_LIBS_ELIZA in ggml-metal-device.m.
set(METALLIB_ELIZA_SOURCES
    eliza-shipped/turbo3.metal
    eliza-shipped/turbo4.metal
    eliza-shipped/turbo3_tcq.metal
    eliza-shipped/qjl.metal
    eliza-shipped/qjl_set_rows.metal
    eliza-shipped/polar.metal
    eliza-shipped/polar_preht.metal
    eliza-shipped/fused_attn_qjl_tbq.metal
    eliza-shipped/fused_attn_qjl_polar.metal
    eliza-shipped/istft.metal
)

# CMAKE_OSX_SYSROOT is an SDK name or path - xcrun accepts both
set(METAL_SDK ${CMAKE_OSX_SYSROOT})
if (NOT METAL_SDK)
    set(METAL_SDK macosx)
endif()

if (CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
    set(METAL_TARGET_SIM "-simulator")
else()
    set(METAL_TARGET_SIM "")
endif()

if (GGML_METAL_SHADER_DEBUG)
    # note: disabling fast math is needed in order to pass tests/test-backend-ops
    # note: adding -fno-inline fixes the tests when using MTL_SHADER_VALIDATION=1
    # note: unfortunately, we have to call it default.metallib instead of ggml.metallib
    #       ref: https://github.com/ggml-org/whisper.cpp/issues/1720
    # note: adding -g causes segmentation fault during compile
    set(XC_FLAGS -fno-fast-math -fno-inline)
else()
    set(XC_FLAGS -O3)
endif()

execute_process(COMMAND xcrun -sdk ${METAL_SDK} --show-sdk-version OUTPUT_VARIABLE METAL_SDK_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
if (METAL_SDK_VERSION VERSION_GREATER_EQUAL 26.0)
    set(GGML_METAL_HAS_TENSOR_LIB ON)
else()
    message(STATUS "Metal SDK ${METAL_SDK_VERSION} does not support the tensor API, skipping ggml-tensor.metallib")
endif()

if (GGML_METAL_MACOSX_VERSION_MIN)
    message(STATUS "Adding  -mmacosx-version-min=${GGML_METAL_MACOSX_VERSION_MIN} flag to metal compilation")
    list   (APPEND XC_FLAGS -mmacosx-version-min=${GGML_METAL_MACOSX_VERSION_MIN})
elseif (NOT GGML_METAL_TARGET_OS STREQUAL "macos" AND CMAKE_OSX_DEPLOYMENT_TARGET)
    message(STATUS "Adding  -mtargetos=${GGML_METAL_TARGET_OS}${CMAKE_OSX_DEPLOYMENT_TARGET}${METAL_TARGET_SIM} flag to metal compilation")
    list   (APPEND XC_FLAGS -mtargetos=${GGML_METAL_TARGET_OS}${CMAKE_OSX_DEPLOYMENT_TARGET}${METAL_TARGET_SIM})
endif()

# the tensor API needs OS 26+ and the Metal 4 language that the OS 26 target
# selects by default, so GGML_METAL_STD (an older -std) must not apply to it
set(XC_FLAGS_TENSOR ${XC_FLAGS} -mtargetos=${GGML_METAL_TARGET_OS}26.0${METAL_TARGET_SIM})

if (GGML_METAL_STD)
    message(STATUS "Adding  -std=${GGML_METAL_STD} flag to metal compilation")
    list   (APPEND XC_FLAGS -std=${GGML_METAL_STD})
endif()

# # ELIZA-METAL-BF16-FIX
# Define GGML_METAL_HAS_BF16 when compiling the precompiled metallibs so that
# bfloat-typed kernels (e.g. kernel_get_rows_bf16, kernel_mul_mv_bf16_*,
# kernel_flash_attn_ext_bf16_*) are emitted. The runtime gates kernel lookup
# on the device's actual has_bfloat property via supports_op. CAUTION
# (#11612): kernels/common.h auto-undefs the macro when __METAL_VERSION__ <
# 310, so a GGML_METAL_STD below metal3.1 silently drops every bf16 kernel
# while bf16-capable GPUs still select them - pass -std=metal3.1 (or newer)
# for any device target that can report has_bfloat. The runtime additionally
# probes the loaded library for kernel_mul_mm_bf16_f32 (ggml-metal-device.m,
# ELIZA-METAL-BF16-LIBRARY-GATE) and downgrades has_bfloat if it is absent.
set(XC_DEFS_KERNELS -DGGML_METAL_HAS_BF16=1)

if (GGML_METAL_EMBED_LIBRARY)
    enable_language(ASM)

    add_compile_definitions(GGML_METAL_EMBED_LIBRARY)

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/autogenerated")

    # # ELIZA-KERNEL-EMBED-PATCH-V1
    # Embed a compiled default.metallib (not Metal source) for embedded-library
    # targets (iOS). Every kernels/<name>.metal source and every eliza-shipped
    # standalone shader is compiled to its own .air and linked into a single
    # metallib, so iOS and desktop load the same kernel set without compiling
    # Metal source at runtime. The eliza standalones redeclare block_* structs
    # and constants that already exist in ggml-common.h, so they cannot share a
    # source TU with kernels/. The kernels/ sources are compiled against the
    # source tree headers (no __embed_ggml-common.h__ inlining is needed).
    set(METALLIB_EMBED_DIR    "${CMAKE_CURRENT_BINARY_DIR}/autogenerated")
    set(METALLIB_EMBED_BINARY "${METALLIB_EMBED_DIR}/default.metallib")
    set(METALLIB_EMBED_ASM    "${METALLIB_EMBED_DIR}/ggml-metal-embed.s")
    set(METALLIB_EMBED_INC
        -I ${CMAKE_CURRENT_SOURCE_DIR}/kernels
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        -I ${CMAKE_CURRENT_SOURCE_DIR}/..)
    set(METALLIB_EMBED_HEADERS
        ${METALLIB_KERNELS_COMMON} ${METALLIB_KERNELS_DEQUANTIZE} ${METALLIB_KERNELS_QUANTIZE}
        ${METALLIB_KERNELS_FA_SHARED} ${METALLIB_COMMON} ggml-metal-impl.h)

    set(EMBED_AIR_FILES "")
    foreach(src ${METALLIB_KERNEL_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        set(AIR "${METALLIB_EMBED_DIR}/${name}.air")
        list(APPEND EMBED_AIR_FILES ${AIR})
        add_custom_command(
            OUTPUT ${AIR}
            COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS} ${XC_DEFS_KERNELS} ${METALLIB_EMBED_INC} -c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${AIR}
            DEPENDS ${src} ${METALLIB_EMBED_HEADERS}
            COMMENT "Compiling ${src} (embedded)"
            VERBATIM
        )
    endforeach()

    foreach(src ${METALLIB_ELIZA_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        set(AIR "${METALLIB_EMBED_DIR}/eliza_${name}.air")
        list(APPEND EMBED_AIR_FILES ${AIR})
        add_custom_command(
            OUTPUT ${AIR}
            COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS} -c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${AIR}
            DEPENDS ${src}
            COMMENT "Compiling ${src} (embedded)"
            VERBATIM
        )
    endforeach()

    add_custom_command(
        OUTPUT "${METALLIB_EMBED_ASM}"
        COMMAND echo "Embedding Metal library (compiled metallib + eliza-shipped kernels)"
        COMMAND xcrun -sdk ${METAL_SDK} metallib ${EMBED_AIR_FILES} -o "${METALLIB_EMBED_BINARY}"
        COMMAND echo ".section __DATA,__ggml_metallib"          >  "${METALLIB_EMBED_ASM}"
        COMMAND echo ".globl _ggml_metallib_start"              >> "${METALLIB_EMBED_ASM}"
        COMMAND echo "_ggml_metallib_start:"                    >> "${METALLIB_EMBED_ASM}"
        COMMAND echo .incbin "\"${METALLIB_EMBED_BINARY}\""    >> "${METALLIB_EMBED_ASM}"
        COMMAND echo ".globl _ggml_metallib_end"                >> "${METALLIB_EMBED_ASM}"
        COMMAND echo "_ggml_metallib_end:"                      >> "${METALLIB_EMBED_ASM}"
        DEPENDS ${EMBED_AIR_FILES}
        COMMENT "Generate assembly for embedded compiled Metal library"
        VERBATIM
    )

    set(METALLIB_EMBED_ASM_FILES "${METALLIB_EMBED_ASM}")

    # the tensor API kernels go in a separate embedded metallib, loaded only where supported
    if (GGML_METAL_HAS_TENSOR_LIB)
        set(METALLIB_EMBED_TENSOR_BINARY "${METALLIB_EMBED_DIR}/ggml-tensor.metallib")
        set(METALLIB_EMBED_TENSOR_ASM    "${METALLIB_EMBED_DIR}/ggml-metal-embed-tensor.s")
        set(EMBED_AIR_MM_TENSOR "${METALLIB_EMBED_DIR}/mul_mm_tensor.air")
        set(EMBED_AIR_FA_TENSOR "${METALLIB_EMBED_DIR}/fa_tensor_tensor.air")

        add_custom_command(
            OUTPUT ${EMBED_AIR_MM_TENSOR}
            COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS_TENSOR} ${XC_DEFS_KERNELS} -DGGML_METAL_HAS_TENSOR ${METALLIB_EMBED_INC} -c ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mm.metal -o ${EMBED_AIR_MM_TENSOR}
            DEPENDS kernels/mul_mm.metal ${METALLIB_EMBED_HEADERS}
            COMMENT "Compiling kernels/mul_mm.metal (tensor API, embedded)"
            VERBATIM
        )

        add_custom_command(
            OUTPUT ${EMBED_AIR_FA_TENSOR}
            COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS_TENSOR} ${XC_DEFS_KERNELS} -DGGML_METAL_HAS_TENSOR ${METALLIB_EMBED_INC} -c ${CMAKE_CURRENT_SOURCE_DIR}/kernels/fa_tensor.metal -o ${EMBED_AIR_FA_TENSOR}
            DEPENDS kernels/fa_tensor.metal ${METALLIB_EMBED_HEADERS}
            COMMENT "Compiling kernels/fa_tensor.metal (tensor API, embedded)"
            VERBATIM
        )

        add_custom_command(
            OUTPUT "${METALLIB_EMBED_TENSOR_ASM}"
            COMMAND xcrun -sdk ${METAL_SDK} metallib ${EMBED_AIR_MM_TENSOR} ${EMBED_AIR_FA_TENSOR} -o "${METALLIB_EMBED_TENSOR_BINARY}"
            COMMAND echo ".section __DATA,__ggml_metallib"          >  "${METALLIB_EMBED_TENSOR_ASM}"
            COMMAND echo ".globl _ggml_metallib_tensor_start"       >> "${METALLIB_EMBED_TENSOR_ASM}"
            COMMAND echo "_ggml_metallib_tensor_start:"             >> "${METALLIB_EMBED_TENSOR_ASM}"
            COMMAND echo .incbin "\"${METALLIB_EMBED_TENSOR_BINARY}\"" >> "${METALLIB_EMBED_TENSOR_ASM}"
            COMMAND echo ".globl _ggml_metallib_tensor_end"         >> "${METALLIB_EMBED_TENSOR_ASM}"
            COMMAND echo "_ggml_metallib_tensor_end:"               >> "${METALLIB_EMBED_TENSOR_ASM}"
            DEPENDS ${EMBED_AIR_MM_TENSOR} ${EMBED_AIR_FA_TENSOR}
            COMMENT "Generate assembly for embedded tensor API Metal library"
            VERBATIM
        )

        add_compile_definitions(GGML_METAL_EMBED_TENSOR_LIBRARY)
        list(APPEND METALLIB_EMBED_ASM_FILES "${METALLIB_EMBED_TENSOR_ASM}")
    endif()

    target_sources(ggml-metal PRIVATE ${METALLIB_EMBED_ASM_FILES})
else()
    # copy header files to bin directory
    configure_file(../ggml-common.h  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-common.h     COPYONLY)
    configure_file(ggml-metal-impl.h ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal-impl.h COPYONLY)

    file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels")
    configure_file(kernels/common.h     ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/common.h     COPYONLY)
    configure_file(kernels/dequantize.h ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/dequantize.h COPYONLY)
    configure_file(kernels/quantize.h   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/quantize.h   COPYONLY)
    foreach(hdr ${METALLIB_KERNELS_FA_SHARED})
        get_filename_component(hdr_name ${hdr} NAME)
        configure_file(${hdr} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/${hdr_name} COPYONLY)
    endforeach()

    foreach(src ${METALLIB_KERNEL_SOURCES})
        configure_file(${src} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${src} COPYONLY)
    endforeach()

    # the runtime source fallback loads eliza-shipped/<name>.metal next to kernels/
    file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/eliza-shipped")
    foreach(src ${METALLIB_ELIZA_SOURCES})
        configure_file(${src} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${src} COPYONLY)
    endforeach()

    # Compile each kernel source to .air, then link into default.metallib
    set(AIR_FILES "")
    foreach(src ${METALLIB_KERNEL_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        set(AIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}.air")
        list(APPEND AIR_FILES ${AIR})
        add_custom_command(
            OUTPUT ${AIR}
            COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS} ${XC_DEFS_KERNELS} -I ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${src} -o ${AIR}
            DEPENDS ${src} ${METALLIB_KERNELS_FA_SHARED} kernels/common.h kernels/dequantize.h kernels/quantize.h ${METALLIB_COMMON} ggml-metal-impl.h
            COMMENT "Compiling ${src}"
            VERBATIM
        )
    endforeach()

    # # ELIZA-KERNEL-PATCH-V1
    # Compile each eliza standalone shader into its own .air and link it into
    # default.metallib next to the kernels/ sources.
    foreach(src ${METALLIB_ELIZA_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        set(AIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/eliza_${name}.air")
        list(APPEND AIR_FILES ${AIR})
        add_custom_command(
            OUTPUT ${AIR}
            COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS} -c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${AIR}
            DEPENDS ${src}
            COMMENT "Compiling ${src}"
            VERBATIM
        )
    endforeach()

    set(METALLIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib)

    # the tensor API kernels go in a separate metallib, loaded only where supported
    if (GGML_METAL_HAS_TENSOR_LIB)
        set(AIR_MM_TENSOR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mul_mm_tensor.air")
        add_custom_command(
            OUTPUT ${AIR_MM_TENSOR}
            COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS_TENSOR} ${XC_DEFS_KERNELS} -DGGML_METAL_HAS_TENSOR -I ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/mul_mm.metal -o ${AIR_MM_TENSOR}
            DEPENDS kernels/mul_mm.metal kernels/common.h kernels/dequantize.h ${METALLIB_COMMON} ggml-metal-impl.h
            COMMENT "Compiling kernels/mul_mm.metal (tensor API)"
            VERBATIM
        )

        set(AIR_FA_TENSOR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fa_tensor_tensor.air")
        add_custom_command(
            OUTPUT ${AIR_FA_TENSOR}
            COMMAND xcrun -sdk ${METAL_SDK} metal ${XC_FLAGS_TENSOR} ${XC_DEFS_KERNELS} -DGGML_METAL_HAS_TENSOR -I ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels/fa_tensor.metal -o ${AIR_FA_TENSOR}
            DEPENDS kernels/fa_tensor.metal kernels/common.h ${METALLIB_COMMON} ggml-metal-impl.h
            COMMENT "Compiling kernels/fa_tensor.metal (tensor API)"
            VERBATIM
        )

        add_custom_command(
            OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-tensor.metallib
            COMMAND xcrun -sdk ${METAL_SDK} metallib ${AIR_MM_TENSOR} ${AIR_FA_TENSOR} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-tensor.metallib
            DEPENDS ${AIR_MM_TENSOR} ${AIR_FA_TENSOR}
            COMMENT "Linking tensor API Metal kernels into ggml-tensor.metallib"
        )

        list(APPEND METALLIB_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-tensor.metallib)
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
        COMMAND xcrun -sdk ${METAL_SDK} metallib ${AIR_FILES} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
        COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-common.h
        COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal-impl.h
        COMMAND rm -rf ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernels
        COMMAND rm -rf ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/eliza-shipped
        DEPENDS ${AIR_FILES} ${AIR_MM_TENSOR} ${AIR_FA_TENSOR}
        COMMENT "Linking Metal kernels (kernels/ + eliza-shipped/) into default.metallib"
    )

    add_custom_target(
        ggml-metal-lib ALL
        DEPENDS ${METALLIB_FILES}
    )
endif() # GGML_METAL_EMBED_LIBRARY

if (NOT GGML_METAL_EMBED_LIBRARY)
    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kernels/
        DESTINATION ${CMAKE_INSTALL_BINDIR}/kernels
        FILES_MATCHING PATTERN "*.metal" PATTERN "*.h"
    )

    install(
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/eliza-shipped/
        DESTINATION ${CMAKE_INSTALL_BINDIR}/eliza-shipped
        FILES_MATCHING PATTERN "*.metal"
    )

    install(
        FILES ${METALLIB_FILES}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
