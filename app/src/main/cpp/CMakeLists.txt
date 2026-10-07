cmake_minimum_required(VERSION 3.22)
project(framegen C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(GLSLC glslc REQUIRED HINTS
    "${CMAKE_ANDROID_NDK}/shader-tools/linux-x86_64"
    "${CMAKE_ANDROID_NDK}/shader-tools/darwin-x86_64"
    "${CMAKE_ANDROID_NDK}/shader-tools/windows-x86_64")
set(BIN2C_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/bin2c.cmake")

# ---- DIS (Dense Inverse Search) engine: GLSL -> SPIR-V headers -> static lib ----
set(DIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dis")
set(DIS_OUT "${CMAKE_CURRENT_BINARY_DIR}/dis_shaders")
file(MAKE_DIRECTORY "${DIS_OUT}/shaders")
set(DIS_SHADERS
    dis_luma_r16 dis_luma_r32 dis_gradient dis_inverse_search dis_propagate dis_densify
    dis_interpolate dis_hist dis_side dis_flow_pack dis_vr_prep dis_vr_d1 dis_vr_d2 dis_vr_w
    dis_vr_coef dis_vr_sor dis_vr_add dis_me_luma)
set(DIS_HEADERS "")
foreach(base ${DIS_SHADERS})
    set(var "${base}_comp")
    set(spv "${DIS_OUT}/shaders/${var}.spv")
    set(hdr "${DIS_OUT}/shaders/${var}.spv.h")
    add_custom_command(
        OUTPUT "${hdr}"
        COMMAND "${GLSLC}" --target-env=vulkan1.1 -O "${DIS_DIR}/include/shaders/${base}.comp" -o "${spv}"
        COMMAND "${CMAKE_COMMAND}" -DINPUT_FILE=${spv} -DOUTPUT_FILE=${hdr} -DVAR_NAME=${var} -P "${BIN2C_SCRIPT}"
        DEPENDS "${DIS_DIR}/include/shaders/${base}.comp" "${BIN2C_SCRIPT}"
        VERBATIM)
    list(APPEND DIS_HEADERS "${hdr}")
endforeach()
add_custom_target(dis_shaders DEPENDS ${DIS_HEADERS})

add_library(wndis STATIC ${DIS_DIR}/src/vkr_dis.c ${DIS_DIR}/src/dis_qcom_me.c)
add_dependencies(wndis dis_shaders)
set_target_properties(wndis PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(wndis PUBLIC VK_USE_PLATFORM_ANDROID_KHR)
target_include_directories(wndis
    PUBLIC  ${DIS_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/shared
    PRIVATE ${DIS_OUT})
target_link_libraries(wndis PRIVATE log dl)

# ---- JNI library (name kept: FrameGenNative loads "winlator") ----
add_library(winlator SHARED
    shared/vk_dispatch.c
    shared/presenter/fg_present.c
    shared/presenter/fg_jni.c
    shared/lsfg_cache_jni.c
    shared/lsfg/lsfg_dll.c
    shared/lsfg/lsfg_dxbc_stub.c
    shared/lsfg/lsfg_alpha.cpp
    shared/lsfg/lsfg_beta.cpp
    shared/lsfg/lsfg_chain.cpp
    shared/lsfg/lsfg_common.cpp
    shared/lsfg/lsfg_delta.cpp
    shared/lsfg/lsfg_gamma.cpp
    shared/lsfg/lsfg_generate.cpp
    shared/lsfg/lsfg_mipmaps.cpp
    shared/lsfg/lsfg_pacer.cpp
    shared/lsfg/lsfg_shaders.cpp
    shared/lsfg/vkr_lsfg.cpp)
target_compile_definitions(winlator PRIVATE VK_USE_PLATFORM_ANDROID_KHR)
target_include_directories(winlator PRIVATE shared)
target_link_libraries(winlator PRIVATE wndis android log mediandk nativewindow dl)
