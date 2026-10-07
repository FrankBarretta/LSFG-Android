# Embedded Wayland compositor: runs a libwayland-server compositor in-process so games launched
# through Wine's winewayland.drv present into it. See docs/WAYLAND-DISPLAY.md for the design and
# CREDITS.md for attribution. Protocol glue is pre-generated in generated/ so no host
# wayland-scanner is needed at build time.
add_library(wnwayland SHARED
        src/compositor.c
        src/vk_present.c
        src/effects_chain.c
        src/vk_loader.c
        src/sc_layer.c
        src/ahb_swapchain.c
        src/waylandcomp_jni.c
        src/desktop_ext.c
        src/wl_clipboard.c
        src/wl_text_input.c
        src/wl_toplevel_icon.c
        src/wl_color_mgmt.c
        src/hdr_compose.c
        src/blend_pass.c
        src/framegen_bridge.c
        src/framegen_jni.c
        generated/color-management-v1-protocol.c
        generated/xdg-shell-protocol.c
        generated/linux-dmabuf-v1-protocol.c
        generated/viewporter-protocol.c
        generated/desktop-v1-protocol.c
        generated/presentation-time-protocol.c
        generated/pointer-constraints-unstable-v1-protocol.c
        generated/relative-pointer-unstable-v1-protocol.c
        generated/wlr-data-control-unstable-v1-protocol.c
        generated/text-input-unstable-v3-protocol.c
        generated/xdg-toplevel-icon-v1-protocol.c
        generated/ahb-v1-protocol.c
)

target_include_directories(wnwayland PRIVATE
        generated
        prebuilt/include
        src
        # Committed SPIR-V of the screen-effect chain (src/effects_chain.c, src/hdr_compose.c).
        shaders
        ${CMAKE_CURRENT_SOURCE_DIR}/../adrenotools/include
)

# The X11 renderer's frame-generation engines run here too, on the compositor's own Turnip
# device. LSFG is compiled in and DIS comes from the wndis static library rather than being linked
# from libwinlator: both libraries keep a private VkDispatch, so the visibility below must stay for
# the two copies not to merge.
set(FRAMEGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../winlator/vk)
if (TARGET dxbc AND TARGET wndis)
    target_sources(wnwayland PRIVATE
            src/framegen_engine.c
            ${FRAMEGEN_DIR}/vk_dispatch.c
            ${FRAMEGEN_DIR}/lsfg/lsfg_dll.c
            ${FRAMEGEN_DIR}/lsfg/lsfg_dxbc.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_common.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_shaders.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_mipmaps.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_generate.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_alpha.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_beta.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_gamma.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_delta.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_chain.cpp
            ${FRAMEGEN_DIR}/lsfg/lsfg_pacer.cpp
            ${FRAMEGEN_DIR}/lsfg/vkr_lsfg.cpp
    )
    target_include_directories(wnwayland PRIVATE ${FRAMEGEN_DIR} ${WINLATOR_GENERATED_DIR})
    target_compile_definitions(wnwayland PRIVATE VK_USE_PLATFORM_ANDROID_KHR)
    target_compile_options(wnwayland PRIVATE -fvisibility=hidden)
    target_compile_features(wnwayland PRIVATE cxx_std_17)
    target_link_libraries(wnwayland dxbc wndis)
    add_dependencies(wnwayland winlator_shaders)
else()
    target_sources(wnwayland PRIVATE src/framegen_engine_stub.c)
endif()

add_library(wayland-server SHARED IMPORTED)
set_target_properties(wayland-server PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/prebuilt/lib/libwayland-server.so)

# Vulkan is loaded at runtime through adrenotools (Turnip), never linked: the compositor needs the
# driver's dma-buf import path, which the system Adreno driver lacks.
target_link_libraries(wnwayland
        wayland-server
        adrenotools
        dl
        log
        android
)
