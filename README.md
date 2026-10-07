# WinNative 0.7.1-beta — FrameGen extracted system

Extracted Frame Generation implementation.

Included: DIS Native (Dense Inverse Search + Vulkan compute interpolation), LSFG Native, Android/Kotlin API, JNI, generation ring/pacing bridge, and Vulkan presenter integration references.

Pipeline: FrameGen Kotlin -> JNI -> framegen_bridge -> framegen_engine -> DIS/LSFG -> Vulkan compute -> generation ring -> present.

DIS-only core: cpp/framegen/* + cpp/dis/* + cpp/shared/vk_loader.h + cpp/shared/vk_dispatch.* plus host compositor/present integration.

This is source extraction, not a standalone app. It depends on the host Vulkan device/queue, command-buffer lifecycle, swapchain/present loop, JNI registration, logging, and build system. DIS does not require the LSFG cache path.
