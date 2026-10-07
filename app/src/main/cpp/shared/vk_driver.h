#pragma once
/* Standalone replacement for WinNative's vk_driver.h: use the device's own libvulkan.so
 * (no adrenotools / Turnip). fg_present.c only needs the dlopen handle. */
#include <dlfcn.h>
#include <jni.h>

static inline void* winlator_open_vulkan(JNIEnv* env, jobject context, const char* driver_name) {
    (void)env; (void)context; (void)driver_name;
    return dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
}
