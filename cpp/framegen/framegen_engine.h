#ifndef FRAMEGEN_ENGINE_H
#define FRAMEGEN_ENGINE_H
/*
 * C face of the frame-generation engines for framegen_bridge.c.
 *
 * framegen_engine.c drives vkr_lsfg and vkr_dis - the same engines the X11 renderer runs -
 * on the compositor's own Turnip device. A build without the winlator tree compiles
 * framegen_engine_stub.c instead, which reports "engines not built in" and generates nothing.
 *
 * Every function runs on the compositor thread.
 */
#include <stdint.h>
#ifndef VK_USE_PLATFORM_ANDROID_KHR
#define VK_USE_PLATFORM_ANDROID_KHR
#endif
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Probe the physical device and build the VkDeviceCreateInfo pNext chain the LSFG chain needs
 * (vulkanMemoryModel + storage-image features; nothing for DIS). Returns the chain, or NULL
 * when the device fails a gate. `ring_fmt` is the format the engines generate into; it must be
 * storage-capable for either engine. */
const void *fge_probe(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd,
                      VkFormat ring_fmt);
/* The device exists; resolve device-level entry points. features_enabled = fge_probe's chain
 * was accepted by vkCreateDevice. */
void fge_device_ready(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd,
                      VkDevice dev, VkQueue queue, uint32_t qfam, int features_enabled);
/* 1 when engine `kind` can run on this device (its capability gates pass). */
int fge_caps_ok(int kind);
/* Why the LSFG gates fail ("supported" when they pass); static string. */
const char *fge_caps_reason(void);
/* 1 when the engines can generate in `fmt` on this device (storage + linear sampling), beyond the
 * ring format probed at fge_probe - the FP16 scene of HDR frames (vk_present.c). */
int fge_format_ok(VkFormat fmt);

/* Create engine `kind` (LSFG needs the cache path; DIS ignores it). 0 = ready, -1 = failed.
 * Only one engine holds GPU resources at a time; a different kind replaces it. */
int fge_start(int kind, const char *cache_path);
void fge_stop(void);
/* The engine came up but later found it cannot generate (chain build failed). */
int fge_unavailable(void);
void fge_configure(uint32_t multiplier, float flow_scale, float refresh_hz, int flow_min_side,
                   int target_fps);   /* flow_min_side is DIS-only; target_fps paces both */
/* Build/rebuild the chain for this extent; the guest extent for the flow pyramid is the same
 * (the Wayland scene is the game's own size in fullscreen). 1 = chain valid. */
int fge_prepare(uint32_t w, uint32_t h, VkFormat fmt);
/* How many frames to generate for this source frame (0 until the history is valid). */
uint32_t fge_plan(uint32_t capacity, uint64_t source_frames, float presented_rate);
/* Take the scene image (GENERAL) as frame N and run the shared part of the chain. */
void fge_process(VkCommandBuffer cmd, VkImage source, uint32_t w, uint32_t h, uint32_t gens);
/* Synthesise generated frame g of `count` into a storage image (GENERAL). */
void fge_generate(VkCommandBuffer cmd, uint32_t g, uint32_t count, VkImage img, VkImageView view,
                  uint32_t w, uint32_t h);
/* The generation targets were recreated: drop cached descriptors. */
void fge_forget_targets(void);
/* Engine-side telemetry: accepted generations, source rate (0 = engine has none), thermal. */
void fge_telemetry(float *accepted, float *source_rate, float *thermal);
const char *fge_engine_name(int kind);
/* "lsfg-native"/"dis-native" build provenance for the start log line. */
const char *fge_build_info(void);

#ifdef __cplusplus
}
#endif
#endif
