#ifndef FRAMEGEN_BRIDGE_H
#define FRAMEGEN_BRIDGE_H
/*
 * Frame generation on the Wayland backend.
 *
 * The two engines the X11 renderer runs - LSFG (the user's Lossless.dll chain, vk/lsfg) and
 * DIS (the dense inverse search optical-flow chain, vk/dis) - are hosted here on the Wayland
 * compositor's Turnip device. Both consume consecutive presented frames and emit interpolated
 * frames between them; the guest never sees any of it.
 *
 * Present order per scene frame (vk_present.c):
 *     scene  ->  effects  ->  FRAME GENERATION  ->  mapping/blit  ->  swapchain
 * The scene is composed 1:1 into an off-screen image, the effects chain runs on it, then
 * vkp_framegen_run() (the one hook) feeds that image to the engine and hands back up to three
 * generated images. vk_present.c blits the generated frames to the screen FIRST and the real
 * frame LAST (interpolation produces frames that belong between N-1 and N), one FIFO present
 * each, so the panel shows them on consecutive vblanks - the same pacing the X11 native path
 * is device-proven with. No sleeps, no host-side pacer.
 *
 * Threads: the set_* controls are called from the app (JNI, any thread) and only store
 * values; everything that touches Vulkan runs on the compositor thread inside vkp_render().
 */
#include <stdint.h>
#ifndef VK_USE_PLATFORM_ANDROID_KHR
#define VK_USE_PLATFORM_ANDROID_KHR
#endif
#include <vulkan/vulkan.h>

enum { VKP_FG_ENGINE_LSFG = 0, VKP_FG_ENGINE_DIS = 1 };
/* Hard ceiling on generated frames per source frame, as on X11: DIS goes to 8x, LSFG to 4x. */
#define VKP_FG_MAX_GENERATIONS 7
#define VKP_FG_LSFG_MAX_GENERATIONS 3

/* ---- app controls (any thread) ------------------------------------------------------- */
void vkp_framegen_set_engine(int kind);
/* multiplier 2..4; armed = 0 stops generating (the ring and the engine stay for a re-arm). */
void vkp_framegen_set_armed(int armed, int multiplier);
/* LSFG Native: the SPIR-V cache built from the user's Lossless.dll (LsfgNative.ensureCache). */
void vkp_framegen_set_lsfg_cache_path(const char *path);
/* Flow scale 0.25..1.0 and the panel's real refresh rate (the pacer never generates above it). */
void vkp_framegen_set_tuning(float flow_scale, float refresh_hz);
/* Flow resolution in pixels of the frame's short side (DIS only) and the target output rate
 * both engines pace to (0 = follow the panel). */
void vkp_framegen_set_engine_tuning(int flow_min_side, int target_fps);
/* Same codes as VulkanRenderer.getFrameGenProblem(): -1 not known yet (device not up), 0 fine,
 * 1 the driver lacks what the selected engine needs (vkp_framegen_caps_reason says what),
 * 2 the engine failed to start / build its chain. */
int vkp_framegen_problem(void);
const char *vkp_framegen_caps_reason(void);
/* Same shape as VulkanRenderer.getFrameGenStats(): {generations trusted, generations planned,
 * real (source) fps, presented fps incl. generated frames, thermal (-1 = none), ms per
 * generated frame (-1 = unknown)}. */
void vkp_framegen_stats(float out[6]);
/* Generated frames presented since the last call (compositor.c, the 10 s stats line). */
unsigned vkp_framegen_stats_take(void);
/* Cumulative counters for the HUD's output-frame source; never reset within a process. */
uint64_t vkp_framegen_total_presented(void);
uint64_t vkp_framegen_total_generated(void);

/* ---- vk_present.c: device setup (compositor thread) ----------------------------------- */
/* Before vkCreateDevice: probe what the engines need on this physical device and return the
 * pNext chain to hang on VkDeviceCreateInfo (NULL when the device cannot run the LSFG chain -
 * DIS needs no feature). Owned by the bridge, valid until the next call. */
const void *vkp_framegen_device_features(VkInstance inst, PFN_vkGetInstanceProcAddr gipa,
                                         VkPhysicalDevice pd);
/* After the device exists (features_enabled = the chain above was accepted). */
void vkp_framegen_device_ready(VkDevice dev, VkQueue queue, uint32_t qfam, int features_enabled);
/* VK_ERROR_DEVICE_LOST: never touch the device again. */
void vkp_framegen_device_lost(void);

/* ---- vk_present.c: present path (compositor thread) ------------------------------------ */
/* 1 when this frame must go through the compositor pass (armed and the engine can run). Safe
 * before the device is up (0). */
int vkp_framegen_active(void);
/* Extra swapchain images wanted so every present of one scene frame is queued without waiting
 * for a vblank: multiplier - 1 while armed, 0 otherwise. A change means "rebuild the swapchain". */
int vkp_framegen_extra_images(void);
/* THE hook. Called once per composed scene frame, after the effects hook, with the scene image
 * (GENERAL, w x h, fmt, TRANSFER_SRC|SAMPLED). Records the engine's chain into `cmd` and
 * returns how many generated frames (0..VKP_FG_MAX_GENERATIONS) it produced, their images in
 * gens[] (GENERAL, w x h, already made visible to TRANSFER reads). 0 = present the scene alone. */
int vkp_framegen_run(VkCommandBuffer cmd, VkImage scene, VkImageView scene_view, int w, int h,
                     VkFormat fmt, VkImage gens[VKP_FG_MAX_GENERATIONS]);
/* After the presents of one scene frame: how many generated frames actually reached the
 * swapchain (rate tracking + the stats counters). */
void vkp_framegen_presented(int generated);
/* 1 when vkp_framegen_run can be fed `fmt` (the 8-bit scene always; FP16 for HDR frames where the
 * device can generate in it and the engine has not refused it this session). A chain that fails to
 * build in a deeper format is not an engine failure: the engine restarts and the format is refused,
 * so the caller goes back to 8 bits. */
int vkp_framegen_format_ok(VkFormat fmt);

#endif
