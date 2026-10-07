/* Runs the X11 renderer's frame-generation engines on the compositor's Turnip device. */
#include "vk_dispatch.h"

#include "framegen_engine.h"
#include "framegen_bridge.h"
#include "vkr_dis.h"
#include "lsfg/vkr_lsfg.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DIS_FLOW_MIN_SIDE_DEFAULT 180u

static VkPhysicalDevice g_pd;
static VkDevice g_dev;
static VkrLsfg *g_lsfg;
static VkrDis *g_dis;
static int g_kind = -1;
static int g_chain_ok;
static int g_chain_failed;

static VkImage g_source;
static uint64_t g_last_plan_ns;

static uint32_t g_multiplier = 2;
static float g_flow_scale = 0.7f;
static float g_refresh_hz;
static uint32_t g_dis_min_side = DIS_FLOW_MIN_SIDE_DEFAULT;
static uint32_t g_target_fps;

static int g_caps_lsfg, g_caps_dis;
static const char *g_reason = "not probed";

static int format_supports(VkFormat fmt, VkFormatFeatureFlags want) {
    if (!g_pd || !vkd.GetPhysicalDeviceFormatProperties) return 0;
    VkFormatProperties props;
    memset(&props, 0, sizeof(props));
    vkd.GetPhysicalDeviceFormatProperties(g_pd, fmt, &props);
    return (props.optimalTilingFeatures & want) == want;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

const void *fge_probe(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd,
                      VkFormat ring_fmt) {
    g_pd = pd;
    if (!vkd_bind_proc(gipa, inst)) {
        g_caps_lsfg = g_caps_dis = 0;
        g_reason = "the compositor's Vulkan entry points could not be resolved";
        return NULL;
    }

    const VkFormatFeatureFlags ring_want =
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
            | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if (!format_supports(ring_fmt, ring_want)) {
        g_caps_lsfg = g_caps_dis = 0;
        g_reason = "the driver cannot store to the generation ring format";
        return NULL;
    }

    /* LSFG keeps its motion pyramid in half-float; DIS needs only the ring. */
    g_caps_lsfg = format_supports(VK_FORMAT_R16G16B16A16_SFLOAT,
                                  VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
                                  | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
    g_caps_dis = 1;
    g_reason = g_caps_lsfg ? "supported" : "the driver has no storage half-float images";

    /* Neither engine needs a feature enabled at device creation. */
    return NULL;
}

void fge_device_ready(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd,
                      VkDevice dev, VkQueue queue, uint32_t qfam, int features_enabled) {
    (void)queue; (void)qfam; (void)features_enabled;
    g_pd = pd;
    g_dev = dev;
    if (!vkd.CreateInstance) vkd_bind_proc(gipa, inst);
}

int fge_caps_ok(int kind) {
    if (!g_dev) return 0;
    return kind == VKP_FG_ENGINE_DIS ? g_caps_dis : g_caps_lsfg;
}

const char *fge_caps_reason(void) { return g_reason; }

int fge_format_ok(VkFormat fmt) {
    return format_supports(fmt, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
                                | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
}

static void apply_config(void) {
    if (g_lsfg) {
        vkr_lsfg_configure(g_lsfg, g_multiplier, g_target_fps, g_flow_scale, g_refresh_hz, 0.0f);
    } else if (g_dis) {
        vkr_dis_configure(g_dis, g_dis_min_side, g_target_fps, g_refresh_hz);
    }
}

int fge_start(int kind, const char *cache_path) {
    fge_stop();
    if (!g_dev || !g_pd) return -1;

    if (kind == VKP_FG_ENGINE_DIS) {
        g_dis = vkr_dis_create(g_dev, g_pd);
        if (!g_dis) return -1;
    } else {
        if (!cache_path || !*cache_path) return -1;
        g_lsfg = vkr_lsfg_create(g_dev, g_pd, cache_path);
        if (!g_lsfg) return -1;
    }
    g_kind = kind;
    g_chain_ok = 0;
    g_chain_failed = 0;
    g_last_plan_ns = 0;
    apply_config();
    return 0;
}

void fge_stop(void) {
    if (g_lsfg) { vkr_lsfg_destroy(g_lsfg); g_lsfg = NULL; }
    if (g_dis) { vkr_dis_destroy(g_dis); g_dis = NULL; }
    g_kind = -1;
    g_chain_ok = 0;
    g_chain_failed = 0;
    g_source = VK_NULL_HANDLE;
}

int fge_unavailable(void) { return (!g_lsfg && !g_dis) || g_chain_failed; }

void fge_configure(uint32_t multiplier, float flow_scale, float refresh_hz, int flow_min_side,
                   int target_fps) {
    g_multiplier = multiplier ? multiplier : 2u;
    g_flow_scale = flow_scale > 0.0f ? flow_scale : 0.7f;
    g_refresh_hz = refresh_hz;
    if (flow_min_side > 0) g_dis_min_side = (uint32_t)flow_min_side;
    g_target_fps = target_fps > 0 ? (uint32_t)target_fps : 0u;
    apply_config();
}

int fge_prepare(uint32_t w, uint32_t h, VkFormat fmt) {
    if (!w || !h) return 0;

    if (g_lsfg) {
        if (g_chain_ok && !vkr_lsfg_needs_rebuild(g_lsfg, w, h, fmt)) return 1;
        vkr_lsfg_forget_targets(g_lsfg);
        vkr_lsfg_set_guest_extent(g_lsfg, w, h);
        g_chain_ok = vkr_lsfg_prepare(g_lsfg, w, h, fmt) ? 1 : 0;
    } else if (g_dis) {
        const VkrDisContentRect content = {0, 0, w, h};
        if (g_chain_ok && !vkr_dis_needs_rebuild(g_dis, w, h, fmt, content)) return 1;
        vkr_dis_forget_targets(g_dis);
        g_chain_ok = vkr_dis_prepare(g_dis, w, h, fmt, content) ? 1 : 0;
    } else {
        return 0;
    }

    g_chain_failed = !g_chain_ok;
    return g_chain_ok;
}

uint32_t fge_plan(uint32_t capacity, uint64_t source_frames, float presented_rate) {
    (void)presented_rate;
    if (!g_chain_ok) return 0;
    if (g_lsfg) {
        vkr_lsfg_set_refresh_rate(g_lsfg, g_refresh_hz);
        return vkr_lsfg_plan(g_lsfg, capacity, source_frames);
    }
    if (g_dis) return vkr_dis_plan(g_dis, capacity, source_frames);
    return 0;
}

void fge_process(VkCommandBuffer cmd, VkImage source, uint32_t w, uint32_t h, uint32_t gens) {
    if (!g_chain_ok) return;
    g_source = source;
    if (g_lsfg) {
        vkr_lsfg_process(g_lsfg, cmd, source, w, h, gens);
        /* The pacer wants the source cadence; the compositor's own frame clock is the measure. */
        uint64_t now = now_ns();
        if (g_last_plan_ns) vkr_lsfg_note_frame(g_lsfg, now - g_last_plan_ns, gens);
        g_last_plan_ns = now;
    } else if (g_dis) {
        vkr_dis_process(g_dis, cmd, source, w, h, gens);
    }
}

void fge_generate(VkCommandBuffer cmd, uint32_t g, uint32_t count, VkImage img, VkImageView view,
                  uint32_t w, uint32_t h) {
    (void)count;
    if (!g_chain_ok) return;
    /* The ring slot is stable for the life of the chain, so it doubles as the descriptor key. */
    if (g_lsfg) {
        vkr_lsfg_generate_into(g_lsfg, cmd, g, g, img, view, w, h);
    } else if (g_dis) {
        vkr_dis_generate_into(g_dis, cmd, g, g, img, view, w, h, g_source);
    }
}

void fge_forget_targets(void) {
    if (g_lsfg) vkr_lsfg_forget_targets(g_lsfg);
    if (g_dis) vkr_dis_forget_targets(g_dis);
}

void fge_telemetry(float *accepted, float *source_rate, float *thermal) {
    *accepted = 0.0f;
    *source_rate = 0.0f;
    *thermal = -1.0f;
}

const char *fge_engine_name(int kind) {
    return kind == VKP_FG_ENGINE_DIS ? "DIS Native" : "LSFG Native";
}

const char *fge_build_info(void) { return "lsfg-native/dis-native"; }
