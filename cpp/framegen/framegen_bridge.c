/* Frame generation on the Wayland backend - see framegen_bridge.h. */
#define _POSIX_C_SOURCE 200809L
#include "framegen_bridge.h"
#include "framegen_engine.h"
#include "vk_loader.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <android/log.h>

void wnc_log(const char *tag, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#define FGLOG(...) wnc_log("framegen", __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "WNWayland", __VA_ARGS__)

/* ---- app controls (written from any thread, read on the compositor thread) ------------- */
static _Atomic int   g_kind = VKP_FG_ENGINE_LSFG;
static _Atomic int   g_armed;
static _Atomic int   g_mult = 2;
static _Atomic int   g_dis_min_side = 180;
static _Atomic int   g_target_fps;
static _Atomic int   g_cfg_dirty = 1;
static float g_flow = 0.8f, g_refresh_hz;          /* under g_lock */
static char *g_cache_path;                          /* under g_lock */
static _Atomic int   g_cache_gen;                   /* bumps when the path changes */
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

/* ---- device -------------------------------------------------------------------------- */
static VkDevice g_dev;
static VkInstance g_inst;
static PFN_vkGetInstanceProcAddr g_gipa;
static VkPhysicalDevice g_pd;
static VkQueue g_queue;
static uint32_t g_qfam;
static int g_dev_ready, g_dev_lost, g_features_enabled;
static VkPhysicalDeviceMemoryProperties g_memprops;
static struct {
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkDeviceWaitIdle DeviceWaitIdle;
} vk;

/* ---- engine + generation ring (compositor thread) ------------------------------------ */
static int g_engine_kind = -1;      /* kind of the running engine, -1 = none */
static int g_engine_ok;
/* The engine of this kind (with this cache) could not start or build its chain; not retried
 * every frame - a new engine selection or a new cache clears it. */
static _Atomic int g_failed_kind = -1;
static _Atomic int g_failed_cache_gen = -1;

struct ring_image { VkImage img; VkDeviceMemory mem; VkImageView view; int fresh; };
static struct ring_image g_ring[VKP_FG_MAX_GENERATIONS];
static uint32_t g_ring_n, g_ring_w, g_ring_h;
static VkFormat g_ring_fmt;

static uint64_t g_source_frames;
static uint32_t g_planned;          /* generations planned for the current source frame */
static int g_was_armed, g_generating_logged;
static uint32_t g_built_w, g_built_h;
static VkFormat g_built_fmt;
/* Deeper formats (FP16 for HDR frames) the engine failed to build its chain in this session. */
static VkFormat g_refused_fmt[4];
static int g_refused_n;
static int format_refused(VkFormat f) {
    for (int i = 0; i < g_refused_n; i++) if (g_refused_fmt[i] == f) return 1;
    return 0;
}

/* ---- stats ------------------------------------------------------------------------- */
static _Atomic unsigned g_stat_generated;           /* since the last stats_take */
static _Atomic uint64_t g_total_generated;
static _Atomic uint64_t g_total_presented;
static float g_presented_rate, g_source_rate;       /* smoothed, 0.5 s windows */
static uint32_t g_present_accum, g_source_accum;
static int64_t g_window_start_ns;

static int64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* ====================================================================== app controls */

void vkp_framegen_set_engine(int kind) {
    if (kind != VKP_FG_ENGINE_LSFG && kind != VKP_FG_ENGINE_DIS) kind = VKP_FG_ENGINE_LSFG;
    atomic_store(&g_kind, kind);
    atomic_store(&g_cfg_dirty, 1);
}

void vkp_framegen_set_armed(int armed, int multiplier) {
    if (multiplier < 2) multiplier = 2;
    if (multiplier > VKP_FG_LSFG_MAX_GENERATIONS + 1) multiplier = VKP_FG_LSFG_MAX_GENERATIONS + 1;
    if (atomic_exchange(&g_mult, multiplier) != multiplier) atomic_store(&g_cfg_dirty, 1);
    atomic_store(&g_armed, armed ? 1 : 0);
    LOGD("framegen: set_armed(%d, x%d)", armed, multiplier);
}

void vkp_framegen_set_lsfg_cache_path(const char *path) {
    pthread_mutex_lock(&g_lock);
    int changed = (path == NULL) != (g_cache_path == NULL) ||
                  (path && g_cache_path && strcmp(path, g_cache_path) != 0);
    if (changed) {
        free(g_cache_path);
        g_cache_path = path ? strdup(path) : NULL;
        atomic_fetch_add(&g_cache_gen, 1);
    }
    pthread_mutex_unlock(&g_lock);
}

void vkp_framegen_set_tuning(float flow_scale, float refresh_hz) {
    if (flow_scale < 0.25f) flow_scale = 0.25f;
    if (flow_scale > 1.0f) flow_scale = 1.0f;
    pthread_mutex_lock(&g_lock);
    g_flow = flow_scale;
    g_refresh_hz = refresh_hz > 1.0f ? refresh_hz : 0.0f;
    pthread_mutex_unlock(&g_lock);
    atomic_store(&g_cfg_dirty, 1);
}

void vkp_framegen_set_engine_tuning(int flow_min_side, int target_fps) {
    if (flow_min_side < 64) flow_min_side = 64;
    if (flow_min_side > 1080) flow_min_side = 1080;
    if (target_fps < 0) target_fps = 0;
    atomic_store(&g_dis_min_side, flow_min_side);
    atomic_store(&g_target_fps, target_fps);
    atomic_store(&g_cfg_dirty, 1);
}

static int start_failed_for(int kind) {
    return atomic_load(&g_failed_kind) == kind && atomic_load(&g_failed_cache_gen) == atomic_load(&g_cache_gen);
}

static void mark_failed(int kind) {
    atomic_store(&g_failed_cache_gen, atomic_load(&g_cache_gen));
    atomic_store(&g_failed_kind, kind);
}

/* Same codes as VulkanRenderer.getFrameGenProblem(); computed from the live state so an engine
 * switch from the app is answered at once. */
int vkp_framegen_problem(void) {
    if (!g_dev_ready) return -1;
    int kind = atomic_load(&g_kind);
    if (!fge_caps_ok(kind)) return 1;
    if (start_failed_for(kind)) return 2;
    if (g_engine_ok && g_engine_kind == kind && fge_unavailable()) return 2;
    return 0;
}
const char *vkp_framegen_caps_reason(void) { return fge_caps_reason(); }

void vkp_framegen_stats(float out[6]) {
    float accepted = 0.0f, src = 0.0f, thermal = -1.0f;
    if (g_engine_ok) fge_telemetry(&accepted, &src, &thermal);
    /* Planned is what is generated; the governor is off on both paths. Our own measured
     * source rate serves both engines (DIS measures none). */
    out[0] = (float)g_planned;
    out[1] = (float)g_planned;
    out[2] = g_source_rate;
    out[3] = g_presented_rate;
    out[4] = thermal;
    out[5] = -1.0f;
}

unsigned vkp_framegen_stats_take(void) { return atomic_exchange(&g_stat_generated, 0u); }

uint64_t vkp_framegen_total_presented(void) { return atomic_load(&g_total_presented); }
uint64_t vkp_framegen_total_generated(void) { return atomic_load(&g_total_generated); }

/* ====================================================================== device setup */

const void *vkp_framegen_device_features(VkInstance inst, PFN_vkGetInstanceProcAddr gipa,
                                         VkPhysicalDevice pd) {
    g_pd = pd; g_inst = inst; g_gipa = gipa;
    return fge_probe(gipa, inst, pd, VK_FORMAT_R8G8B8A8_UNORM);
}

void vkp_framegen_device_ready(VkDevice dev, VkQueue queue, uint32_t qfam, int features_enabled) {
    g_dev = dev; g_queue = queue; g_qfam = qfam; g_features_enabled = features_enabled;
#define R(n) vk.n = (PFN_vk##n)g_vk.GetDeviceProcAddr(dev, "vk" #n)
    R(CreateImage); R(DestroyImage); R(CreateImageView); R(DestroyImageView);
    R(GetImageMemoryRequirements); R(AllocateMemory); R(FreeMemory); R(BindImageMemory);
    R(CmdPipelineBarrier); R(DeviceWaitIdle);
#undef R
    g_vk.GetPhysicalDeviceMemoryProperties(g_pd, &g_memprops);
    fge_device_ready(g_gipa, g_inst, g_pd, dev, queue, qfam, features_enabled);
    g_dev_ready = vk.CreateImage && vk.CreateImageView && vk.CmdPipelineBarrier;
    FGLOG("engines ready on the compositor's device: LSFG Native %s (%s), DIS Native %s",
          fge_caps_ok(VKP_FG_ENGINE_LSFG) ? "available" : "unavailable", fge_caps_reason(),
          fge_caps_ok(VKP_FG_ENGINE_DIS) ? "available" : "unavailable");
}

void vkp_framegen_device_lost(void) { g_dev_lost = 1; }

/* ====================================================================== ring */

static void destroy_ring(void) {
    if (!g_ring_n) return;
    if (vk.DeviceWaitIdle) vk.DeviceWaitIdle(g_dev);
    for (uint32_t i = 0; i < g_ring_n; i++) {
        if (g_ring[i].view) vk.DestroyImageView(g_dev, g_ring[i].view, NULL);
        if (g_ring[i].img) vk.DestroyImage(g_dev, g_ring[i].img, NULL);
        if (g_ring[i].mem) vk.FreeMemory(g_dev, g_ring[i].mem, NULL);
        memset(&g_ring[i], 0, sizeof(g_ring[i]));
    }
    g_ring_n = 0;
    if (g_engine_ok) fge_forget_targets();
}

static int memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (g_memprops.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    return -1;
}

/* The generation targets: written by the chain through a storage view, read by the screen
 * blit. Kept in GENERAL for their whole life (both uses allow it), so there is no layout dance
 * per frame - only access barriers. */
static int ensure_ring(uint32_t w, uint32_t h, VkFormat fmt, uint32_t n) {
    if (n > VKP_FG_MAX_GENERATIONS) n = VKP_FG_MAX_GENERATIONS;
    if (g_ring_n == n && g_ring_w == w && g_ring_h == h && g_ring_fmt == fmt) return 1;
    destroy_ring();
    for (uint32_t i = 0; i < n; i++) {
        struct ring_image *r = &g_ring[i];
        VkImageCreateInfo ici = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        if (vk.CreateImage(g_dev, &ici, NULL, &r->img) != VK_SUCCESS) { destroy_ring(); return 0; }
        VkMemoryRequirements req;
        vk.GetImageMemoryRequirements(g_dev, r->img, &req);
        int idx = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    .allocationSize = req.size, .memoryTypeIndex = (uint32_t)idx};
        if (idx < 0 || vk.AllocateMemory(g_dev, &mai, NULL, &r->mem) != VK_SUCCESS) { destroy_ring(); return 0; }
        vk.BindImageMemory(g_dev, r->img, r->mem, 0);
        VkImageViewCreateInfo vci = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = r->img,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vk.CreateImageView(g_dev, &vci, NULL, &r->view) != VK_SUCCESS) { destroy_ring(); return 0; }
        r->fresh = 1;
        g_ring_n = i + 1;
    }
    g_ring_w = w; g_ring_h = h; g_ring_fmt = fmt;
    LOGD("framegen: generation ring %ux%u x%u", w, h, n);
    return 1;
}

/* ====================================================================== present path */

static int engine_usable(void) {
    int kind = atomic_load(&g_kind);
    return g_dev_ready && !g_dev_lost && fge_caps_ok(kind) && !start_failed_for(kind);
}

int vkp_framegen_format_ok(VkFormat fmt) {
    if (fmt == VK_FORMAT_R8G8B8A8_UNORM) return 1;
    if (!g_dev_ready || g_dev_lost || format_refused(fmt)) return 0;
    return fge_format_ok(fmt);
}

/* The engine could not build its chain in a deeper format: refuse the format for the session and
 * restart the engine on the next frame (a failed build is sticky inside both engines). */
static void refuse_format(VkFormat fmt, int kind, int w, int h, const char *what) {
    if (g_refused_n < (int)(sizeof(g_refused_fmt) / sizeof(g_refused_fmt[0]))) g_refused_fmt[g_refused_n++] = fmt;
    FGLOG("%s could not %s in format %d (the HDR picture's FP16) at %dx%d: it restarts, and HDR frames go through "
          "it in 8 bits from the next frame", fge_engine_name(kind), what, (int)fmt, w, h);
    destroy_ring();
    fge_stop();
    g_engine_ok = 0; g_engine_kind = -1;
}

int vkp_framegen_active(void) {
    return atomic_load(&g_armed) && engine_usable();
}

/* DIS fills the gap to the refresh rate itself; LSFG takes the multiplier's share, or with a
 * target rate lets its pacer pick the count. Either way, up to the engine's most. */
static uint32_t generation_slots(void) {
    if (atomic_load(&g_kind) == VKP_FG_ENGINE_DIS) return VKP_FG_MAX_GENERATIONS;
    if (atomic_load(&g_target_fps) > 0) return VKP_FG_LSFG_MAX_GENERATIONS;
    uint32_t m = (uint32_t)atomic_load(&g_mult);
    return m - 1 > VKP_FG_LSFG_MAX_GENERATIONS ? VKP_FG_LSFG_MAX_GENERATIONS : m - 1;
}

int vkp_framegen_extra_images(void) {
    if (!vkp_framegen_active()) return 0;
    return (int)generation_slots();
}

static int ensure_engine(int kind) {
    if (g_engine_ok && g_engine_kind == kind) return 1;
    if (start_failed_for(kind)) return 0;
    if (g_engine_ok) { destroy_ring(); fge_stop(); g_engine_ok = 0; g_engine_kind = -1; }
    char *path = NULL;
    pthread_mutex_lock(&g_lock);
    if (g_cache_path) path = strdup(g_cache_path);
    pthread_mutex_unlock(&g_lock);
    if (kind == VKP_FG_ENGINE_LSFG && !path) {
        FGLOG("LSFG Native can't start: no shader cache (import Lossless.dll in Settings)");
        mark_failed(kind);
        return 0;
    }
    int r = fge_start(kind, path);
    free(path);
    if (r != 0) {
        FGLOG("%s failed to start on the compositor's driver; frame generation stays off",
              fge_engine_name(kind));
        mark_failed(kind);
        return 0;
    }
    g_engine_ok = 1; g_engine_kind = kind;
    atomic_store(&g_cfg_dirty, 1);
    g_built_w = g_built_h = 0;
    FGLOG("%s engine ready (%s)", fge_engine_name(kind), fge_build_info());
    return 1;
}

static int g_logged_mult;
static void log_arm_transition(int armed, int kind, int mult) {
    if (armed == g_was_armed) {
        if (armed && mult != g_logged_mult) {
            g_logged_mult = mult;
            FGLOG("%s multiplier changed to x%d (%d interpolated frames per game frame)", fge_engine_name(kind), mult, mult - 1);
        }
        return;
    }
    g_was_armed = armed;
    if (armed) {
        g_logged_mult = mult;
        float flow, hz;
        pthread_mutex_lock(&g_lock); flow = g_flow; hz = g_refresh_hz; pthread_mutex_unlock(&g_lock);
        FGLOG("%s x%d armed (flow scale %.2f, panel %.0f Hz): real frames are presented one slot late "
              "with the interpolated frames ahead of them", fge_engine_name(kind), mult, (double)flow, (double)hz);
    } else {
        FGLOG("frame generation off (%llu frames generated this session)", (unsigned long long)atomic_load(&g_total_generated));
        g_generating_logged = 0;
    }
}

int vkp_framegen_run(VkCommandBuffer cmd, VkImage scene, VkImageView scene_view, int w, int h,
                     VkFormat fmt, VkImage gens[VKP_FG_MAX_GENERATIONS]) {
    (void)scene_view;
    int kind = atomic_load(&g_kind);
    int armed = atomic_load(&g_armed);
    int mult = atomic_load(&g_mult);
    g_planned = 0;
    if (!armed || !engine_usable() || w <= 0 || h <= 0) { log_arm_transition(0, kind, mult); return 0; }
    if (!ensure_engine(kind)) { log_arm_transition(0, kind, mult); return 0; }
    log_arm_transition(1, kind, mult);

    if (atomic_exchange(&g_cfg_dirty, 0)) {
        float flow, hz;
        pthread_mutex_lock(&g_lock); flow = g_flow; hz = g_refresh_hz; pthread_mutex_unlock(&g_lock);
        fge_configure((uint32_t)mult, flow, hz, atomic_load(&g_dis_min_side),
                      atomic_load(&g_target_fps));
    }
    if (!fge_prepare((uint32_t)w, (uint32_t)h, fmt)) {
        if (fmt != VK_FORMAT_R8G8B8A8_UNORM && fge_unavailable()) {
            refuse_format(fmt, kind, w, h, "build its chain");
            return 0;
        }
        if (fge_unavailable()) {
            FGLOG("%s could not build its chain at %dx%d; frame generation stays off", fge_engine_name(kind), w, h);
            mark_failed(kind);
        }
        return 0;
    }
    if (g_built_w != (uint32_t)w || g_built_h != (uint32_t)h || g_built_fmt != fmt) {
        g_built_w = (uint32_t)w; g_built_h = (uint32_t)h; g_built_fmt = fmt;
        g_generating_logged = 0;
    }
    uint32_t want = generation_slots();
    if (!ensure_ring((uint32_t)w, (uint32_t)h, fmt, want)) {
        if (fmt != VK_FORMAT_R8G8B8A8_UNORM) { refuse_format(fmt, kind, w, h, "make its generation ring"); return 0; }
        FGLOG("no memory for the generation ring (%dx%d x%u); frame generation stays off", w, h, want);
        mark_failed(kind);
        return 0;
    }

    uint32_t n_gen = fge_plan(g_ring_n, ++g_source_frames, g_presented_rate);
    if (n_gen > g_ring_n) n_gen = g_ring_n;
    g_planned = n_gen;

    /* The scene image was just written (blits, and whatever the effects chain did); make that
     * visible to the engine's copy (TRANSFER) and shaders (COMPUTE). Fresh ring images get their
     * one UNDEFINED -> GENERAL transition here; the others need TRANSFER-read -> COMPUTE-write. */
    VkImageMemoryBarrier bars[1 + VKP_FG_MAX_GENERATIONS];
    uint32_t nb = 0;
    bars[nb++] = (VkImageMemoryBarrier){
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = scene, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    for (uint32_t g = 0; g < n_gen; g++) {
        struct ring_image *r = &g_ring[g];
        bars[nb++] = (VkImageMemoryBarrier){
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = r->fresh ? 0 : VK_ACCESS_TRANSFER_READ_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = r->fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = r->img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        r->fresh = 0;
    }
    vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, NULL, 0, NULL, nb, bars);

    /* Frame N in; the shared passes (the whole flow pyramid) run only when generating. */
    fge_process(cmd, scene, (uint32_t)w, (uint32_t)h, n_gen);
    for (uint32_t g = 0; g < n_gen; g++) {
        fge_generate(cmd, g, n_gen, g_ring[g].img, g_ring[g].view, (uint32_t)w, (uint32_t)h);
        gens[g] = g_ring[g].img;
    }
    if (n_gen) {
        /* Compute writes -> the screen blits (this command buffer and the later per-present ones;
         * a barrier's first scope covers everything submitted earlier on the queue). */
        nb = 0;
        for (uint32_t g = 0; g < n_gen; g++)
            bars[nb++] = (VkImageMemoryBarrier){
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = g_ring[g].img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, NULL, 0, NULL, nb, bars);
        if (!g_generating_logged) {
            g_generating_logged = 1;
            FGLOG("generating: %s x%d at %dx%d (%u interpolated frame%s per game frame%s)",
                  fge_engine_name(kind), mult, w, h, n_gen, n_gen == 1 ? "" : "s",
                  fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? ", FP16: the HDR picture" : "");
        }
    }
    return (int)n_gen;
}

void vkp_framegen_presented(int generated) {
    if (generated < 0) generated = 0;
    int64_t now = now_ns();
    if (!g_window_start_ns) { g_window_start_ns = now; g_present_accum = 0; g_source_accum = 0; }
    g_present_accum += (uint32_t)generated + 1u;
    g_source_accum += 1u;
    atomic_fetch_add(&g_total_presented, (uint64_t)generated + 1u);
    if (generated) {
        atomic_fetch_add(&g_stat_generated, (unsigned)generated);
        atomic_fetch_add(&g_total_generated, (uint64_t)generated);
    }
    float elapsed = (float)(now - g_window_start_ns) / 1e9f;
    if (elapsed < 0.5f) return;                     /* half-second window, like the X11 renderer */
    float rate = (float)g_present_accum / elapsed, src = (float)g_source_accum / elapsed;
    g_presented_rate = g_presented_rate > 0.0f ? g_presented_rate + (rate - g_presented_rate) * 0.25f : rate;
    g_source_rate = g_source_rate > 0.0f ? g_source_rate + (src - g_source_rate) * 0.25f : src;
    g_window_start_ns = now;
    g_present_accum = 0;
    g_source_accum = 0;
}
