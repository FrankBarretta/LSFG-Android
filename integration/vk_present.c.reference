/* Android-surface render backend — see vk_present.h. Uses Turnip via vk_loader
 * (g_vk.*), not the process-default system Adreno driver. */
#define _POSIX_C_SOURCE 200809L
#include "vk_present.h"
#include "vk_loader.h"
#include "sc_layer.h"
#include "effects_chain.h"
#include "framegen_bridge.h"
#include "hdr_compose.h"
#include "blend_pass.h"
#include "color_mgmt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <poll.h>
#include <time.h>
#include <pthread.h>
#include <android/log.h>

#define TAG "WNWayland"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) wnc_log("error", __VA_ARGS__)
#define MOD_INVALID VKP_MOD_INVALID

struct vkp_image {
    VkImage image;
    VkDeviceMemory mem;
    int w, h;
    int dmabuf;               /* imported from a client; owned by the foreign queue family */
    int blit_dst;             /* layer mode: a pool buffer this backend blits into (never a source) */
    void *map;                /* shm images: persistently mapped linear memory */
    VkDeviceSize offset, row_pitch;
    int in_general;           /* shm images: moved from PREINITIALIZED to GENERAL */
    VkFormat fmt;
    int sampled;              /* created with SAMPLED usage: the alpha pass can read it */
};

static ANativeWindow *g_window;  /* the window frames go to; compositor thread only */
static char *g_driver_path, *g_library_name, *g_native_lib_dir;
static int g_dev_state;       /* 0 = not yet, 1 = ok, -1 = failed, -2 = lost (VK_ERROR_DEVICE_LOST) */
static VkInstance g_inst;
static VkPhysicalDevice g_pd;
static VkDevice g_dev;
static VkQueue g_queue;
static uint32_t g_qfam;
static VkCommandPool g_pool;
/* One command buffer + acquire/render-done semaphore pair per PRESENT. A scene frame is one
 * present, or up to 1 + VKP_FG_MAX_GENERATIONS with frame generation (the generated frames go
 * out first, the real frame last), each on its own slot so no semaphore ever carries two
 * pending signals. Slot 0 is also the layer-mode blit's command buffer. */
#define MAX_PRESENTS (1 + VKP_FG_MAX_GENERATIONS)
static VkCommandBuffer g_cmds[MAX_PRESENTS];
static VkSemaphore g_acqs[MAX_PRESENTS], g_rnds[MAX_PRESENTS];
static VkFence g_fence;
/* Extra swapchain images this swapchain was built with (frame generation: one per generated
 * frame, so all presents of a scene frame queue without waiting for a vblank). */
static int g_swap_extra;
/* Format of the compositor pass's images (the scene image below, the effects chain's targets and
 * the frame-generation ring): a blit converts every client format into it, the chains read RGBA. */
#define SCENE_FMT VK_FORMAT_R8G8B8A8_UNORM
static VkPhysicalDeviceMemoryProperties g_memprops;

static VkSurfaceKHR g_surface;
static VkSwapchainKHR g_swapchain;
/* The display's own rotation of everything we present (VkSurfaceCapabilitiesKHR::currentTransform),
 * captured when the swapchain is built. Not a panel or device allowlist: it is what the presentation
 * engine says it will do to our layers. */
static VkSurfaceTransformFlagBitsKHR g_surface_transform;
static int g_surface_transform_known;
static VkImage *g_images;
static uint32_t g_nimg;
static VkExtent2D g_extent;

static int g_first_frame_done; /* one-shot: fire wnc_on_first_frame() on first present */

/* The base surface under the display layers (vkp_base_black): 1 once a plain black frame has been
 * presented on the CURRENT swapchain and nothing else since. Every other present clears it, and so does
 * every swapchain teardown, so the next layer frame puts one black frame back first. While it is set,
 * a frame that lives entirely on the display layers draws and presents nothing here. */
static int g_base_black;
/* The layer path presents nothing on the base surface while the game keeps its layer, so the swapchain's
 * OUT_OF_DATE (a resized surface, a new transform) would never be seen there: check_surface_changed()
 * asks the surface directly instead, at most every 100 ms. caps.currentExtent at the last build. */
static VkExtent2D g_caps_extent;
static int64_t g_surface_checked_ns;

/* Letterbox bars: vkCmdClearColorImage clears whole images only, so the bars around the picture are
 * blitted from this small black image instead of clearing the whole output under a picture that covers
 * most of it. Created and cleared once with the device. 1 = ready, 0/-1 = not available (whole clear). */
#define BLACK_DIM 16
static VkImage g_black_img;
static VkDeviceMemory g_black_mem;
static int g_black_state;

/* The display's release fence of a layer buffer, as a GPU wait (VK_KHR_external_semaphore_fd): the
 * pool's next copy into that buffer waits for it on the GPU instead of the compositor thread polling it. */
static PFN_vkImportSemaphoreFdKHR g_import_sem_fd;
static VkSemaphore g_wait_sem;

/* The 10 s perf line's counters (vkp_perf_take). Compositor thread only. */
static struct vkp_perf g_perf;
static int64_t perf_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static void perf_add(int64_t *sum, int64_t *max, unsigned *count, int64_t d) {
    if (d < 0) d = 0;
    *sum += d;
    if (d > *max) *max = d;
    (*count)++;
}
/* vkWaitForFences on the compositor's own work, timed for the perf line. */
static VkResult timed_wait(VkFence fence, uint64_t timeout) {
    int64_t t0 = perf_now();
    VkResult r = g_vk.WaitForFences(g_dev, 1, &fence, VK_TRUE, timeout);
    perf_add(&g_perf.wait_ns, &g_perf.wait_max_ns, &g_perf.waits, perf_now() - t0);
    return r;
}

/* The scene image (effects path only): every draw composited 1:1 at scene size, the input of the
 * screen-effect chain (effects_chain.c) — and of frame generation once that lands. Recreated on a
 * scene size change; frames are fenced, so never while the GPU reads it. */
static struct { VkImage img; VkDeviceMemory mem; int w, h; } g_scene;

/* HDR composition (hdr_compose.h): the 10-bit image every draw is blitted into 1:1 in its own encoding
 * ("mixed"), and the 10-bit HDR picture the encode pass makes of it for the game's display layer. Only
 * ever created while an HDR game shares the scene with something the display layer cannot show alone. */
struct vkp_img_slot { VkImage img; VkDeviceMemory mem; int w, h; VkFormat fmt; };
static struct vkp_img_slot g_mixed, g_hdrscene, g_fgscene;
#define HDR_FMT VK_FORMAT_A2B10G10R10_UNORM_PACK32
/* Frame generation with HDR frames through an HDR10 swapchain: the scene, the effects and the engine's
 * frames in FP16 - the format lsfg-vk itself uses for HDR - where the engine can build its chain in it,
 * so the PQ picture keeps more than 8 bits through interpolation (framegen_bridge.h). */
#define FG_HDR_FMT VK_FORMAT_R16G16B16A16_SFLOAT

/* HDR10 swapchain (frame generation with an HDR game): VK_EXT_swapchain_colorspace is enabled on the
 * instance only when the session asked for HDR; the swapchain is built as A2B10G10R10 + HDR10_ST2084
 * while such frames are being presented and the Android surface lists that pair (its WSI then sets the
 * surface dataspace to BT2020_PQ). */
static int g_colorspace_ext;       /* the instance has VK_EXT_swapchain_colorspace */
static int g_swap_hdr_want;        /* the next swapchain should be HDR10 */
static int g_swap_is_hdr;          /* the live swapchain is HDR10 */
static int g_swap_hdr_unavailable; /* this surface lists no HDR10 format: tone-map instead (said once) */
static VkFormat g_swap_fmt;        /* the live swapchain's format */
/* VK_EXT_hdr_metadata (HDR sessions, where the device offers it): the HDR10 swapchain carries the game's
 * mastering metadata, sent once per swapchain and image description. */
static PFN_vkSetHdrMetadataEXT g_set_hdr_metadata;
static uint32_t g_swap_md_identity;
static int g_surface_formats_said;
static const char *vk_result_name(VkResult r);

/* The Android surface is created/destroyed on the app's UI thread while the compositor thread
 * may be inside a WSI call (acquire and present can block for a refresh or more). The UI thread
 * therefore never touches the swapchain: it leaves the new window here and returns at once; the
 * compositor thread picks it up before its next frame (vkp_apply_window_request). g_req_lock is
 * only ever held for these few assignments, never across a Vulkan call. */
static pthread_mutex_t g_req_lock = PTHREAD_MUTEX_INITIALIZER;
static ANativeWindow *g_req_window;
static int g_req_pending;

/* Scale mode + alignment (app values, see vk_present.h); written from any thread, read per frame. */
static volatile int g_mode = VKP_MODE_STRETCH, g_align = VKP_ALIGN_CENTER;

/* The scene -> output mapping of the last frame (compositor thread): scene pixel (x,y) lands at
 * output (off_x + x * kx, off_y + y * ky), clipped to the region rectangle. */
static struct {
    int scene_w, scene_h, out_w, out_h, mode, align;
    float kx, ky, off_x, off_y;
    int rx, ry, rw, rh;                     /* region the picture may occupy */
    int valid;
} g_map;

/* A failed swapchain creation is retried no sooner than this (the window may be mid-teardown),
 * and the failure is logged once per streak instead of every frame. */
static int64_t g_swap_retry_at_ns;
static int g_swap_fail_logged;

/* Implemented in waylandcomp_jni.c — notifies Java (dismiss launch overlay). */
extern void wnc_on_first_frame(void);

static char g_gpu_name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
const char *vkp_gpu_name(void) { return g_gpu_name; }

/* DRM fourccs name the channels of a little-endian 32-bit word: XRGB8888 is B,G,R,X in memory
 * (= VK B8G8R8A8) and XBGR8888 is R,G,B,X (= VK R8G8B8A8). Turnip's Wayland WSI sends XB24 for
 * R8G8B8A8 swapchains, so reading everything as BGRA swaps red and blue. */
static VkFormat drm_to_vk(uint32_t drm) {
    switch (drm) {
    case 0x34324241: /* AB24 */
    case 0x34324258: /* XB24 */
        return VK_FORMAT_R8G8B8A8_UNORM;
    case 0x30334241: /* AB30 */
    case 0x30334258: /* XB30 */
        return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 0x30335241: /* AR30 */
    case 0x30335258: /* XR30 */
        return VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    case 0x48344241: /* AB4H */
    case 0x48344258: /* XB4H */
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    default: /* AR24 / XR24 */
        return VK_FORMAT_B8G8R8A8_UNORM;
    }
}

void vk_present_set_driver(const char *driver_path, const char *library_name,
                           const char *native_lib_dir) {
    free(g_driver_path); free(g_library_name); free(g_native_lib_dir);
    g_driver_path = driver_path ? strdup(driver_path) : NULL;
    g_library_name = library_name ? strdup(library_name) : NULL;
    g_native_lib_dir = native_lib_dir ? strdup(native_lib_dir) : NULL;
}

/* Destroy and recreate every acquire/render-done semaphore. Done with the swapchain (after the
 * device went idle): a frame that acquired an image and then bailed out leaves its acquire
 * semaphore signalled with nothing to wait on it, and reusing it in the next acquire is invalid
 * (Adreno answers OUT_OF_DATE, which rebuilds again - a loop). Fresh objects cannot carry that. */
static void reset_sync(void) {
    VkSemaphoreCreateInfo semci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (int k = 0; k < MAX_PRESENTS; k++) {
        if (g_acqs[k]) g_vk.DestroySemaphore(g_dev, g_acqs[k], NULL);
        if (g_rnds[k]) g_vk.DestroySemaphore(g_dev, g_rnds[k], NULL);
        g_vk.CreateSemaphore(g_dev, &semci, NULL, &g_acqs[k]);
        g_vk.CreateSemaphore(g_dev, &semci, NULL, &g_rnds[k]);
    }
}

static void destroy_swapchain(void) {
    g_base_black = 0; /* a new swapchain starts with nothing on it */
    if (g_dev_state != 1) return;
    g_vk.DeviceWaitIdle(g_dev);
    reset_sync();
    for (uint32_t i = 0; i < g_nimg; i++) blendp_forget_image(g_images[i]);
    if (g_swapchain) g_vk.DestroySwapchainKHR(g_dev, g_swapchain, NULL);
    g_swapchain = VK_NULL_HANDLE;
    g_swap_md_identity = 0; /* a new swapchain gets the metadata again */
    if (g_surface) g_vk.DestroySurfaceKHR(g_inst, g_surface, NULL);
    g_surface = VK_NULL_HANDLE;
    free(g_images);
    g_images = NULL;
    g_nimg = 0;
}

void vk_present_set_window(ANativeWindow *window) {
    ANativeWindow *superseded = NULL;
    pthread_mutex_lock(&g_req_lock);
    /* Two requests before the compositor thread got to the first: the first window was never
     * used, so its reference is dropped here. */
    if (g_req_pending && g_req_window && g_req_window != window) superseded = g_req_window;
    g_req_window = window;
    g_req_pending = 1;
    pthread_mutex_unlock(&g_req_lock);
    if (superseded) ANativeWindow_release(superseded);
}

int vkp_apply_window_request(void) {
    ANativeWindow *w;
    pthread_mutex_lock(&g_req_lock);
    if (!g_req_pending) { pthread_mutex_unlock(&g_req_lock); return 0; }
    w = g_req_window;
    g_req_window = NULL;
    g_req_pending = 0;
    pthread_mutex_unlock(&g_req_lock);
    if (w == g_window) {
        /* The same window handed over again (ANativeWindow_fromSurface adds a reference each time). */
        if (w) ANativeWindow_release(w);
        return 0;
    }
    sc_layer_window_gone();  /* the layer (if any) belongs to the old window */
    destroy_swapchain(); /* recreated against the new window on the next frame */
    g_swap_hdr_unavailable = 0; /* a new surface (another display, say) may offer HDR10 */
    if (g_window) ANativeWindow_release(g_window);
    g_window = w;
    g_swap_retry_at_ns = 0;
    g_swap_fail_logged = 0;
    wnc_log("gpu", w ? "screen surface attached" : "screen surface gone: presenting paused");
    return 1;
}

int vkp_has_window(void) {
    int r;
    pthread_mutex_lock(&g_req_lock);
    r = g_req_pending ? g_req_window != NULL : g_window != NULL;
    pthread_mutex_unlock(&g_req_lock);
    return r;
}

int vkp_device_lost(void) { return g_dev_state == -2; }

void vk_present_set_scale_mode(int mode, int alignment) {
    if (mode < VKP_MODE_OFF || mode > VKP_MODE_INTEGER) mode = VKP_MODE_FIT;
    if (alignment < VKP_ALIGN_CENTER || alignment > VKP_ALIGN_BOTTOM) alignment = VKP_ALIGN_CENTER;
    g_mode = mode;
    g_align = alignment;
}

static const char *mode_name(int m) {
    switch (m) {
    case VKP_MODE_OFF: return "off (letterbox)";
    case VKP_MODE_FIT: return "fit";
    case VKP_MODE_STRETCH: return "stretch";
    case VKP_MODE_FILL: return "fill";
    case VKP_MODE_INTEGER: return "integer";
    default: return "?";
    }
}
static const char *align_name(int a) {
    switch (a) {
    case VKP_ALIGN_TOP: return "top";
    case VKP_ALIGN_BOTTOM: return "bottom";
    default: return "center";
    }
}

/* Mirror of ViewTransformation.update(outer = output, inner = scene, mode, alignment) — keep the
 * arithmetic identical: the app maps touch input through that class with the same inputs, so any
 * difference here puts the pointer beside what it is pointing at. OFF and FIT are both an
 * aspect-preserving letterbox (OFF only differs in the app's fullscreen gates); TOP/BOTTOM confine
 * the picture to the top/bottom half of the output (the app's handheld split, #413). */
static void update_map(int scene_w, int scene_h) {
    int mode = g_mode, align = g_align;
    int W = (int)g_extent.width, H = (int)g_extent.height;
    if (g_map.valid && g_map.scene_w == scene_w && g_map.scene_h == scene_h && g_map.out_w == W &&
        g_map.out_h == H && g_map.mode == mode && g_map.align == align)
        return;
    g_map.scene_w = scene_w; g_map.scene_h = scene_h; g_map.out_w = W; g_map.out_h = H;
    g_map.mode = mode; g_map.align = align;

    int half = H / 2;
    switch (align) {
    case VKP_ALIGN_TOP:    g_map.rx = 0; g_map.ry = 0;    g_map.rw = W; g_map.rh = half; break;
    case VKP_ALIGN_BOTTOM: g_map.rx = 0; g_map.ry = half; g_map.rw = W; g_map.rh = H - half; break;
    default:               g_map.rx = 0; g_map.ry = 0;    g_map.rw = W; g_map.rh = H; break;
    }
    float sx = (float)g_map.rw / scene_w, sy = (float)g_map.rh / scene_h;
    if (mode == VKP_MODE_STRETCH) {
        g_map.kx = sx; g_map.ky = sy;
        g_map.off_x = (float)g_map.rx; g_map.off_y = (float)g_map.ry;
    } else {
        float aspect;
        if (mode == VKP_MODE_FILL) aspect = sx > sy ? sx : sy;
        else if (mode == VKP_MODE_INTEGER) {
            float m = sx < sy ? sx : sy;
            aspect = (float)(int)m; /* floor for m >= 1 */
            if (aspect < 1.0f) aspect = 1.0f;
        } else aspect = sx < sy ? sx : sy;
        g_map.kx = g_map.ky = aspect;
        /* Same integer truncation as ViewTransformation's viewOffsetX/Y. */
        g_map.off_x = (float)(g_map.rx + (int)((g_map.rw - scene_w * aspect) * 0.5f));
        g_map.off_y = (float)(g_map.ry + (int)((g_map.rh - scene_h * aspect) * 0.5f));
    }
    g_map.valid = 1;
    wnc_log("screen", "%s, %s: %dx%d scene shown %dx%d at %d,%d on the %dx%d output",
               mode_name(mode), align_name(align), scene_w, scene_h,
               (int)(scene_w * g_map.kx + 0.5f), (int)(scene_h * g_map.ky + 0.5f),
               (int)g_map.off_x, (int)g_map.off_y, W, H);
}

int vkp_output_to_scene(double ox, double oy, double *sx, double *sy) {
    if (!g_map.valid || g_map.kx <= 0 || g_map.ky <= 0) return 0;
    *sx = (ox - g_map.off_x) / g_map.kx;
    *sy = (oy - g_map.off_y) / g_map.ky;
    return 1;
}

void vkp_output_size(int *w, int *h) {
    *w = (int)g_extent.width;
    *h = (int)g_extent.height;
}

static void init_black_image(void);

static int has_ext(VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

/* Instance + device + command objects. Doesn't need the window. */
static int dev_init(void) {
    if (g_dev_state != 0) return g_dev_state == 1 ? 0 : -1;

    /* Load Turnip (adrenotools) and its entry points — NOT the system driver. */
    if (vk_loader_open(g_driver_path, g_library_name, g_native_lib_dir) != 0) {
        LOGE("present: vk_loader_open failed"); g_dev_state = -1; return -1;
    }

    const char *inst_exts[3] = {VK_KHR_SURFACE_EXTENSION_NAME,
                                VK_KHR_ANDROID_SURFACE_EXTENSION_NAME, NULL};
    uint32_t n_inst_exts = 2;
    /* HDR sessions only (color_mgmt.h): the colour-space extension lets the swapchain carry HDR10 for
     * frame generation. Asked for only when the loader has it, and only when HDR was asked for, so
     * every other session creates exactly the instance it always did. */
    if (wnc_color_requested() && g_vk.EnumerateInstanceExtensionProperties) {
        uint32_t ne = 0;
        g_vk.EnumerateInstanceExtensionProperties(NULL, &ne, NULL);
        VkExtensionProperties *ie = ne ? calloc(ne, sizeof(*ie)) : NULL;
        if (ie && g_vk.EnumerateInstanceExtensionProperties(NULL, &ne, ie) == VK_SUCCESS) {
            for (uint32_t i = 0; i < ne; i++)
                if (!strcmp(ie[i].extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) {
                    inst_exts[n_inst_exts++] = VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME;
                    g_colorspace_ext = 1;
                    break;
                }
        }
        free(ie);
        wnc_log("color", "compositor instance %s VK_EXT_swapchain_colorspace (an HDR10 swapchain for frame "
                   "generation %s)", g_colorspace_ext ? "enables" : "has no",
                   g_colorspace_ext ? "is possible where the screen surface offers one" : "is not possible: tone-mapped instead");
    }
    /* 1.3 like the X11 renderer's instance: the frame-generation probe (framegen_engine.cpp)
     * queries VkPhysicalDeviceVulkan12Features, and a 1.1 instance may be answered as 1.1. */
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "wn-wayland-present",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app,
                                .enabledExtensionCount = n_inst_exts,
                                .ppEnabledExtensionNames = inst_exts};
    if (g_vk.CreateInstance(&ici, NULL, &g_inst) != VK_SUCCESS) {
        LOGE("present: vkCreateInstance failed"); g_dev_state = -1; return -1;
    }
    vk_loader_load_instance(g_inst);

    uint32_t npd = 0;
    g_vk.EnumeratePhysicalDevices(g_inst, &npd, NULL);
    if (!npd) { LOGE("present: no physical devices"); g_dev_state = -1; return -1; }
    VkPhysicalDevice pds[8]; if (npd > 8) npd = 8;
    g_vk.EnumeratePhysicalDevices(g_inst, &npd, pds);
    g_pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < npd && g_pd == VK_NULL_HANDLE; i++) {
        uint32_t nq = 0;
        g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &nq, NULL);
        VkQueueFamilyProperties qs[16]; if (nq > 16) nq = 16;
        g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &nq, qs);
        for (uint32_t q = 0; q < nq; q++)
            if (qs[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) { g_pd = pds[i]; g_qfam = q; break; }
    }
    if (g_pd == VK_NULL_HANDLE) { LOGE("present: no graphics queue"); g_dev_state = -1; return -1; }
    {
        VkPhysicalDeviceProperties props;
        g_vk.GetPhysicalDeviceProperties(g_pd, &props);
        snprintf(g_gpu_name, sizeof(g_gpu_name), "%s", props.deviceName);
        wnc_log("gpu", "compositor renders on %s with %s", props.deviceName,
                   g_library_name ? g_library_name : "the system Vulkan driver");
        if (g_driver_path) wnc_log("gpu", "driver folder %s", g_driver_path);
    }
    g_vk.GetPhysicalDeviceMemoryProperties(g_pd, &g_memprops);

    /* Verify the dmabuf-import extensions are present, and log any that are missing. */
    const char *dev_exts[7] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_external_memory_fd",
                               "VK_EXT_external_memory_dma_buf", "VK_EXT_image_drm_format_modifier",
                               "VK_KHR_image_format_list", NULL, NULL};
    uint32_t n_dev_exts = 5;
    uint32_t ne = 0;
    g_vk.EnumerateDeviceExtensionProperties(g_pd, NULL, &ne, NULL);
    VkExtensionProperties *exts = calloc(ne ? ne : 1, sizeof(*exts));
    g_vk.EnumerateDeviceExtensionProperties(g_pd, NULL, &ne, exts);
    for (unsigned i = 0; i < 5; i++)
        if (!has_ext(exts, ne, dev_exts[i]))
            LOGE("present: driver MISSING %s (dmabuf import will fail)", dev_exts[i]);
    /* HDR sessions only: the HDR10 swapchain (frame generation) can carry the game's metadata. */
    int want_hdr_md = 0;
    if (wnc_color_requested()) {
        want_hdr_md = has_ext(exts, ne, VK_EXT_HDR_METADATA_EXTENSION_NAME);
        if (want_hdr_md) dev_exts[n_dev_exts++] = VK_EXT_HDR_METADATA_EXTENSION_NAME;
    }
    /* The layer pool's release fences as GPU waits (sync_file -> semaphore): optional, the pool falls
     * back to waiting for them on the CPU without it. */
    const int want_sem_fd = has_ext(exts, ne, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    if (want_sem_fd) dev_exts[n_dev_exts++] = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
    free(exts);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = g_qfam, .queueCount = 1, .pQueuePriorities = &prio};
    /* Frame generation (framegen_bridge.c): the LSFG chain needs the memory-model and
     * storage-image features enabled at device creation; the probe hands back the pNext chain
     * for them, or NULL on a device that cannot run it (then DIS Native alone is offered). A
     * driver that rejects the chain costs nothing: retried once without it, exactly as before. */
    const void *fg_features = vkp_framegen_device_features(g_inst, vk_loader_gipa(), g_pd);
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = fg_features,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = n_dev_exts, .ppEnabledExtensionNames = dev_exts};
    VkResult dr = g_vk.CreateDevice(g_pd, &dci, NULL, &g_dev);
    if (dr != VK_SUCCESS && fg_features) {
        wnc_log("framegen", "the driver refused the LSFG feature set (%s); device created without it",
                   vk_result_name(dr));
        dci.pNext = NULL;
        fg_features = NULL;
        dr = g_vk.CreateDevice(g_pd, &dci, NULL, &g_dev);
    }
    if (dr != VK_SUCCESS) {
        LOGE("present: vkCreateDevice failed"); g_dev_state = -1; return -1;
    }
    vk_loader_load_device(g_dev);
    g_vk.GetDeviceQueue(g_dev, g_qfam, 0, &g_queue);
    if (want_sem_fd) {
        g_import_sem_fd = (PFN_vkImportSemaphoreFdKHR)g_vk.GetDeviceProcAddr(g_dev, "vkImportSemaphoreFdKHR");
        VkSemaphoreCreateInfo wsci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (!g_import_sem_fd || g_vk.CreateSemaphore(g_dev, &wsci, NULL, &g_wait_sem) != VK_SUCCESS) {
            g_import_sem_fd = NULL;
            g_wait_sem = VK_NULL_HANDLE;
        }
    }
    wnc_log("perf", "layer buffers: the display's release fences are waited for %s",
               g_import_sem_fd ? "on the GPU (VK_KHR_external_semaphore_fd)"
                               : "on the CPU (this driver has no VK_KHR_external_semaphore_fd)");
    if (want_hdr_md)
        g_set_hdr_metadata = (PFN_vkSetHdrMetadataEXT)g_vk.GetDeviceProcAddr(g_dev, "vkSetHdrMetadataEXT");
    if (wnc_color_requested())
        wnc_log("color", "compositor device %s VK_EXT_hdr_metadata (the HDR10 swapchain for frame generation %s)",
                   g_set_hdr_metadata ? "enables" : "has no",
                   g_set_hdr_metadata ? "carries the game's mastering metadata" : "goes without the game's metadata");

    VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                   .queueFamilyIndex = g_qfam};
    g_vk.CreateCommandPool(g_dev, &pci, NULL, &g_pool);
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = g_pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = MAX_PRESENTS};
    g_vk.AllocateCommandBuffers(g_dev, &cai, g_cmds);
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    g_vk.CreateFence(g_dev, &fci, NULL, &g_fence);

    vkp_effects_bind_device(g_dev, g_pd, &g_memprops);
    hdrc_bind_device(g_dev, &g_memprops);
    blendp_bind_device(g_dev);
    g_dev_state = 1;
    reset_sync();
    init_black_image();
    vkp_framegen_device_ready(g_dev, g_queue, g_qfam, fg_features != NULL);
    return 0;
}

int vkp_ready(void) { return dev_init(); }

static int swap_init_locked(void);

static int swap_init(void) {
    struct timespec ts;
    int64_t now;
    if (!g_window) return -1;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    now = (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    if (now < g_swap_retry_at_ns) return -1;
    if (swap_init_locked() == 0) {
        g_swap_retry_at_ns = 0;
        g_swap_fail_logged = 0;
        return 0;
    }
    destroy_swapchain();
    g_swap_retry_at_ns = now + 500000000LL;
    if (!g_swap_fail_logged) {
        g_swap_fail_logged = 1;
        LOGE("present: no swapchain on the screen surface; retrying every 0.5 s");
    }
    return -1;
}

#define SWLOGE(...) do { if (!g_swap_fail_logged) LOGE(__VA_ARGS__); } while (0)

/* Short names for the session log (the formats and colour spaces an Android surface lists). */
static void vk_format_short(VkFormat f, char *out, size_t n) {
    const char *s = NULL;
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: s = "RGBA8"; break;
    case VK_FORMAT_R8G8B8A8_SRGB: s = "RGBA8_SRGB"; break;
    case VK_FORMAT_B8G8R8A8_UNORM: s = "BGRA8"; break;
    case VK_FORMAT_B8G8R8A8_SRGB: s = "BGRA8_SRGB"; break;
    case VK_FORMAT_R5G6B5_UNORM_PACK16: s = "RGB565"; break;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: s = "A2B10G10R10"; break;
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32: s = "A2R10G10B10"; break;
    case VK_FORMAT_R16G16B16A16_SFLOAT: s = "RGBA16F"; break;
    default: break;
    }
    if (s) snprintf(out, n, "%s", s); else snprintf(out, n, "format%d", (int)f);
}
static int vk_format_bits(VkFormat f) {
    switch (f) {
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return 10;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 16;
    default: return 8;
    }
}
static const char *vk_colorspace_short(VkColorSpaceKHR c) {
    switch ((int)c) {
    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: return "sRGB";
    case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT: return "P3";
    case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT: return "extended-sRGB-linear";
    case VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT: return "P3-linear";
    case VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT: return "DCI-P3";
    case VK_COLOR_SPACE_BT709_LINEAR_EXT: return "BT709-linear";
    case VK_COLOR_SPACE_BT709_NONLINEAR_EXT: return "BT709";
    case VK_COLOR_SPACE_BT2020_LINEAR_EXT: return "BT2020-linear";
    case VK_COLOR_SPACE_HDR10_ST2084_EXT: return "HDR10";
    case VK_COLOR_SPACE_DOLBYVISION_EXT: return "DolbyVision";
    case VK_COLOR_SPACE_HDR10_HLG_EXT: return "HLG";
    case VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT: return "AdobeRGB-linear";
    case VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT: return "AdobeRGB";
    case VK_COLOR_SPACE_PASS_THROUGH_EXT: return "pass-through";
    case VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT: return "extended-sRGB";
    default: return NULL;
    }
}
/* One line: the total, how many colour spaces each format comes with, and every HDR-capable pair
 * (HDR10 / HLG / extended sRGB, any format) - never "the first N pairs". */
static int hdr_capable_space(VkColorSpaceKHR cs) {
    int c = (int)cs;
    return c == VK_COLOR_SPACE_HDR10_ST2084_EXT || c == VK_COLOR_SPACE_HDR10_HLG_EXT ||
           c == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT || c == VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT;
}
static void log_surface_formats(const VkSurfaceFormatKHR *f, uint32_t n) {
    char per[200] = "", hdr[256] = "";
    size_t pp = 0, hp = 0;
    unsigned hdr_n = 0, hdr_cut = 0;
    for (uint32_t i = 0; i < n; i++) {
        int first = 1;
        for (uint32_t j = 0; j < i; j++) if (f[j].format == f[i].format) { first = 0; break; }
        if (first && pp < sizeof(per) - 32) {
            unsigned cnt = 0;
            for (uint32_t j = i; j < n; j++) if (f[j].format == f[i].format) cnt++;
            char nm[24];
            vk_format_short(f[i].format, nm, sizeof(nm));
            pp += (size_t)snprintf(per + pp, sizeof(per) - pp, "%s%s x%u", pp ? ", " : "", nm, cnt);
        }
    }
    /* The deep formats (10-bit, FP16) first, so a long list never pushes them out of the line. */
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t i = 0; i < n; i++) {
            if (!hdr_capable_space(f[i].colorSpace) || (vk_format_bits(f[i].format) > 8) != (pass == 0)) continue;
            hdr_n++;
            char nm[24];
            vk_format_short(f[i].format, nm, sizeof(nm));
            if (hp < sizeof(hdr) - 48)
                hp += (size_t)snprintf(hdr + hp, sizeof(hdr) - hp, "%s%s/%s", hp ? ", " : "", nm,
                                       vk_colorspace_short(f[i].colorSpace));
            else hdr_cut++;
        }
    wnc_log("color", "screen surface lists %u format/colour-space pairs (%s); HDR-capable: %s%s", n, per,
               hdr_n ? hdr : "none", hdr_cut ? " (+more)" : "");
}

static int swap_init_locked(void) {

    VkAndroidSurfaceCreateInfoKHR aci = {
        .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, .window = g_window};
    if (g_vk.CreateAndroidSurfaceKHR(g_inst, &aci, NULL, &g_surface) != VK_SUCCESS) {
        SWLOGE("present: create android surface failed"); return -1;
    }
    VkBool32 sup = VK_FALSE;
    g_vk.GetPhysicalDeviceSurfaceSupportKHR(g_pd, g_qfam, g_surface, &sup);
    if (!sup) { SWLOGE("present: queue can't present to the window"); return -1; }

    VkSurfaceCapabilitiesKHR caps;
    g_vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, g_surface, &caps);
    /* Remembered for the layer path: currentTransform is the rotation the presentation engine (and
     * therefore SurfaceFlinger and the DPU) applies to everything we put on this display. Anything
     * but IDENTITY means every display layer we hand over is a ROTATED layer. sc_layer.c needs that
     * to decide whether a second layer is affordable — see sc_layer_present_overlay(). */
    g_surface_transform = caps.currentTransform;
    g_surface_transform_known = 1;
    g_caps_extent = caps.currentExtent; /* check_surface_changed() compares against this */
    g_surface_checked_ns = perf_now();
    /* Every format/colour-space pair the surface lists. With VK_EXT_swapchain_colorspace the Android
     * WSI lists each format once per colour space (the Fold: 11 per format), so a fixed-size array
     * silently drops the 10-bit and FP16 rows - which is where HDR10 lives. */
    uint32_t nfmt = 0;
    g_vk.GetPhysicalDeviceSurfaceFormatsKHR(g_pd, g_surface, &nfmt, NULL);
    VkSurfaceFormatKHR *fmts = nfmt ? calloc(nfmt, sizeof(*fmts)) : NULL;
    if (!fmts) { SWLOGE("present: the surface lists no formats"); return -1; }
    if (g_vk.GetPhysicalDeviceSurfaceFormatsKHR(g_pd, g_surface, &nfmt, fmts) < 0 || !nfmt) {
        free(fmts); SWLOGE("present: vkGetPhysicalDeviceSurfaceFormatsKHR failed"); return -1;
    }
    VkSurfaceFormatKHR chosen = fmts[0];
    const VkSurfaceFormatKHR sdr_choice = fmts[0];
    if (wnc_color_requested() && (!g_surface_formats_said || g_swap_hdr_want)) {
        g_surface_formats_said = 1;
        log_surface_formats(fmts, nfmt);
    }
    /* HDR10 for frame generation with an HDR game (render_impl sets g_swap_hdr_want): the best format
     * the surface lists with VK_COLOR_SPACE_HDR10_ST2084_EXT - 10-bit, then FP16, then 8-bit (still a
     * PQ BT.2020 picture, with less precision). None listed -> the frames are tone-mapped into the
     * ordinary swapchain, and that is said once. */
    g_swap_is_hdr = 0;
    if (g_swap_hdr_want) {
        static const VkFormat pref[] = {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32,
                                        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM,
                                        VK_FORMAT_B8G8R8A8_UNORM};
        int pick = -1;
        for (unsigned k = 0; k < sizeof(pref) / sizeof(pref[0]) && pick < 0; k++)
            for (uint32_t i = 0; i < nfmt; i++)
                if (fmts[i].colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT && fmts[i].format == pref[k]) {
                    pick = (int)i; break;
                }
        if (pick >= 0) {
            chosen = fmts[pick];
            g_swap_is_hdr = 1;
            char fname[24];
            vk_format_short(chosen.format, fname, sizeof(fname));
            wnc_log("color", "screen swapchain built as HDR10 (format %d %s, HDR10_ST2084): frame-generated HDR "
                       "frames reach the display as PQ BT.2020%s", (int)chosen.format, fname,
                       vk_format_bits(chosen.format) <= 8 ? " - an 8-bit swapchain: banding possible in smooth gradients"
                                                          : "");
        } else {
            g_swap_hdr_unavailable = 1;
            wnc_log("color", "the screen surface lists no HDR10 swapchain format among its %u format/colour-space "
                       "pairs (see the line above): frames with frame generation are tone-mapped to SDR instead",
                       nfmt);
        }
    }
    free(fmts);
    g_swap_fmt = chosen.format;

    g_extent = caps.currentExtent;
    if (g_extent.width == 0xFFFFFFFF) {
        g_extent.width = ANativeWindow_getWidth(g_window);
        g_extent.height = ANativeWindow_getHeight(g_window);
    }
    /* + one image per generated frame while frame generation is armed: every present of a scene
     * frame can then be queued at once and FIFO spaces them onto consecutive vblanks. */
    g_swap_extra = vkp_framegen_extra_images();
    uint32_t want = caps.minImageCount + 1 + (uint32_t)g_swap_extra;
    if (caps.maxImageCount && want > caps.maxImageCount) want = caps.maxImageCount;

    /* Use IDENTITY preTransform when the surface supports it. Setting preTransform =
     * currentTransform tells the presentation engine our content is ALREADY pre-rotated by
     * that amount — but our blit doesn't rotate, so on a device whose surface reports a 90°
     * currentTransform the display then rotates our upright frame 90° (game shows sideways).
     * IDENTITY = "don't rotate what I present", which is what we want. */
    VkSurfaceTransformFlagBitsKHR pretrans =
        (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
            ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;

    VkSwapchainCreateInfoKHR sci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = g_surface,
        .minImageCount = want, .imageFormat = chosen.format, .imageColorSpace = chosen.colorSpace,
        .imageExtent = g_extent, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = pretrans,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE};
    VkResult cr = g_vk.CreateSwapchainKHR(g_dev, &sci, NULL, &g_swapchain);
    if (cr != VK_SUCCESS && g_swap_is_hdr) {
        /* The surface listed the HDR10 pair but refused the swapchain: say so, never ask again this
         * surface, and build the ordinary one now (frames tone-mapped) - never a black screen. */
        wnc_log("color", "the driver refused the HDR10 swapchain (%s): frames with frame generation are "
                   "tone-mapped to SDR instead", vk_result_name(cr));
        g_swap_is_hdr = 0;
        g_swap_hdr_unavailable = 1;
        g_swapchain = VK_NULL_HANDLE;
        sci.imageFormat = sdr_choice.format;
        sci.imageColorSpace = sdr_choice.colorSpace;
        g_swap_fmt = sdr_choice.format;
        cr = g_vk.CreateSwapchainKHR(g_dev, &sci, NULL, &g_swapchain);
    }
    if (cr != VK_SUCCESS) {
        g_swapchain = VK_NULL_HANDLE;
        SWLOGE("present: vkCreateSwapchainKHR failed (%d)", (int)cr);
        return -1;
    }
    g_vk.GetSwapchainImagesKHR(g_dev, g_swapchain, &g_nimg, NULL);
    g_images = calloc(g_nimg, sizeof(VkImage));
    g_vk.GetSwapchainImagesKHR(g_dev, g_swapchain, &g_nimg, g_images);

    wnc_log("gpu", "screen output %ux%u, %u buffers, vsync%s", g_extent.width, g_extent.height, g_nimg,
               g_swap_extra ? " (frame generation: several presents per frame)" : "");
    return 0;
}

static int memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (g_memprops.memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

/* The small black image the letterbox bars are blitted from (see g_black_img): created, cleared and left
 * in TRANSFER_SRC_OPTIMAL once, with a one-time submit, right after the device comes up. Any failure only
 * means the bars are cleared the old way (the whole output). */
static void init_black_image(void) {
    g_black_state = -1;
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {BLACK_DIM, BLACK_DIM, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    if (g_vk.CreateImage(g_dev, &ici, NULL, &g_black_img) != VK_SUCCESS) { g_black_img = VK_NULL_HANDLE; return; }
    VkMemoryRequirements req;
    g_vk.GetImageMemoryRequirements(g_dev, g_black_img, &req);
    int idx = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (idx < 0) idx = memory_type(req.memoryTypeBits, 0);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                .memoryTypeIndex = (uint32_t)(idx < 0 ? 0 : idx)};
    if (idx < 0 || g_vk.AllocateMemory(g_dev, &mai, NULL, &g_black_mem) != VK_SUCCESS) {
        g_vk.DestroyImage(g_dev, g_black_img, NULL);
        g_black_img = VK_NULL_HANDLE; g_black_mem = VK_NULL_HANDLE;
        return;
    }
    g_vk.BindImageMemory(g_dev, g_black_img, g_black_mem, 0);

    VkCommandBuffer cmd = g_cmds[0];
    g_vk.ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    g_vk.BeginCommandBuffer(cmd, &bi);
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier to_dst = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_black_img, .subresourceRange = range,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, NULL, 0, NULL, 1, &to_dst);
    VkClearColorValue black = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
    g_vk.CmdClearColorImage(cmd, g_black_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    VkImageMemoryBarrier to_src = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_black_img, .subresourceRange = range,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, NULL, 0, NULL, 1, &to_src);
    g_vk.EndCommandBuffer(cmd);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    g_vk.ResetFences(g_dev, 1, &g_fence);
    if (g_vk.QueueSubmit(g_queue, 1, &si, g_fence) == VK_SUCCESS &&
        g_vk.WaitForFences(g_dev, 1, &g_fence, VK_TRUE, 1000000000ULL) == VK_SUCCESS)
        g_black_state = 1;
    else
        wnc_log("gpu", "letterbox black image unavailable: the whole output is cleared every copy-path frame");
}

/* Blit the black image over everything of swapchain image `img` (in TRANSFER_DST_OPTIMAL) OUTSIDE the
 * destination rectangle of `covered` - the letterbox bars around a picture that fills that rectangle
 * completely. Up to four rectangles, one blit command. 0 = not possible (no black image): the caller
 * clears the whole image instead. */
static int record_bars_clear(VkCommandBuffer cmd, VkImage img, const VkImageBlit *covered) {
    if (g_black_state != 1 || !covered) return 0;
    const int W = (int)g_extent.width, H = (int)g_extent.height;
    int x0 = covered->dstOffsets[0].x, x1 = covered->dstOffsets[1].x;
    int y0 = covered->dstOffsets[0].y, y1 = covered->dstOffsets[1].y;
    if (x0 > x1) { int t = x0; x0 = x1; x1 = t; }
    if (y0 > y1) { int t = y0; y0 = y1; y1 = t; }
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > W) x1 = W;
    if (y1 > H) y1 = H;
    if (x1 <= x0 || y1 <= y0) return 0;
    int rects[4][4], n = 0;
    if (y0 > 0) { rects[n][0] = 0;  rects[n][1] = 0;  rects[n][2] = W;  rects[n][3] = y0; n++; }  /* top */
    if (y1 < H) { rects[n][0] = 0;  rects[n][1] = y1; rects[n][2] = W;  rects[n][3] = H;  n++; }  /* bottom */
    if (x0 > 0) { rects[n][0] = 0;  rects[n][1] = y0; rects[n][2] = x0; rects[n][3] = y1; n++; }  /* left */
    if (x1 < W) { rects[n][0] = x1; rects[n][1] = y0; rects[n][2] = W;  rects[n][3] = y1; n++; }  /* right */
    if (!n) return 1; /* the picture covers the whole output: nothing to clear */
    VkImageBlit blits[4];
    for (int i = 0; i < n; i++)
        blits[i] = (VkImageBlit){.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                 .srcOffsets = {{0, 0, 0}, {BLACK_DIM, BLACK_DIM, 1}},
                                 .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                 .dstOffsets = {{rects[i][0], rects[i][1], 0}, {rects[i][2], rects[i][3], 1}}};
    g_vk.CmdBlitImage(cmd, g_black_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)n, blits, VK_FILTER_NEAREST);
    return 1;
}

const char *vkp_modifier_name(uint64_t modifier) {
    static char other[40];
    if (modifier == VKP_MOD_LINEAR) return "linear";
    if (modifier == VKP_MOD_QCOM_COMPRESSED) return "qcom_compressed";
    if (modifier == VKP_MOD_INVALID) return "implicit";
    snprintf(other, sizeof(other), "modifier %#llx", (unsigned long long)modifier);
    return other;
}

/* Can the driver create the image vkp_image_import_dmabuf() creates for this format+modifier
 * (2D, DRM_FORMAT_MODIFIER tiling, this usage, dma-buf memory) — and import it? */
static int modifier_importable(VkFormat fmt, uint64_t modifier, VkImageUsageFlags usage) {
    if (!g_vk.GetPhysicalDeviceImageFormatProperties2) return 1; /* can't ask; the create will tell */
    VkPhysicalDeviceExternalImageFormatInfo ext = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkPhysicalDeviceImageDrmFormatModifierInfoEXT mod = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT, .pNext = &ext,
        .drmFormatModifier = modifier, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkPhysicalDeviceImageFormatInfo2 info = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, .pNext = &mod,
        .format = fmt, .type = VK_IMAGE_TYPE_2D, .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = usage};
    VkExternalImageFormatProperties extp = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 props = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, .pNext = &extp};
    if (g_vk.GetPhysicalDeviceImageFormatProperties2(g_pd, &info, &props) != VK_SUCCESS) return 0;
    return (extp.externalMemoryProperties.externalMemoryFeatures &
            VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
}

int vkp_dmabuf_modifiers(uint32_t drm_format, uint64_t *out, int max) {
    if (max <= 0 || dev_init() != 0 || !g_vk.GetPhysicalDeviceFormatProperties2) return 0;
    VkFormat fmt = drm_to_vk(drm_format);
    VkDrmFormatModifierPropertiesListEXT list = {
        .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT};
    VkFormatProperties2 fp = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &list};
    g_vk.GetPhysicalDeviceFormatProperties2(g_pd, fmt, &fp);
    if (!list.drmFormatModifierCount) return 0;
    VkDrmFormatModifierPropertiesEXT *props = calloc(list.drmFormatModifierCount, sizeof(*props));
    if (!props) return 0;
    list.pDrmFormatModifierProperties = props;
    g_vk.GetPhysicalDeviceFormatProperties2(g_pd, fmt, &fp);

    int n = 0;
    for (uint32_t i = 0; i < list.drmFormatModifierCount; i++) {
        uint64_t m = props[i].drmFormatModifier;
        const char *why = NULL;
        if (m != VKP_MOD_LINEAR && m != VKP_MOD_QCOM_COMPRESSED) why = "unknown layout";
        else if (props[i].drmFormatModifierPlaneCount != 1) why = "not single-plane";
        else if (!(props[i].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) why = "no blit source";
        else if (!modifier_importable(fmt, m, VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) why = "not importable as a dma-buf";
        if (why) {
            LOGI("dmabuf: %c%c%c%c %s reported by the driver but not advertised: %s",
                 drm_format & 0xff, (drm_format >> 8) & 0xff, (drm_format >> 16) & 0xff,
                 (drm_format >> 24) & 0xff, vkp_modifier_name(m), why);
            continue;
        }
        if (n < max) out[n++] = m;
    }
    free(props);
    return n;
}

struct vkp_image *vkp_image_from_dmabuf(int fd, uint32_t drm_format, uint64_t modifier, int w, int h,
                                        uint32_t stride, uint32_t offset) {
    return vkp_image_import_dmabuf(fd, drm_format, modifier, w, h, stride, offset, 0);
}

int vkp_image_is_dmabuf(const struct vkp_image *img) { return img && img->dmabuf && !img->blit_dst; }

struct vkp_image *vkp_image_import_dmabuf(int fd, uint32_t drm_format, uint64_t modifier, int w, int h,
                                          uint32_t stride, uint32_t offset, int as_blit_dst) {
    if (modifier == MOD_INVALID || w <= 0 || h <= 0) return NULL;
    if (dev_init() != 0) return NULL;

    struct vkp_image *img = calloc(1, sizeof(*img));
    if (!img) return NULL;
    img->w = w; img->h = h; img->dmabuf = 1; img->blit_dst = as_blit_dst ? 1 : 0;
    img->fmt = drm_to_vk(drm_format);
    /* Sampled as well where the driver says this layout can be: only then can the alpha pass read it. */
    const VkImageUsageFlags sampled_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    img->sampled = !as_blit_dst && g_vk.GetPhysicalDeviceImageFormatProperties2 &&
                   modifier_importable(img->fmt, modifier, sampled_usage);

    VkSubresourceLayout plane = {.offset = offset, .rowPitch = stride};
    VkImageDrmFormatModifierExplicitCreateInfoEXT modInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = modifier, .drmFormatModifierPlaneCount = 1, .pPlaneLayouts = &plane};
    VkExternalMemoryImageCreateInfo extImg = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &modInfo,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &extImg,
        .imageType = VK_IMAGE_TYPE_2D, .format = img->fmt, .extent = {w, h, 1},
        .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = as_blit_dst ? VK_IMAGE_USAGE_TRANSFER_DST_BIT
                 : img->sampled ? sampled_usage : VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult cr = g_vk.CreateImage(g_dev, &ici, NULL, &img->image);
    if (cr != VK_SUCCESS) {
        LOGE("%s: vkCreateImage(%s, %dx%d, pitch %u, offset %u) -> %s%s", as_blit_dst ? "layer" : "dmabuf",
             vkp_modifier_name(modifier), w, h, stride, offset, vk_result_name(cr),
             (!as_blit_dst && modifier == VKP_MOD_QCOM_COMPRESSED)
                 ? ": the game's UBWC layout was refused; BANNER_WAYLAND_UBWC=0 forces linear buffers" : "");
        free(img); return NULL;
    }

    int dupfd = dup(fd);
    uint32_t allowed = 0xffffffff;
    if (g_vk.GetMemoryFdPropertiesKHR) {
        VkMemoryFdPropertiesKHR fp = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
        if (g_vk.GetMemoryFdPropertiesKHR(g_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                          dupfd, &fp) == VK_SUCCESS)
            allowed = fp.memoryTypeBits;
    }
    VkMemoryRequirements req;
    g_vk.GetImageMemoryRequirements(g_dev, img->image, &req);
    uint32_t bits = req.memoryTypeBits & allowed;
    int idx = -1;
    for (int i = 0; i < 32; i++) if (bits & (1u << i)) { idx = i; break; }
    if (idx < 0) {
        LOGE("%s: no memory type can import the %s dma-buf (image types %#x, fd types %#x)",
             as_blit_dst ? "layer" : "dmabuf", vkp_modifier_name(modifier), req.memoryTypeBits, allowed);
        g_vk.DestroyImage(g_dev, img->image, NULL); close(dupfd); free(img); return NULL;
    }

    VkImportMemoryFdInfoKHR imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                                   .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                   .fd = dupfd};
    VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                         .pNext = &imp, .image = img->image};
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                .allocationSize = req.size, .memoryTypeIndex = (uint32_t)idx};
    VkResult mr = g_vk.AllocateMemory(g_dev, &mai, NULL, &img->mem);
    if (mr != VK_SUCCESS) {
        LOGE("%s: importing the %s dma-buf (%dx%d, %llu bytes) -> %s", as_blit_dst ? "layer" : "dmabuf",
             vkp_modifier_name(modifier), w, h, (unsigned long long)req.size, vk_result_name(mr));
        g_vk.DestroyImage(g_dev, img->image, NULL); close(dupfd); free(img); return NULL;
    }
    VkResult br = g_vk.BindImageMemory(g_dev, img->image, img->mem, 0);
    if (br != VK_SUCCESS) {
        LOGE("%s: vkBindImageMemory(%s dma-buf) -> %s", as_blit_dst ? "layer" : "dmabuf",
             vkp_modifier_name(modifier), vk_result_name(br));
        g_vk.FreeMemory(g_dev, img->mem, NULL); g_vk.DestroyImage(g_dev, img->image, NULL);
        free(img); return NULL;
    }
    return img;
}

/* Whether the alpha pass can read a wl_shm image: linear B8G8R8A8, sampled with a linear filter. */
static int shm_sampled(void) {
    static int known;
    if (!known) {
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        VkFormatProperties2 fp = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};
        if (g_vk.GetPhysicalDeviceFormatProperties2)
            g_vk.GetPhysicalDeviceFormatProperties2(g_pd, VK_FORMAT_B8G8R8A8_UNORM, &fp);
        known = (fp.formatProperties.linearTilingFeatures & need) == need ? 1 : -1;
    }
    return known == 1;
}

struct vkp_image *vkp_image_create_shm(int w, int h) {
    if (w <= 0 || h <= 0 || dev_init() != 0) return NULL;

    struct vkp_image *img = calloc(1, sizeof(*img));
    if (!img) return NULL;
    img->w = w; img->h = h;
    img->fmt = VK_FORMAT_B8G8R8A8_UNORM;
    img->sampled = shm_sampled();

    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = img->fmt, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (img->sampled ? VK_IMAGE_USAGE_SAMPLED_BIT : 0), .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED};
    if (g_vk.CreateImage(g_dev, &ici, NULL, &img->image) != VK_SUCCESS) { free(img); return NULL; }

    VkMemoryRequirements req;
    g_vk.GetImageMemoryRequirements(g_dev, img->image, &req);
    int idx = memory_type(req.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size, .memoryTypeIndex = (uint32_t)idx};
    if (idx < 0 || g_vk.AllocateMemory(g_dev, &mai, NULL, &img->mem) != VK_SUCCESS) {
        g_vk.DestroyImage(g_dev, img->image, NULL); free(img); return NULL;
    }
    g_vk.BindImageMemory(g_dev, img->image, img->mem, 0);

    VkImageSubresource subr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
    VkSubresourceLayout lay;
    g_vk.GetImageSubresourceLayout(g_dev, img->image, &subr, &lay);
    img->offset = lay.offset;
    img->row_pitch = lay.rowPitch;
    if (g_vk.MapMemory(g_dev, img->mem, 0, req.size, 0, &img->map) != VK_SUCCESS) {
        vkp_image_destroy(img); return NULL;
    }
    return img;
}

/* Renders are synchronous (we wait for the frame's fence), so writing between frames
 * never races the GPU. */
void vkp_image_upload_shm(struct vkp_image *img, const void *data, int stride) {
    if (!img || !img->map || !data) return;
    size_t rowbytes = (size_t)img->w * 4;
    if ((size_t)stride < rowbytes) rowbytes = (size_t)stride;
    for (int y = 0; y < img->h; y++)
        memcpy((uint8_t *)img->map + img->offset + (size_t)y * img->row_pitch,
               (const uint8_t *)data + (size_t)y * stride, rowbytes);
}

int vkp_image_width(const struct vkp_image *img) { return img ? img->w : 0; }
int vkp_image_height(const struct vkp_image *img) { return img ? img->h : 0; }

void vkp_image_destroy(struct vkp_image *img) {
    if (!img) return;
    if (g_dev) {
        blendp_forget_image(img->image);
        if (img->map) g_vk.UnmapMemory(g_dev, img->mem);
        if (img->image) g_vk.DestroyImage(g_dev, img->image, NULL);
        if (img->mem) g_vk.FreeMemory(g_dev, img->mem, NULL);
    }
    free(img);
}

/* Map a draw into swapchain pixels through the scale mode, clipping the destination to the
 * picture's region (the whole output, or its half on TOP/BOTTOM; FILL's overflow is cut here)
 * and trimming the source to match. Returns 0 if nothing is left to draw. */
static int map_draw(const struct vkp_draw *d, float kx, float ky, float off_x, float off_y,
                    float L, float T, float R, float B, VkImageBlit *blit) {
    float x0 = off_x + d->dx * kx, y0 = off_y + d->dy * ky;
    float x1 = off_x + (d->dx + d->dw) * kx, y1 = off_y + (d->dy + d->dh) * ky;
    float sx0 = d->sx, sy0 = d->sy, sx1 = d->sx + d->sw, sy1 = d->sy + d->sh;

    if (x1 <= x0 || y1 <= y0 || sx1 <= sx0 || sy1 <= sy0) return 0;
    if (x0 < L) { sx0 += (L - x0) / (x1 - x0) * (sx1 - sx0); x0 = L; }
    if (y0 < T) { sy0 += (T - y0) / (y1 - y0) * (sy1 - sy0); y0 = T; }
    if (x1 > R) { sx1 -= (x1 - R) / (x1 - x0) * (sx1 - sx0); x1 = R; }
    if (y1 > B) { sy1 -= (y1 - B) / (y1 - y0) * (sy1 - sy0); y1 = B; }

    int ix0 = (int)(x0 + 0.5f), iy0 = (int)(y0 + 0.5f), ix1 = (int)(x1 + 0.5f), iy1 = (int)(y1 + 0.5f);
    int isx0 = (int)sx0, isy0 = (int)sy0, isx1 = (int)(sx1 + 0.5f), isy1 = (int)(sy1 + 0.5f);
    if (isx0 < 0) isx0 = 0;
    if (isy0 < 0) isy0 = 0;
    if (isx1 > d->img->w) isx1 = d->img->w;
    if (isy1 > d->img->h) isy1 = d->img->h;
    if (ix1 <= ix0 || iy1 <= iy0 || isx1 <= isx0 || isy1 <= isy0) return 0;

    *blit = (VkImageBlit){.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .srcOffsets = {{isx0, isy0, 0}, {isx1, isy1, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstOffsets = {{ix0, iy0, 0}, {ix1, iy1, 1}}};
    return 1;
}

static int draw_to_blit(const struct vkp_draw *d, VkImageBlit *blit) {
    float R = (float)(g_map.rx + g_map.rw), B = (float)(g_map.ry + g_map.rh);
    if (R > (float)g_extent.width) R = (float)g_extent.width;
    if (B > (float)g_extent.height) B = (float)g_extent.height;
    return map_draw(d, g_map.kx, g_map.ky, g_map.off_x, g_map.off_y, (float)g_map.rx, (float)g_map.ry, R, B, blit);
}

/* The same draw 1:1 into the scene image (no mapping: scene pixels are the destination). */
static int draw_to_scene_blit(const struct vkp_draw *d, int scene_w, int scene_h, VkImageBlit *blit) {
    return map_draw(d, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, (float)scene_w, (float)scene_h, blit);
}

/* One draw onto a composition target (in TRANSFER_DST_OPTIMAL): a translucent surface through the
 * alpha pass where that can read it, everything else - and any surface it cannot take - as a blit. */
static void record_draw(VkCommandBuffer cmd, const struct vkp_draw *d, const VkImageBlit *blit, VkImage target,
                        VkFormat target_fmt, int target_w, int target_h, VkFilter filter) {
    const struct vkp_image *im = d->img;
    if (d->blend && im->sampled &&
        blendp_draw(cmd, im->image, im->fmt, im->w, im->h, !im->dmabuf, target, target_fmt, target_w, target_h,
                    blit) == 0)
        return;
    g_vk.CmdBlitImage(cmd, im->image, im->dmabuf ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL,
                      target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, blit, filter);
}

static void destroy_scene_image(void) {
    if (g_scene.img) hdrc_forget_image(g_scene.img); /* the HDR pass may have drawn into it */
    if (g_scene.img) blendp_forget_image(g_scene.img);
    if (g_scene.img) g_vk.DestroyImage(g_dev, g_scene.img, NULL);
    if (g_scene.mem) g_vk.FreeMemory(g_dev, g_scene.mem, NULL);
    memset(&g_scene, 0, sizeof(g_scene));
}

/* The scene image at this size (R8G8B8A8: a blit converts every client format into it, and the
 * chain's shaders read it as RGBA). 0 = ready. */
static int ensure_scene_image(int w, int h) {
    if (g_scene.img && g_scene.w == w && g_scene.h == h) return 0;
    destroy_scene_image();
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {(uint32_t)w, (uint32_t)h, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, /* the HDR encode pass writes it (frame generation) */
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = g_vk.CreateImage(g_dev, &ici, NULL, &g_scene.img);
    if (r != VK_SUCCESS) { LOGE("effects: scene image %dx%d: vkCreateImage %s", w, h, vk_result_name(r)); g_scene.img = VK_NULL_HANDLE; return -1; }
    VkMemoryRequirements req;
    g_vk.GetImageMemoryRequirements(g_dev, g_scene.img, &req);
    int idx = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (idx < 0) idx = memory_type(req.memoryTypeBits, 0);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                .memoryTypeIndex = (uint32_t)(idx < 0 ? 0 : idx)};
    r = g_vk.AllocateMemory(g_dev, &mai, NULL, &g_scene.mem);
    if (r != VK_SUCCESS) { LOGE("effects: scene image %dx%d: vkAllocateMemory %s", w, h, vk_result_name(r)); destroy_scene_image(); return -1; }
    g_vk.BindImageMemory(g_dev, g_scene.img, g_scene.mem, 0);
    g_scene.w = w; g_scene.h = h;
    wnc_log("effects", "scene image %dx%d for the effect chain", w, h);
    return 0;
}

/* One of the HDR composition's device-local images at this size and format. 0 = ready. */
static int ensure_img(struct vkp_img_slot *s, int w, int h, VkFormat fmt, VkImageUsageFlags usage, const char *what) {
    if (s->img && s->w == w && s->h == h && s->fmt == fmt) return 0;
    if (s->img) hdrc_forget_image(s->img);
    if (s->img) blendp_forget_image(s->img);
    if (s->img) g_vk.DestroyImage(g_dev, s->img, NULL);
    if (s->mem) g_vk.FreeMemory(g_dev, s->mem, NULL);
    memset(s, 0, sizeof(*s));
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
        .extent = {(uint32_t)w, (uint32_t)h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = g_vk.CreateImage(g_dev, &ici, NULL, &s->img);
    if (r != VK_SUCCESS) { LOGE("color: %s image %dx%d: vkCreateImage %s", what, w, h, vk_result_name(r)); s->img = VK_NULL_HANDLE; return -1; }
    VkMemoryRequirements req;
    g_vk.GetImageMemoryRequirements(g_dev, s->img, &req);
    int idx = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (idx < 0) idx = memory_type(req.memoryTypeBits, 0);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                .memoryTypeIndex = (uint32_t)(idx < 0 ? 0 : idx)};
    r = g_vk.AllocateMemory(g_dev, &mai, NULL, &s->mem);
    if (r != VK_SUCCESS) {
        LOGE("color: %s image %dx%d: vkAllocateMemory %s", what, w, h, vk_result_name(r));
        g_vk.DestroyImage(g_dev, s->img, NULL);
        memset(s, 0, sizeof(*s));
        return -1;
    }
    g_vk.BindImageMemory(g_dev, s->img, s->mem, 0);
    s->w = w; s->h = h; s->fmt = fmt;
    char fname[24];
    vk_format_short(fmt, fname, sizeof(fname));
    wnc_log("color", "%s image %dx%d (%s)", what, w, h, fname);
    return 0;
}

/* The HDR composition (hdr_compose.h): clear the mixed image, blit every draw into it 1:1 in its own
 * encoding, then encode it into `out` (left in TRANSFER_DST_OPTIMAL). The caller has already taken the
 * draws' images (queue-family acquire / host-write barriers). Which rectangles are HDR comes from
 * hf->is_hdr: listed top draw first, down to the lowest HDR draw (everything under that is SDR).
 * 0 = done, -1 = the HDR pass is unavailable (nothing was recorded into `out`). */
static int compose_hdr(VkCommandBuffer cmd, const struct vkp_draw *draws, int n, const struct vkp_hdr_frame *hf,
                       int scene_w, int scene_h, VkImage out, VkFormat out_fmt, enum hdrc_mode mode) {
    if (!hf || !hf->is_hdr || n <= 0) return -1;
    if (ensure_img(&g_mixed, scene_w, scene_h, HDR_FMT,
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                   "HDR composition (mixed)") != 0)
        return -1;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                              .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                              .image = g_mixed.img, .subresourceRange = range,
                              .srcAccessMask = VK_ACCESS_SHADER_READ_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, NULL, 0, NULL, 1, &b);
    VkClearColorValue black = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
    g_vk.CmdClearColorImage(cmd, g_mixed.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);

    struct hdrc_params p = {.mode = mode, .sdr_white_nits = wnc_color_sdr_white(),
                            .peak_nits = hf->peak_nits > 0.0f ? hf->peak_nits : 1000.0f};
    int lowest_hdr = -1;
    for (int i = 0; i < n; i++) {
        VkImageBlit blit;
        if (!draws[i].img || !draw_to_scene_blit(&draws[i], scene_w, scene_h, &blit)) continue;
        /* A blit converts UNORM to UNORM by value: 8-bit sRGB and 10-bit PQ both land unchanged. */
        record_draw(cmd, &draws[i], &blit, g_mixed.img, HDR_FMT, scene_w, scene_h, VK_FILTER_LINEAR);
        if (hf->is_hdr[i] && lowest_hdr < 0) lowest_hdr = i;
    }
    static int truncated_said;
    if (lowest_hdr >= 0) {
        int i;
        for (i = n - 1; i >= lowest_hdr && p.count < HDRC_MAX_RECTS; i--) {
            VkImageBlit blit;
            if (!draws[i].img || !draw_to_scene_blit(&draws[i], scene_w, scene_h, &blit)) continue;
            p.rects[p.count][0] = (float)blit.dstOffsets[0].x; p.rects[p.count][1] = (float)blit.dstOffsets[0].y;
            p.rects[p.count][2] = (float)blit.dstOffsets[1].x; p.rects[p.count][3] = (float)blit.dstOffsets[1].y;
            if (hf->is_hdr[i]) p.hdr_mask |= 1u << p.count;
            p.count++;
        }
        if (i >= lowest_hdr && !truncated_said) {
            truncated_said = 1;
            wnc_log("color", "HDR composition: more than %d windows are stacked over the HDR game - the lowest "
                       "ones are treated as SDR", HDRC_MAX_RECTS);
        }
    }
    return hdrc_encode(cmd, g_mixed.img, out, out_fmt, scene_w, scene_h, &p);
}

/* VK_ERROR_DEVICE_LOST: nothing on this device works any more, and there is no way back short
 * of a new session. Say so once and stop touching the swapchain; clients keep being paced by the
 * compositor (see pace_without_output) so they don't wedge, they just aren't shown. */
static void device_lost(const char *where) {
    if (g_dev_state == -2) return;
    g_dev_state = -2;
    g_base_black = 0;
    vkp_framegen_device_lost();
    wnc_log("error", "GPU device lost (VK_ERROR_DEVICE_LOST in %s): the compositor has stopped presenting; "
               "restart the session", where);
    for (uint32_t i = 0; i < g_nimg; i++) blendp_forget_image(g_images[i]);
    if (g_swapchain) g_vk.DestroySwapchainKHR(g_dev, g_swapchain, NULL);
    g_swapchain = VK_NULL_HANDLE;
    if (g_surface) g_vk.DestroySurfaceKHR(g_inst, g_surface, NULL);
    g_surface = VK_NULL_HANDLE;
    free(g_images);
    g_images = NULL;
    g_nimg = 0;
}

static const char *vk_result_name(VkResult r) {
    switch (r) {
    case VK_ERROR_OUT_OF_DATE_KHR: return "OUT_OF_DATE";
    case VK_SUBOPTIMAL_KHR: return "SUBOPTIMAL";
    case VK_ERROR_SURFACE_LOST_KHR: return "SURFACE_LOST";
    case VK_ERROR_DEVICE_LOST: return "DEVICE_LOST";
    case VK_TIMEOUT: return "TIMEOUT";
    case VK_NOT_READY: return "NOT_READY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "OUT_OF_HOST_MEMORY";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "FORMAT_NOT_SUPPORTED";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: return "INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT";
    default: {
        static char other[24];
        snprintf(other, sizeof(other), "error %d", (int)r);
        return other;
    }
    }
}

/* ---------------------------------------------------------------- the compositor pass */

/* The mapping/blit stage for one present: clear the swapchain image to black (the letterbox),
 * blit `src` (src_w x src_h, in GENERAL: the effects chain's result or a generated frame, either
 * standing for the whole scene) through the scene -> output mapping, and leave the image ready
 * to present. */
static void record_screen_blit(VkCommandBuffer cmd, VkImage src, int src_w, int src_h,
                               int scene_w, int scene_h, uint32_t img, VkFilter filter) {
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier pre[2] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = g_images[img], .subresourceRange = range, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT},
        /* Whatever wrote src last (blits, the effects chain, a compute dispatch, in this or an
         * earlier submission on this queue) must be visible to the blit's read. */
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = src, .subresourceRange = range,
         .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT}};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, NULL, 0, NULL, 2, pre);
    struct vkp_image tmp = {.w = src_w, .h = src_h}; /* only its size is looked at */
    struct vkp_draw d = {&tmp, 0, 0, (float)src_w, (float)src_h, 0, 0, scene_w, scene_h};
    VkImageBlit blit;
    const int have_blit = draw_to_blit(&d, &blit);
    /* The result fills its whole mapped rectangle, so only the letterbox bars around it need black. */
    if (!have_blit || !record_bars_clear(cmd, g_images[img], &blit)) {
        VkClearColorValue black = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
        g_vk.CmdClearColorImage(cmd, g_images[img], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    }
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 1, &mb, 0, NULL, 0, NULL);
    if (have_blit)
        g_vk.CmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, g_images[img],
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, filter);
    VkImageMemoryBarrier b_present = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_images[img],
        .subresourceRange = range, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            0, 0, NULL, 0, NULL, 1, &b_present);
}

/* Acquire a swapchain image for present slot k. On the frame's FIRST acquire the swapchain is
 * (re)created when needed and rebuilt once when the surface changed under us (OUT_OF_DATE after a
 * resize/rotation, SURFACE_LOST when the window was torn down) - once, so a frame is not lost to
 * a recreate that would have succeeded. Later slots have presents in flight on this swapchain,
 * so they never rebuild; they fail and the frame ends early. 0 on success. */
static int acquire_image(int k, int first, uint32_t *img, VkResult *ar_out) {
    for (int attempt = 0; ; attempt++) {
        if (!g_swapchain && (!first || swap_init() != 0)) return -1;
        /* A bounded wait: a surface that went away without telling us must not park the
         * compositor thread forever (clients are paced from this thread). */
        const int64_t t_acq = perf_now();
        VkResult ar = g_vk.AcquireNextImageKHR(g_dev, g_swapchain, 1000000000ULL, g_acqs[k], VK_NULL_HANDLE, img);
        perf_add(&g_perf.acquire_ns, &g_perf.acquire_max_ns, &g_perf.acquires, perf_now() - t_acq);
        if (ar == VK_SUCCESS || ar == VK_SUBOPTIMAL_KHR) { *ar_out = ar; return 0; }
        if (ar == VK_ERROR_DEVICE_LOST) { device_lost("acquire"); return -1; }
        if (ar == VK_ERROR_OUT_OF_DATE_KHR || ar == VK_ERROR_SURFACE_LOST_KHR) {
            wnc_log("gpu", "screen surface %s on acquire: rebuilding the swapchain", vk_result_name(ar));
            destroy_swapchain();
            if (first && attempt == 0) continue;
            return -1;
        }
        wnc_log("error", "present: acquire %d failed (%s %d)", k, vk_result_name(ar), (int)ar);
        if (ar == VK_TIMEOUT || ar == VK_NOT_READY) return -1;
        destroy_swapchain(); /* anything else: start over next frame */
        return -1;
    }
}

/* vkp_render_plain: the base surface's black frame with the compositor pass skipped (HDR game on its
 * layer, compositor.c) - neither the effects chain nor frame generation runs, and so neither paces
 * this loop with extra presents. */
static int g_plain_frame;
int vkp_render_plain(int scene_w, int scene_h) {
    g_plain_frame = 1;
    int r = vkp_render(scene_w, scene_h, NULL, 0);
    g_plain_frame = 0;
    return r;
}

/* A surface that changed size or transform under a swapchain nobody presents to (the base surface
 * while the game is on its display layer) never reports OUT_OF_DATE, so ask it - at most every
 * 100 ms, on the compositor thread - and tear the swapchain down when it no longer matches; the next
 * frame rebuilds it (and, in layer mode, puts one black frame on it). */
static void check_surface_changed(void) {
    if (!g_swapchain || !g_surface || g_dev_state != 1) return;
    const int64_t now = perf_now();
    if (now - g_surface_checked_ns < 100000000LL) return;
    g_surface_checked_ns = now;
    VkSurfaceCapabilitiesKHR caps;
    if (g_vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, g_surface, &caps) != VK_SUCCESS) return;
    if (caps.currentExtent.width == g_caps_extent.width && caps.currentExtent.height == g_caps_extent.height &&
        caps.currentTransform == g_surface_transform)
        return;
    wnc_log("gpu", "screen surface changed (%ux%u -> %ux%u, transform %d -> %d): rebuilding the swapchain",
               g_caps_extent.width, g_caps_extent.height, caps.currentExtent.width, caps.currentExtent.height,
               (int)g_surface_transform, (int)caps.currentTransform);
    destroy_swapchain();
}

int vkp_base_black(int scene_w, int scene_h) {
    vkp_apply_window_request();
    check_surface_changed();
    if (g_base_black && g_swapchain && g_window && g_dev_state == 1) {
        g_perf.base_kept++;
        return 0; /* the black frame from before is still what the base surface shows */
    }
    int r = vkp_render_plain(scene_w, scene_h);
    if (r == 0) g_perf.base_presents++;
    return r;
}

void vkp_perf_take(struct vkp_perf *out) {
    *out = g_perf;
    memset(&g_perf, 0, sizeof(g_perf));
}

int vkp_can_wait_sync_fd(void) { return g_import_sem_fd != NULL && g_wait_sem != VK_NULL_HANDLE; }

/* Hand a sync_file to the next submit as a GPU wait (temporary import into g_wait_sem; the driver
 * owns the fd from then on). Returns 1 when the submit must wait on g_wait_sem, 0 when there is nothing
 * to wait on - the fd was -1, or it could not be imported and was waited for on the CPU (bounded) and
 * closed instead. -1 = the display is still reading the buffer after 100 ms: do not write it. */
static int take_wait_fd(int fd) {
    if (fd < 0) return 0;
    if (vkp_can_wait_sync_fd()) {
        VkImportSemaphoreFdInfoKHR ii = {.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                                         .semaphore = g_wait_sem, .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                                         .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, .fd = fd};
        if (g_import_sem_fd(g_dev, &ii) == VK_SUCCESS) {
            g_perf.gpu_release_waits++;
            return 1;
        }
        static int said;
        if (!said) {
            said = 1;
            wnc_log("perf", "layer buffers: importing a release fence as a GPU wait failed; waiting for it on the CPU");
        }
    }
    struct pollfd p = {.fd = fd, .events = POLLIN};
    int r;
    do { r = poll(&p, 1, 100); } while (r < 0 && errno == EINTR);
    close(fd);
    return r == 0 ? -1 : 0;
}

/* The HDR10 swapchain's metadata: the game's SMPTE 2086 / CTA-861.3 values (its image description) via
 * VK_EXT_hdr_metadata, once per swapchain and description. A game that gave none gets none. */
static void send_hdr_metadata(const struct wnc_color *c) {
    static int said_none, said_nodesc;
    if (!c) return;
    if (!g_set_hdr_metadata) {
        if (!said_none) { said_none = 1; wnc_log("color", "HDR10 swapchain without the game's metadata: this device "
                                                    "has no VK_EXT_hdr_metadata (the display uses its defaults)"); }
        return;
    }
    if (!c->has_st2086 && !c->has_cta861) {
        if (said_nodesc != (int)c->identity) {
            said_nodesc = (int)c->identity;
            wnc_log("color", "HDR10 swapchain: image description #%u carries no HDR metadata (the game sent none), "
                       "so none is set - the display uses its defaults", c->identity);
        }
        return;
    }
    VkHdrMetadataEXT md = {.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT};
    if (c->has_st2086) {
        md.displayPrimaryRed = (VkXYColorEXT){c->red[0], c->red[1]};
        md.displayPrimaryGreen = (VkXYColorEXT){c->green[0], c->green[1]};
        md.displayPrimaryBlue = (VkXYColorEXT){c->blue[0], c->blue[1]};
        md.whitePoint = (VkXYColorEXT){c->white[0], c->white[1]};
        md.maxLuminance = c->max_lum;
        md.minLuminance = c->min_lum;
    }
    if (c->has_cta861) {
        md.maxContentLightLevel = c->max_cll;
        md.maxFrameAverageLightLevel = c->max_fall;
    }
    g_set_hdr_metadata(g_dev, 1, &g_swapchain, &md);
    wnc_log("color", "HDR10 swapchain: the game's metadata set via VK_EXT_hdr_metadata (image description #%u: %s)",
               c->identity, c->text);
}

/* The whole present: vkp_render (hf == NULL, exactly as it always was) and vkp_render_hdr (a scene with
 * HDR draws: composed into one encoding, hdr_compose.h). */
static int render_impl(int scene_w, int scene_h, const struct vkp_draw *draws, int n,
                       const struct vkp_hdr_frame *hf, int *how) {
    if (how) *how = 0;
    vkp_apply_window_request();
    if (g_dev_state == -2) return -1;
    if (dev_init() != 0 || !g_window || scene_w <= 0 || scene_h <= 0) return -1;

    /* Frame generation queues several presents per scene frame and needs swapchain images for
     * them: rebuild when that changes (armed, disarmed, another multiplier). */
    if (g_swapchain && vkp_framegen_extra_images() != g_swap_extra) {
        wnc_log("gpu", "frame generation now wants %d present%s per frame: rebuilding the swapchain",
                   1 + vkp_framegen_extra_images(), vkp_framegen_extra_images() ? "s" : "");
        destroy_swapchain();
    }
    /* HDR frames presented here (frame generation with an HDR game) want an HDR10 swapchain - unless the
     * drawer's HDR output switch is off (tone-mapped) - nothing else does. A change of that rebuilds the
     * swapchain (only in HDR sessions: g_colorspace_ext). */
    const int want_hdr = hf && !hf->tonemap && g_colorspace_ext && !g_swap_hdr_unavailable;
    if (g_swapchain && want_hdr != g_swap_is_hdr) {
        wnc_log("color", want_hdr ? "HDR frames with frame generation: rebuilding the screen swapchain as HDR10"
                            : hf     ? "HDR output switched off: rebuilding the screen swapchain as SDR (frames tone-mapped)"
                                     : "no HDR frames through the screen swapchain any more: rebuilding it as SDR");
        destroy_swapchain();
    }
    g_swap_hdr_want = want_hdr;

    uint32_t img = 0;
    VkResult ar;
    if (acquire_image(0, 1, &img, &ar) != 0) return -1;
    update_map(scene_w, scene_h);
    /* A new HDR10 swapchain, or a new image description on it: hand the game's metadata over. */
    if (g_swap_is_hdr && hf && hf->color && hf->color->identity != g_swap_md_identity) {
        g_swap_md_identity = hf->color->identity;
        send_hdr_metadata(hf->color);
    }

    /* The compositor pass - scene -> effects -> frame generation -> mapping/blit -> swapchain -
     * composes the scene off-screen when either stage needs the whole frame: the screen-effect
     * chain (effects_chain.c) or frame generation (framegen_bridge.c). With both off the draws are
     * blitted straight through the mapping into the swapchain image, as they always were; only the
     * blit filter follows the scaling mode. */
    const int fx = !g_plain_frame && vkp_effects_active();
    const int fg = !g_plain_frame && vkp_framegen_active();
    /* HDR frames with frame generation through an HDR10 swapchain: the scene, the effects and the
     * engine's frames in FP16 (FG_HDR_FMT) where the engine can take that format - more than 8 bits of
     * PQ through interpolation - else the 8-bit scene, as before. */
    const int deep = hf && n > 0 && fg && g_swap_is_hdr && !hf->tonemap && vkp_framegen_format_ok(FG_HDR_FMT) &&
                     ensure_img(&g_fgscene, scene_w, scene_h, FG_HDR_FMT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                "HDR frame generation (scene)") == 0;
    {
        static int said = -1;
        if (fg && hf && g_swap_is_hdr && deep != said) {
            said = deep;
            char sw[24];
            vk_format_short(g_swap_fmt, sw, sizeof(sw));
            if (deep)
                wnc_log("color", "frame generation runs on the HDR picture in FP16 (RGBA16F): more than 8 bits of PQ "
                           "through interpolation, into the HDR10 swapchain (%s)", sw);
            else
                wnc_log("color", "frame generation runs on the HDR picture in 8 bits (the engine cannot take FP16 here), "
                           "into the HDR10 swapchain (%s): banding possible in smooth gradients", sw);
        }
    }
    /* An HDR scene always takes the pass: its draws are composed into one encoding first. */
    const int pass = deep || ((fx || fg || (hf && n > 0)) && ensure_scene_image(scene_w, scene_h) == 0);
    VkImage scene_img = deep ? g_fgscene.img : g_scene.img;
    const VkFormat scene_fmt = deep ? FG_HDR_FMT : SCENE_FMT;
    const VkFilter blit_filter = vkp_effects_blit_filter();

    VkCommandBuffer cmd = g_cmds[0];
    g_vk.ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    g_vk.BeginCommandBuffer(cmd, &bi);
    blendp_begin_frame();
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    /* Source images: take dmabufs from the client's queue family, move shm images to
     * GENERAL once (host writes stay visible across frames: memory is coherent and each
     * frame is a new submission). One barrier per distinct image. The destination joins them:
     * the swapchain image on the direct path, the scene image on the compositor pass (the
     * per-present blits transition the swapchain images themselves there). */
    VkImageMemoryBarrier *bars = calloc((size_t)n + 1, sizeof(*bars));
    int nb = 0;
    bars[nb++] = (VkImageMemoryBarrier){
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = pass ? scene_img : g_images[img], .subresourceRange = range,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    for (int i = 0; i < n; i++) {
        struct vkp_image *im = draws[i].img;
        int seen = 0;
        for (int j = 0; j < i; j++) if (draws[j].img == im) { seen = 1; break; }
        if (seen || !im) continue;
        if (im->dmabuf) {
            bars[nb++] = (VkImageMemoryBarrier){
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .dstQueueFamilyIndex = g_qfam,
                .image = im->image, .subresourceRange = range, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
        } else if (!im->in_general) {
            bars[nb++] = (VkImageMemoryBarrier){
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = im->image, .subresourceRange = range,
                .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
            im->in_general = 1;
        }
    }
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
                                 (pass ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : 0),
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, (uint32_t)nb, bars);
    free(bars);

    /* Clear the composition target to black (the letterbox on the direct path, the desktop's
     * background on the pass) before the draws land on it. A draw that covers the whole scene
     * overwrites every pixel of its rectangle (blits never blend), so then only what lies outside it
     * needs black: the letterbox bars on the direct path, nothing at all on the pass (the scene image
     * is exactly the scene). Anything else - no such draw, the HDR composition - clears the whole
     * target as before. */
    VkImage target = pass ? scene_img : g_images[img];
    int covered = 0;
    VkImageBlit cover_blit;
    if (!(hf && pass)) {
        for (int i = n - 1; i >= 0 && !covered; i--) {
            const struct vkp_draw *d = &draws[i];
            if (!d->img || d->blend || d->dx > 0 || d->dy > 0 || d->dx + d->dw < scene_w || d->dy + d->dh < scene_h) continue;
            covered = pass ? draw_to_scene_blit(d, scene_w, scene_h, &cover_blit) : draw_to_blit(d, &cover_blit);
        }
    }
    const int cleared = covered && (pass || record_bars_clear(cmd, target, &cover_blit));
    if (!cleared) {
        VkClearColorValue black = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
        g_vk.CmdClearColorImage(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    }
    {
        VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
        g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0, 1, &mb, 0, NULL, 0, NULL);
    }

    /* 1. Scene: the draws in order, 1:1 into the scene image (pass) or mapped onto the output. An HDR
     *    scene is composed into ONE encoding instead (hdr_compose.h): PQ BT.2020 for an HDR10
     *    swapchain, tone-mapped sRGB for an ordinary one; the scene image comes back in
     *    TRANSFER_DST_OPTIMAL like the blits leave it. If that pass is unavailable, the blits below. */
    int drawn = 0;
    int hdr_how = 0;
    if (hf && pass && n > 0 &&
        compose_hdr(cmd, draws, n, hf, scene_w, scene_h, scene_img, scene_fmt,
                    g_swap_is_hdr ? HDRC_OUT_PQ : HDRC_OUT_SDR) == 0) {
        hdr_how = g_swap_is_hdr ? 1 : 2;
        drawn = n;
    }
    for (int i = 0; i < n && !hdr_how; i++) {
        VkImageBlit blit;
        if (!draws[i].img) continue;
        if (pass ? !draw_to_scene_blit(&draws[i], scene_w, scene_h, &blit) : !draw_to_blit(&draws[i], &blit)) continue;
        record_draw(cmd, &draws[i], &blit, target, pass ? scene_fmt : g_swap_fmt,
                    pass ? scene_w : (int)g_extent.width, pass ? scene_h : (int)g_extent.height,
                    pass ? VK_FILTER_LINEAR : blit_filter);
        drawn++;
    }

    int ngen = 0;
    VkImage gens[VKP_FG_MAX_GENERATIONS] = {VK_NULL_HANDLE};
    VkImage result = scene_img;
    int rw = scene_w, rh = scene_h;
    if (!pass) {
        VkImageMemoryBarrier b_present = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_images[img],
            .subresourceRange = range, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
        g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                0, 0, NULL, 0, NULL, 1, &b_present);
    } else {
        /* 2. Effects (the hook): scaling to the scene's mapped size, then the effects, in the X11
         *    order. The result stands for the whole scene (rw x rh) and comes back in
         *    TRANSFER_SRC_OPTIMAL; with nothing on the scene stays as composited. */
        VkImageLayout result_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        if (fx) {
            int mapped_w = (int)(scene_w * g_map.kx + 0.5f), mapped_h = (int)(scene_h * g_map.ky + 0.5f);
            /* This path's scene is 8-bit. Set only where the chain really runs: a plain frame (the black
             * base under an HDR picture) must not flip the chain's format back every frame. */
            vkp_effects_set_formats(scene_fmt, scene_fmt);
            result = vkp_effects_run(cmd, scene_img, scene_w, scene_h, mapped_w, mapped_h, &rw, &rh);
            result_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        }
        /* Hand-off: from here the frame lives in GENERAL - the frame-generation engines copy and
         * sample it there, and every screen blit reads it there. Whatever wrote it last (the
         * composite blits or the chain's last pass) is made visible to both. */
        VkImageMemoryBarrier b_general = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = result_layout,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = result, .subresourceRange = range,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT};
        g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 0, NULL, 0, NULL, 1, &b_general);

        /* 3. Frame generation (the hook): the result is frame N; the engine interpolates between
         *    N-1 and N and hands back the frames that belong in between. Shown first, N last. */
        if (fg) {
            ngen = vkp_framegen_run(cmd, result, VK_NULL_HANDLE, rw, rh, scene_fmt, gens);
            if (ngen < 0) ngen = 0;
            if (ngen > VKP_FG_MAX_GENERATIONS) ngen = VKP_FG_MAX_GENERATIONS;
        }

        /* 4. Mapping/blit: the first present carries generated frame 0, or the result itself. */
        record_screen_blit(cmd, ngen ? gens[0] : result, rw, rh, scene_w, scene_h, img, blit_filter);
    }
    g_vk.EndCommandBuffer(cmd);

    /* One submit + present per slot: generated frames first, the real frame last. The frame
     * fence rides on the last submit (a fence signal is ordered after everything submitted
     * earlier on the queue, so it still covers all of this frame's work). With FIFO the
     * presentation engine shows them on consecutive vblanks: that IS the pacing, no sleeps. */
    const int presents = 1 + ngen;
    int fence_submitted = 0, presented_gen = 0;
    VkResult pr = VK_SUCCESS;
    g_vk.ResetFences(g_dev, 1, &g_fence);
    for (int k = 0; k < presents; k++) {
        if (k > 0) {
            VkResult ark;
            if (acquire_image(k, 0, &img, &ark) != 0) {
                wnc_log("framegen", "present %d/%d: no swapchain image; the frame ends early", k + 1, presents);
                pr = VK_ERROR_OUT_OF_DATE_KHR;
                break;
            }
            if (ark == VK_SUBOPTIMAL_KHR) ar = ark;
            VkCommandBuffer ck = g_cmds[k];
            g_vk.ResetCommandBuffer(ck, 0);
            g_vk.BeginCommandBuffer(ck, &bi);
            record_screen_blit(ck, k < ngen ? gens[k] : result, rw, rh, scene_w, scene_h, img, blit_filter);
            g_vk.EndCommandBuffer(ck);
        }
        const int last = (k + 1 == presents);
        VkPipelineStageFlags wait = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1,
                           .pWaitSemaphores = &g_acqs[k], .pWaitDstStageMask = &wait, .commandBufferCount = 1,
                           .pCommandBuffers = &g_cmds[k], .signalSemaphoreCount = 1, .pSignalSemaphores = &g_rnds[k]};
        VkResult qr = g_vk.QueueSubmit(g_queue, 1, &si, last ? g_fence : VK_NULL_HANDLE);
        if (qr != VK_SUCCESS) {
            if (qr == VK_ERROR_DEVICE_LOST) device_lost("submit");
            else LOGE("present: submit %d/%d failed (%d)", k + 1, presents, (int)qr);
            /* The acquired image is never presented; the swapchain would be stuck with it. */
            if (g_dev_state != -2) destroy_swapchain();
            return -1;
        }
        fence_submitted = last;
        VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
                               .pWaitSemaphores = &g_rnds[k], .swapchainCount = 1,
                               .pSwapchains = &g_swapchain, .pImageIndices = &img};
        const int64_t t_pres = perf_now();
        pr = g_vk.QueuePresentKHR(g_queue, &pi);
        perf_add(&g_perf.present_ns, &g_perf.present_max_ns, &g_perf.presents, perf_now() - t_pres);
        if (k < ngen) presented_gen++;
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_ERROR_SURFACE_LOST_KHR || pr == VK_ERROR_DEVICE_LOST) break;
    }
    if (!fence_submitted) {
        /* Ended early: an empty submit carrying the fence is ordered after everything queued
         * above, so the wait below still means "this frame's work is done". */
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
        g_vk.QueueSubmit(g_queue, 1, &si, g_fence);
    }
    VkResult fr = timed_wait(g_fence, UINT64_MAX);
    /* Counted on every screen frame, not just generating ones: the HUD's output-frame source
     * reads these totals even when the engine is off (a panel-side interpolator is still on). */
    vkp_framegen_presented(presented_gen);
    if (fr == VK_ERROR_DEVICE_LOST || pr == VK_ERROR_DEVICE_LOST) { device_lost("present"); return -1; }
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_ERROR_SURFACE_LOST_KHR) {
        /* The surface changed or went away under this frame: rebuild before the next one. */
        wnc_log("gpu", "screen surface %s on present: rebuilding the swapchain", vk_result_name(pr));
        destroy_swapchain();
        return -1;
    } else if (pr == VK_SUBOPTIMAL_KHR || ar == VK_SUBOPTIMAL_KHR) {
        /* Expected and harmless here: the swapchain uses IDENTITY preTransform on purpose (see
         * swap_init), so a rotated panel reports SUBOPTIMAL on every frame. Rebuilding would change
         * nothing (and a rebuild per frame would be far worse), so present as is; say so once. */
        static int said;
        if (!said) { said = 1; wnc_log("gpu", "surface reports SUBOPTIMAL (panel rotation); presenting as is"); }
    } else if (pr != VK_SUCCESS) {
        wnc_log("error", "present: vkQueuePresentKHR failed (%s %d)", vk_result_name(pr), (int)pr);
        destroy_swapchain();
        return -1;
    }

    /* What the base surface shows from now on: black only after a plain frame with nothing drawn
     * (vkp_base_black keeps presenting nothing while that holds); anything else it presented is
     * content, and the next layer frame must put a black frame back under the layers first. */
    g_base_black = (g_plain_frame && n == 0 && !hf) ? 1 : 0;

    /* Signal the app once, on the first real client frame reaching the screen, so the
     * launch/preloader overlay can dismiss (wayland has no XServer window-content hook). */
    if (drawn && !g_first_frame_done) {
        g_first_frame_done = 1;
        wnc_on_first_frame();
    }
    if (how) *how = hdr_how;
    return 0;
}

int vkp_render(int scene_w, int scene_h, const struct vkp_draw *draws, int n) {
    return render_impl(scene_w, scene_h, draws, n, NULL, NULL);
}

int vkp_render_hdr(int scene_w, int scene_h, const struct vkp_draw *draws, int n,
                   const struct vkp_hdr_frame *hf, int *how) {
    return render_impl(scene_w, scene_h, draws, n, hf, how);
}

/* ---------------------------------------------------------------- layer mode helpers */

/* Degrees the display rotates our layers by: 0, 90, 180 or 270; -1 before the swapchain exists.
 * Read from the surface, so it follows the panel's install orientation and the session's own
 * orientation together, on any device. */
int vkp_surface_rotation_degrees(void) {
    if (!g_surface_transform_known) return -1;
    switch (g_surface_transform) {
        case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:  return 90;
        case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR: return 180;
        case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR: return 270;
        default: return 0; /* IDENTITY, and the mirrored forms we never see here */
    }
}

ANativeWindow *vkp_window(void) { return g_window; }

void vkp_signal_first_frame(void) {
    if (g_first_frame_done) return;
    g_first_frame_done = 1;
    wnc_on_first_frame();
}

int vkp_update_map(int scene_w, int scene_h) {
    if (g_dev_state == -2 || dev_init() != 0 || !g_window || scene_w <= 0 || scene_h <= 0) return -1;
    check_surface_changed(); /* layer mode may present nothing on the base surface for a long time */
    if (!g_swapchain && swap_init() != 0) return -1;
    update_map(scene_w, scene_h);
    return g_map.valid ? 0 : -1;
}

int vkp_map_rect(int img_w, int img_h, int scene_w, int scene_h, int out[8]) {
    struct vkp_image tmp = {.w = img_w, .h = img_h}; /* only its size is looked at */
    struct vkp_draw d = {&tmp, 0, 0, (float)img_w, (float)img_h, 0, 0, scene_w, scene_h};
    return vkp_map_draw(&d, out);
}

int vkp_map_draw(const struct vkp_draw *d, int out[8]) {
    VkImageBlit blit;
    if (!d || !d->img || !g_map.valid || !draw_to_blit(d, &blit)) return 0;
    out[0] = blit.srcOffsets[0].x; out[1] = blit.srcOffsets[0].y;
    out[2] = blit.srcOffsets[1].x; out[3] = blit.srcOffsets[1].y;
    out[4] = blit.dstOffsets[0].x; out[5] = blit.dstOffsets[0].y;
    out[6] = blit.dstOffsets[1].x; out[7] = blit.dstOffsets[1].y;
    return 1;
}

/* Copy src (a client frame) into dst (a layer pool buffer) 1:1 and wait for it. Both images are
 * owned by the "foreign" queue family (the game's driver / the display) between our uses, so each
 * use acquires them and the destination is released back for the display to read. */
int vkp_blit_image(struct vkp_image *src, struct vkp_image *dst, int wait_fd) {
    if (!src || !dst || !dst->blit_dst || g_dev_state == -2 || dev_init() != 0) {
        if (wait_fd >= 0) close(wait_fd);
        return -1;
    }
    const int gpu_wait = take_wait_fd(wait_fd); /* the display's release of dst, before the copy writes it */
    if (gpu_wait < 0) return -1;
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkCommandBuffer g_cmd = g_cmds[0];
    g_vk.ResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    g_vk.BeginCommandBuffer(g_cmd, &bi);
    VkImageMemoryBarrier acq[2] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = src->dmabuf ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = src->dmabuf ? g_qfam : VK_QUEUE_FAMILY_IGNORED,
         .image = src->image, .subresourceRange = range, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .dstQueueFamilyIndex = g_qfam,
         .image = dst->image, .subresourceRange = range, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT}};
    if (!src->dmabuf) { /* shm image: host-written, GENERAL */
        acq[0].oldLayout = src->in_general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_PREINITIALIZED;
        acq[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        acq[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        src->in_general = 1;
    }
    g_vk.CmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 2, acq);
    int bw = src->w < dst->w ? src->w : dst->w, bh = src->h < dst->h ? src->h : dst->h;
    VkImageBlit blit = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .srcOffsets = {{0, 0, 0}, {bw, bh, 1}},
                        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .dstOffsets = {{0, 0, 0}, {bw, bh, 1}}};
    g_vk.CmdBlitImage(g_cmd, src->image,
                      src->dmabuf ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL,
                      dst->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    VkImageMemoryBarrier rel = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = g_qfam,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .image = dst->image, .subresourceRange = range,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.CmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            0, 0, NULL, 0, NULL, 1, &rel);
    g_vk.EndCommandBuffer(g_cmd);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_cmd,
                       .waitSemaphoreCount = gpu_wait ? 1u : 0u, .pWaitSemaphores = gpu_wait ? &g_wait_sem : NULL,
                       .pWaitDstStageMask = gpu_wait ? &wait_stage : NULL};
    g_vk.ResetFences(g_dev, 1, &g_fence);
    VkResult qr = g_vk.QueueSubmit(g_queue, 1, &si, g_fence);
    if (qr != VK_SUCCESS) {
        if (qr == VK_ERROR_DEVICE_LOST) device_lost("layer blit");
        else LOGE("layer: blit submit failed (%d)", (int)qr);
        return -1;
    }
    VkResult fr = timed_wait(g_fence, 1000000000ULL);
    if (fr == VK_ERROR_DEVICE_LOST) { device_lost("layer blit"); return -1; }
    if (fr != VK_SUCCESS) { LOGE("layer: blit fence wait -> %s", vk_result_name(fr)); return -1; }
    return 0;
}

/* ---------------------------------------------------------------- the compositor pass, off-screen
 *
 * Layer mode with screen effects on: the game's frame still has to go through the compositor pass
 * (scene image -> effects chain), but the RESULT does not have to end up in the app's swapchain -
 * it is copied into a gralloc buffer and put on the game's own Android layer, so hardware
 * composition stays alive while a Look is applied and the scene -> output mapping is done by the
 * display (setGeometry) instead of a second full-screen GPU blit.
 *
 * The chain runs in its OWN submit, with no swapchain image acquired. That is deliberate and was
 * measured: folding it into the submit that owns the acquired image (one command buffer, one
 * present, which looks cheaper on paper) puts the whole 13-pass chain behind the acquire
 * semaphore, i.e. behind a vblank - 72 fps instead of 111 fps on the Pocket FIT with Retro CRT at
 * 1920x1080. The base surface's black frame is presented by the caller AFTER the layer
 * transaction, so the layer never waits for a vblank either.
 *
 * Three steps because the chain's result size is only known once the chain has run, and the layer
 * buffer is allocated from it:
 *     vkp_pass_begin()  -> records composite + effects into the frame's command buffer
 *     vkp_pass_copy_to()-> appends the copy into the layer buffer, submits, waits
 *     vkp_pass_abort()  -> drops it unsubmitted (no free layer buffer this frame)
 * Frame generation is NOT run here: its extra frames need a present each, on consecutive vblanks,
 * which one buffer per layer transaction cannot pace (see WAYLAND_RUNTIME.md). */

static struct {
    int active;
    VkImage result;   /* in TRANSFER_SRC_OPTIMAL */
    int rw, rh;
} g_pass;

static int pass_begin_impl(int scene_w, int scene_h, const struct vkp_draw *draws, int n,
                           const struct vkp_hdr_frame *hf, int *rw, int *rh);

int vkp_pass_begin(int scene_w, int scene_h, const struct vkp_draw *draws, int n, int *rw, int *rh) {
    return pass_begin_impl(scene_w, scene_h, draws, n, NULL, rw, rh);
}

int vkp_pass_begin_hdr(int scene_w, int scene_h, const struct vkp_draw *draws, int n,
                       const struct vkp_hdr_frame *hf, int *rw, int *rh) {
    if (!hf) return -1;
    return pass_begin_impl(scene_w, scene_h, draws, n, hf, rw, rh);
}

/* The off-screen pass for a layer: the SDR one (hf == NULL: composite into the 8-bit scene image, the
 * effects in 8-bit - unchanged) or the HDR one (the draws composed into one 10-bit PQ BT.2020 picture,
 * the effects run on it in 10-bit). */
static int pass_begin_impl(int scene_w, int scene_h, const struct vkp_draw *draws, int n,
                           const struct vkp_hdr_frame *hf, int *rw, int *rh) {
    if (g_pass.active) vkp_pass_abort();
    if (g_dev_state == -2 || dev_init() != 0 || !g_window || scene_w <= 0 || scene_h <= 0) return -1;
    if (!g_swapchain && swap_init() != 0) return -1; /* the mapping is in output pixels */
    update_map(scene_w, scene_h);
    if (!g_map.valid) return -1;
    if (hf) {
        if (ensure_img(&g_hdrscene, scene_w, scene_h, HDR_FMT,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                       "HDR composition (picture)") != 0)
            return -1;
    } else if (ensure_scene_image(scene_w, scene_h) != 0) return -1;

    VkCommandBuffer cmd = g_cmds[0];
    g_vk.ResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    g_vk.BeginCommandBuffer(cmd, &bi);
    blendp_begin_frame();
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    /* Same acquisition as vkp_render's compositor pass: the sources come from the client's queue
     * family (dmabuf) or from a host write (shm), the scene image is the transfer destination. */
    VkImageMemoryBarrier *bars = calloc((size_t)n + 1, sizeof(*bars));
    if (!bars) { g_vk.EndCommandBuffer(cmd); return -1; }
    int nb = 0;
    VkImage scene_img = hf ? g_hdrscene.img : g_scene.img;
    bars[nb++] = (VkImageMemoryBarrier){
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = scene_img, .subresourceRange = range, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    for (int i = 0; i < n; i++) {
        struct vkp_image *im = draws[i].img;
        int seen = 0;
        for (int j = 0; j < i; j++) if (draws[j].img == im) { seen = 1; break; }
        if (seen || !im) continue;
        if (im->dmabuf) {
            bars[nb++] = (VkImageMemoryBarrier){
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .dstQueueFamilyIndex = g_qfam,
                .image = im->image, .subresourceRange = range, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
        } else if (!im->in_general) {
            bars[nb++] = (VkImageMemoryBarrier){
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = im->image, .subresourceRange = range,
                .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
            im->in_general = 1;
        }
    }
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, (uint32_t)nb, bars);
    free(bars);

    if (hf) {
        /* HDR: every draw into the mixed image, then ONE encoding into the 10-bit picture: PQ BT.2020, or
         * tone-mapped sRGB while the drawer's HDR output switch is off. */
        if (compose_hdr(cmd, draws, n, hf, scene_w, scene_h, g_hdrscene.img, HDR_FMT,
                        hf->tonemap ? HDRC_OUT_SDR : HDRC_OUT_PQ) != 0) {
            g_vk.EndCommandBuffer(cmd); /* never submitted; reset before its next use */
            return -1;
        }
        if (vkp_effects_active()) vkp_effects_set_formats(HDR_FMT, HDR_FMT);
    } else {
        VkClearColorValue black = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
        g_vk.CmdClearColorImage(cmd, g_scene.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
        g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0, 1, &mb, 0, NULL, 0, NULL);
        for (int i = 0; i < n; i++) {
            VkImageBlit blit;
            if (!draws[i].img || !draw_to_scene_blit(&draws[i], scene_w, scene_h, &blit)) continue;
            record_draw(cmd, &draws[i], &blit, g_scene.img, VK_FORMAT_R8G8B8A8_UNORM, scene_w, scene_h,
                        VK_FILTER_LINEAR);
        }
        if (vkp_effects_active()) vkp_effects_set_formats(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM);
    }

    int mapped_w = (int)(scene_w * g_map.kx + 0.5f), mapped_h = (int)(scene_h * g_map.ky + 0.5f);
    g_pass.rw = scene_w; g_pass.rh = scene_h;
    g_pass.result = vkp_effects_run(cmd, scene_img, scene_w, scene_h, mapped_w, mapped_h,
                                    &g_pass.rw, &g_pass.rh);
    g_pass.active = 1;
    if (rw) *rw = g_pass.rw;
    if (rh) *rh = g_pass.rh;
    return 0;
}

int vkp_pass_copy_to(struct vkp_image *dst, int wait_fd) {
    if (!g_pass.active) { if (wait_fd >= 0) close(wait_fd); return -1; }
    g_pass.active = 0;
    VkCommandBuffer cmd = g_cmds[0];
    if (!dst || !dst->blit_dst || g_dev_state == -2) {
        if (wait_fd >= 0) close(wait_fd);
        g_vk.EndCommandBuffer(cmd);
        return -1;
    }
    const int gpu_wait = take_wait_fd(wait_fd); /* the display's release of dst, before the copy writes it */
    if (gpu_wait < 0) { g_vk.EndCommandBuffer(cmd); return -1; } /* never submitted; reset before its next use */
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageMemoryBarrier acq = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
        .dstQueueFamilyIndex = g_qfam, .image = dst->image, .subresourceRange = range,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, NULL, 0, NULL, 1, &acq);
    /* The result stands for the whole scene, and so does the layer buffer (the DPU maps it onto
     * the output), so this is 1:1 whenever the buffer was allocated at the size begin() reported;
     * a stale-sized buffer is resampled rather than shown wrong. */
    VkImageBlit blit = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .srcOffsets = {{0, 0, 0}, {g_pass.rw, g_pass.rh, 1}},
                        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                        .dstOffsets = {{0, 0, 0}, {dst->w, dst->h, 1}}};
    g_vk.CmdBlitImage(cmd, g_pass.result, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst->image,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                      (g_pass.rw == dst->w && g_pass.rh == dst->h) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    VkImageMemoryBarrier rel = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = g_qfam,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .image = dst->image, .subresourceRange = range,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            0, 0, NULL, 0, NULL, 1, &rel);
    g_vk.EndCommandBuffer(cmd);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
                       .waitSemaphoreCount = gpu_wait ? 1u : 0u, .pWaitSemaphores = gpu_wait ? &g_wait_sem : NULL,
                       .pWaitDstStageMask = gpu_wait ? &wait_stage : NULL};
    g_vk.ResetFences(g_dev, 1, &g_fence);
    VkResult qr = g_vk.QueueSubmit(g_queue, 1, &si, g_fence);
    if (qr != VK_SUCCESS) {
        if (qr == VK_ERROR_DEVICE_LOST) device_lost("layer pass");
        else LOGE("layer: effects pass submit failed (%d)", (int)qr);
        return -1;
    }
    VkResult fr = timed_wait(g_fence, 1000000000ULL);
    if (fr == VK_ERROR_DEVICE_LOST) { device_lost("layer pass"); return -1; }
    if (fr != VK_SUCCESS) { LOGE("layer: effects pass fence wait -> %s", vk_result_name(fr)); return -1; }
    return 0;
}

void vkp_pass_abort(void) {
    if (!g_pass.active) return;
    g_pass.active = 0;
    g_vk.EndCommandBuffer(g_cmds[0]); /* never submitted; the buffer is reset before its next use */
}
