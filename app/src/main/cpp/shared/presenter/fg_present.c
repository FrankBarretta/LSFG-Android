#include "fg_present.h"

#include <android/hardware_buffer.h>
#include <android/log.h>
#include <errno.h>
#include <math.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <poll.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../vk_dispatch.h"
#include "../vk_driver.h"
#include "vkr_dis.h"
#include "../lsfg/vkr_lsfg.h"

#define LOG_TAG "FgPresent"
#define FG_LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define FG_LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define FG_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define FG_FRAMES_IN_FLIGHT 2u

#if VKR_DIS_MAX_GENERATIONS > VKR_LSFG_MAX_GENERATIONS
#define FG_MAX_GENERATIONS VKR_DIS_MAX_GENERATIONS
#else
#define FG_MAX_GENERATIONS VKR_LSFG_MAX_GENERATIONS
#endif

#define FG_MAX_TARGETS (FG_FRAMES_IN_FLIGHT + FG_MAX_GENERATIONS)
#define FG_MAX_SWAPCHAIN_IMAGES 8u
#define FG_READER_IMAGES 4
#define FG_IMPORT_CACHE 8u
#define FG_ACQUIRE_TIMEOUT_NS 4000000ULL
#define FG_ACQUIRE_TIMEOUT_MAX_NS 33000000ULL
#define FG_MAX_RETIRED 32u
#define FG_FENCE_WAIT_MS 250
#define FG_TELEMETRY_FRAMES 120ULL
#define FG_ARRIVAL_WINDOW 0.5f
#define FG_ARRIVAL_TOLERANCE 0.08f

typedef struct {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
} FgTarget;

typedef struct {
    AHardwareBuffer* buffer;
    VkImage image;
    VkDeviceMemory memory;
    uint32_t width;
    uint32_t height;
} FgImport;

typedef struct {
    VkCommandBuffer cmd;
    VkFence fence;
    VkSemaphore acquire;
    VkSemaphore acquire_gen[FG_MAX_GENERATIONS];
    AImage* held;
    bool submitted;
} FgFrame;

struct FgPresenter {
    void* vulkan_handle;
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID get_ahb_props;
    VkPhysicalDeviceMemoryProperties memory_properties;

    ANativeWindow* output;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkFormat swapchain_format;
    VkExtent2D extent;
    VkImage swapchain_images[FG_MAX_SWAPCHAIN_IMAGES];
    VkSemaphore swapchain_ready[FG_MAX_SWAPCHAIN_IMAGES];
    uint32_t swapchain_image_count;

    VkCommandPool command_pool;
    FgFrame frames[FG_FRAMES_IN_FLIGHT];
    VkFence flush_fence;
    uint32_t frame_index;

    FgTarget targets[FG_MAX_TARGETS];
    VkFormat target_format;
    bool targets_built;

    FgImport imports[FG_IMPORT_CACHE];

    AImageReader* reader;
    AImageReader_ImageListener listener;
    ANativeWindow* producer;

    VkrLsfg* lsfg;
    VkrDis* dis;
    uint32_t active_engine;
    char* cache_path;

    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t signal;
    bool running;
    bool image_pending;

    uint32_t multiplier;
    uint32_t target_rate;
    float flow_scale;
    uint32_t engine;
    uint32_t dis_min_side;
    float refresh_rate;
    float source_rate;
    bool config_dirty;

    VkSemaphore retired[FG_MAX_RETIRED];
    uint32_t retired_count;

    uint32_t source_divisor;
    uint32_t divisor_phase;
    uint64_t raw_frames;
    struct timespec raw_mark;
    float raw_rate;

    uint64_t source_frames;
    uint64_t real_frames;
    uint64_t generated_frames;
    uint64_t log_real;
    uint64_t log_generated;
    uint64_t acquire_misses;
};

static void fg_reset_swapchain(FgPresenter* fg);

static bool fg_has_extension(const VkExtensionProperties* list, uint32_t count, const char* name) {
    for (uint32_t i = 0; i < count; i++) {
        if (strcmp(list[i].extensionName, name) == 0) return true;
    }
    return false;
}

static uint32_t fg_find_memory_type(FgPresenter* fg, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < fg->memory_properties.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        if ((fg->memory_properties.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return UINT32_MAX;
}

static void fg_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout,
                       VkImageLayout new_layout, VkPipelineStageFlags src_stage,
                       VkPipelineStageFlags dst_stage, VkAccessFlags src_access,
                       VkAccessFlags dst_access) {
    VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &barrier);
}

static void fg_blit(VkCommandBuffer cmd, VkImage src, VkImageLayout src_layout, uint32_t src_w,
                    uint32_t src_h, VkImage dst, VkImageLayout dst_layout, uint32_t dst_w,
                    uint32_t dst_h) {
    VkImageBlit blit = {0};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1].x = (int32_t)src_w;
    blit.srcOffsets[1].y = (int32_t)src_h;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[1].x = (int32_t)dst_w;
    blit.dstOffsets[1].y = (int32_t)dst_h;
    blit.dstOffsets[1].z = 1;
    vkCmdBlitImage(cmd, src, src_layout, dst, dst_layout, 1, &blit, VK_FILTER_LINEAR);
}

static bool fg_create_instance(FgPresenter* fg) {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &count, NULL);
    VkExtensionProperties* exts = calloc(count ? count : 1, sizeof(VkExtensionProperties));
    if (!exts) return false;
    vkEnumerateInstanceExtensionProperties(NULL, &count, exts);

    const char* required[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
    for (uint32_t i = 0; i < 2; i++) {
        if (!fg_has_extension(exts, count, required[i])) {
            FG_LOGE("missing instance extension %s", required[i]);
            free(exts);
            return false;
        }
    }
    free(exts);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "WinNative";
    app.pEngineName = "WinNativeFrameGen";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2;
    ici.ppEnabledExtensionNames = required;

    VkResult res = vkCreateInstance(&ici, NULL, &fg->instance);
    if (res != VK_SUCCESS) {
        app.apiVersion = VK_API_VERSION_1_1;
        res = vkCreateInstance(&ici, NULL, &fg->instance);
    }
    if (res != VK_SUCCESS) {
        FG_LOGE("vkCreateInstance -> %d", res);
        return false;
    }
    return vkd_load_instance(fg->instance);
}

static bool fg_pick_physical_device(FgPresenter* fg) {
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(fg->instance, &count, NULL) != VK_SUCCESS || count == 0) {
        return false;
    }
    if (count > 8) count = 8;
    VkPhysicalDevice devices[8];
    if (vkEnumeratePhysicalDevices(fg->instance, &count, devices) != VK_SUCCESS) return false;

    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.apiVersion < VK_API_VERSION_1_3) {
            FG_LOGW("%s reports Vulkan %u.%u; frame generation needs 1.3", props.deviceName,
                    VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));
            continue;
        }

        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &families, NULL);
        if (families == 0) continue;
        if (families > 16) families = 16;
        VkQueueFamilyProperties family_props[16];
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &families, family_props);

        for (uint32_t q = 0; q < families; q++) {
            const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            if (family_props[q].queueCount == 0) continue;
            if ((family_props[q].queueFlags & need) != need) continue;
            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(devices[i], q, fg->surface, &supported);
            if (!supported) continue;
            fg->physical_device = devices[i];
            fg->queue_family = q;
            vkGetPhysicalDeviceMemoryProperties(devices[i], &fg->memory_properties);
            FG_LOGI("frame generation device %s queue family %u", props.deviceName, q);
            return true;
        }
    }
    return false;
}

static bool fg_create_device(FgPresenter* fg) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(fg->physical_device, NULL, &count, NULL);
    VkExtensionProperties* exts = calloc(count ? count : 1, sizeof(VkExtensionProperties));
    if (!exts) return false;
    vkEnumerateDeviceExtensionProperties(fg->physical_device, NULL, &count, exts);

    const char* required[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
        VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
    };
    const char* enable[8];
    uint32_t enable_n = 0;
    for (uint32_t i = 0; i < sizeof(required) / sizeof(required[0]); i++) {
        if (!fg_has_extension(exts, count, required[i])) {
            FG_LOGE("missing device extension %s", required[i]);
            free(exts);
            return false;
        }
        enable[enable_n++] = required[i];
    }
    if (fg_has_extension(exts, count, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
        enable[enable_n++] = VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME;
    }
    if (fg_has_extension(exts, count, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME)) {
        enable[enable_n++] = VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME;
    }
    free(exts);

    VkPhysicalDeviceVulkanMemoryModelFeatures memory_model = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES
    };
    VkPhysicalDeviceFeatures2 features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &memory_model;
    vkGetPhysicalDeviceFeatures2(fg->physical_device, &features);

    if (!memory_model.vulkanMemoryModel || !features.features.shaderStorageImageWriteWithoutFormat
        || !features.features.shaderStorageImageExtendedFormats) {
        FG_LOGE("device lacks the shader features the Lossless Scaling chain requires");
        return false;
    }

    VkPhysicalDeviceFeatures enabled = {0};
    enabled.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    enabled.shaderStorageImageExtendedFormats = VK_TRUE;

    VkPhysicalDeviceVulkanMemoryModelFeatures enable_memory_model = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES
    };
    enable_memory_model.vulkanMemoryModel = VK_TRUE;
    enable_memory_model.vulkanMemoryModelDeviceScope = memory_model.vulkanMemoryModelDeviceScope;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = fg->queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enable_memory_model;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = enable_n;
    dci.ppEnabledExtensionNames = enable;
    dci.pEnabledFeatures = &enabled;

    VkResult res = vkCreateDevice(fg->physical_device, &dci, NULL, &fg->device);
    if (res != VK_SUCCESS) {
        FG_LOGE("vkCreateDevice -> %d", res);
        return false;
    }
    vkGetDeviceQueue(fg->device, fg->queue_family, 0, &fg->queue);

    fg->get_ahb_props = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(
        fg->device, "vkGetAndroidHardwareBufferPropertiesANDROID");
    if (!fg->get_ahb_props) {
        FG_LOGE("vkGetAndroidHardwareBufferPropertiesANDROID unavailable");
        return false;
    }
    return true;
}

static void fg_destroy_swapchain(FgPresenter* fg) {
    for (uint32_t i = 0; i < FG_MAX_SWAPCHAIN_IMAGES; i++) {
        if (fg->swapchain_ready[i]) {
            vkDestroySemaphore(fg->device, fg->swapchain_ready[i], NULL);
            fg->swapchain_ready[i] = VK_NULL_HANDLE;
        }
        fg->swapchain_images[i] = VK_NULL_HANDLE;
    }
    fg->swapchain_image_count = 0;
    if (fg->swapchain) {
        vkDestroySwapchainKHR(fg->device, fg->swapchain, NULL);
        fg->swapchain = VK_NULL_HANDLE;
    }
}

static uint32_t fg_wanted_images(const FgPresenter* fg) {
    uint32_t generations = fg->target_rate != 0 ? FG_MAX_GENERATIONS
                                                : (fg->multiplier > 1 ? fg->multiplier - 1 : 1);
    if (generations > FG_MAX_GENERATIONS) generations = FG_MAX_GENERATIONS;
    return (generations + 1) * 2;
}

static bool fg_create_swapchain(FgPresenter* fg) {
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(fg->physical_device, fg->surface, &caps)
        != VK_SUCCESS) {
        return false;
    }

    uint32_t format_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(fg->physical_device, fg->surface, &format_count, NULL);
    if (format_count == 0) return false;
    if (format_count > 32) format_count = 32;
    VkSurfaceFormatKHR formats[32];
    vkGetPhysicalDeviceSurfaceFormatsKHR(fg->physical_device, fg->surface, &format_count, formats);

    VkSurfaceFormatKHR chosen = formats[0];
    for (uint32_t i = 0; i < format_count; i++) {
        if (formats[i].format == VK_FORMAT_R8G8B8A8_UNORM
            && formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = formats[i];
            break;
        }
        if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM) chosen = formats[i];
    }

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) extent = fg->extent;
    if (extent.width == 0 || extent.height == 0) return false;

    uint32_t images = fg_wanted_images(fg);
    if (images < caps.minImageCount + 1) images = caps.minImageCount + 1;
    if (caps.maxImageCount != 0 && images > caps.maxImageCount) images = caps.maxImageCount;
    if (images > FG_MAX_SWAPCHAIN_IMAGES) images = FG_MAX_SWAPCHAIN_IMAGES;

    VkSwapchainCreateInfoKHR sci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = fg->surface;
    sci.minImageCount = images;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                           ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                           : caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(fg->device, &sci, NULL, &fg->swapchain) != VK_SUCCESS) {
        FG_LOGE("vkCreateSwapchainKHR failed");
        return false;
    }

    uint32_t got = 0;
    vkGetSwapchainImagesKHR(fg->device, fg->swapchain, &got, NULL);
    if (got > FG_MAX_SWAPCHAIN_IMAGES) got = FG_MAX_SWAPCHAIN_IMAGES;
    vkGetSwapchainImagesKHR(fg->device, fg->swapchain, &got, fg->swapchain_images);
    fg->swapchain_image_count = got;
    fg->swapchain_format = chosen.format;
    fg->extent = extent;

    VkSemaphoreCreateInfo sem = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < got; i++) {
        if (vkCreateSemaphore(fg->device, &sem, NULL, &fg->swapchain_ready[i]) != VK_SUCCESS) {
            return false;
        }
    }

    FG_LOGI("swapchain %ux%u format=%d images=%u currentTransform=0x%x preTransform=0x%x",
            extent.width, extent.height, (int)chosen.format, got, caps.currentTransform,
            sci.preTransform);
    return true;
}

static void fg_destroy_targets(FgPresenter* fg) {
    for (uint32_t i = 0; i < FG_MAX_TARGETS; i++) {
        FgTarget* t = &fg->targets[i];
        if (t->view) vkDestroyImageView(fg->device, t->view, NULL);
        if (t->image) vkDestroyImage(fg->device, t->image, NULL);
        if (t->memory) vkFreeMemory(fg->device, t->memory, NULL);
        memset(t, 0, sizeof(*t));
    }
    fg->targets_built = false;
}

static VkFormat fg_pick_target_format(FgPresenter* fg) {
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
                                    | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
                                    | VK_FORMAT_FEATURE_BLIT_SRC_BIT
                                    | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    const VkFormat candidates[2] = {fg->swapchain_format, VK_FORMAT_R8G8B8A8_UNORM};
    for (uint32_t i = 0; i < 2; i++) {
        if (candidates[i] == VK_FORMAT_UNDEFINED) continue;
        VkFormatProperties props;
        memset(&props, 0, sizeof(props));
        vkGetPhysicalDeviceFormatProperties(fg->physical_device, candidates[i], &props);
        if ((props.optimalTilingFeatures & need) == need) return candidates[i];
    }
    return VK_FORMAT_UNDEFINED;
}

static bool fg_create_targets(FgPresenter* fg) {
    fg_destroy_targets(fg);

    fg->target_format = fg_pick_target_format(fg);
    if (fg->target_format == VK_FORMAT_UNDEFINED) {
        FG_LOGE("no storage-capable colour format for the composite ring");
        return false;
    }

    for (uint32_t i = 0; i < FG_MAX_TARGETS; i++) {
        FgTarget* t = &fg->targets[i];

        VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fg->target_format;
        ici.extent.width = fg->extent.width;
        ici.extent.height = fg->extent.height;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                  | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(fg->device, &ici, NULL, &t->image) != VK_SUCCESS) goto fail;

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(fg->device, t->image, &req);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex =
            fg_find_memory_type(fg, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX) goto fail;
        if (vkAllocateMemory(fg->device, &mai, NULL, &t->memory) != VK_SUCCESS) goto fail;
        if (vkBindImageMemory(fg->device, t->image, t->memory, 0) != VK_SUCCESS) goto fail;

        VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = t->image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fg->target_format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(fg->device, &vci, NULL, &t->view) != VK_SUCCESS) goto fail;
    }

    fg->targets_built = true;
    return true;

fail:
    fg_destroy_targets(fg);
    return false;
}

static bool fg_create_frames(FgPresenter* fg) {
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = fg->queue_family;
    if (vkCreateCommandPool(fg->device, &cpi, NULL, &fg->command_pool) != VK_SUCCESS) return false;

    VkFenceCreateInfo ffi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(fg->device, &ffi, NULL, &fg->flush_fence) != VK_SUCCESS) return false;

    for (uint32_t i = 0; i < FG_FRAMES_IN_FLIGHT; i++) {
        FgFrame* f = &fg->frames[i];

        VkCommandBufferAllocateInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbi.commandPool = fg->command_pool;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(fg->device, &cbi, &f->cmd) != VK_SUCCESS) return false;

        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(fg->device, &fci, NULL, &f->fence) != VK_SUCCESS) return false;

        VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(fg->device, &sci, NULL, &f->acquire) != VK_SUCCESS) return false;
        for (uint32_t g = 0; g < FG_MAX_GENERATIONS; g++) {
            if (vkCreateSemaphore(fg->device, &sci, NULL, &f->acquire_gen[g]) != VK_SUCCESS) {
                return false;
            }
        }
    }
    return true;
}

static void fg_release_imports(FgPresenter* fg) {
    for (uint32_t i = 0; i < FG_IMPORT_CACHE; i++) {
        FgImport* imp = &fg->imports[i];
        if (imp->image) vkDestroyImage(fg->device, imp->image, NULL);
        if (imp->memory) vkFreeMemory(fg->device, imp->memory, NULL);
        if (imp->buffer) AHardwareBuffer_release(imp->buffer);
        memset(imp, 0, sizeof(*imp));
    }
}

static FgImport* fg_import_buffer(FgPresenter* fg, AHardwareBuffer* buffer) {
    for (uint32_t i = 0; i < FG_IMPORT_CACHE; i++) {
        if (fg->imports[i].buffer == buffer) return &fg->imports[i];
    }

    FgImport* slot = NULL;
    for (uint32_t i = 0; i < FG_IMPORT_CACHE; i++) {
        if (!fg->imports[i].buffer) {
            slot = &fg->imports[i];
            break;
        }
    }
    if (!slot) {
        vkDeviceWaitIdle(fg->device);
        fg_release_imports(fg);
        slot = &fg->imports[0];
    }

    VkAndroidHardwareBufferFormatPropertiesANDROID format_props = {
        VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID
    };
    VkAndroidHardwareBufferPropertiesANDROID props = {
        VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID
    };
    props.pNext = &format_props;
    if (fg->get_ahb_props(fg->device, buffer, &props) != VK_SUCCESS) {
        FG_LOGW("vkGetAndroidHardwareBufferPropertiesANDROID failed");
        return NULL;
    }
    AHardwareBuffer_Desc desc;
    memset(&desc, 0, sizeof(desc));
    AHardwareBuffer_describe(buffer, &desc);

    if (format_props.format == VK_FORMAT_UNDEFINED) {
        FG_LOGW("producer buffer format %u is external-only; frame generation cannot sample it",
                desc.format);
        return NULL;
    }

    VkExternalMemoryImageCreateInfo emi = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.pNext = &emi;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format_props.format;
    ici.extent.width = desc.width;
    ici.extent.height = desc.height;
    ici.extent.depth = 1;
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(fg->device, &ici, NULL, &image) != VK_SUCCESS) {
        FG_LOGW("AHB vkCreateImage failed");
        return NULL;
    }

    VkImportAndroidHardwareBufferInfoANDROID import = {
        VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID
    };
    import.buffer = buffer;

    VkMemoryDedicatedAllocateInfo dedicated = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.image = image;
    dedicated.pNext = &import;

    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &dedicated;
    mai.allocationSize = props.allocationSize;
    mai.memoryTypeIndex = fg_find_memory_type(fg, props.memoryTypeBits, 0);

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (mai.memoryTypeIndex == UINT32_MAX
        || vkAllocateMemory(fg->device, &mai, NULL, &memory) != VK_SUCCESS) {
        FG_LOGW("AHB memory import failed");
        vkDestroyImage(fg->device, image, NULL);
        return NULL;
    }
    if (vkBindImageMemory(fg->device, image, memory, 0) != VK_SUCCESS) {
        FG_LOGW("AHB vkBindImageMemory failed");
        vkFreeMemory(fg->device, memory, NULL);
        vkDestroyImage(fg->device, image, NULL);
        return NULL;
    }

    FG_LOGI("imported producer buffer %ux%u ahb format %u as vk format %d", desc.width,
            desc.height, desc.format, (int)format_props.format);
    AHardwareBuffer_acquire(buffer);
    slot->buffer = buffer;
    slot->image = image;
    slot->memory = memory;
    slot->width = desc.width;
    slot->height = desc.height;
    return slot;
}

static void fg_await_fence_fd(int fence_fd) {
    if (fence_fd < 0) return;
    struct pollfd pfd = {fence_fd, POLLIN, 0};
    int rc;
    do {
        rc = poll(&pfd, 1, FG_FENCE_WAIT_MS);
    } while (rc < 0 && errno == EINTR);
    close(fence_fd);
}

static void fg_image_available(void* context, AImageReader* reader) {
    (void)reader;
    FgPresenter* fg = (FgPresenter*)context;
    pthread_mutex_lock(&fg->lock);
    fg->image_pending = true;
    pthread_cond_signal(&fg->signal);
    pthread_mutex_unlock(&fg->lock);
}

static uint32_t fg_dis_target(uint32_t multiplier, uint32_t target_rate, float source_rate) {
    if (target_rate != 0) return target_rate;
    if (multiplier > 1 && source_rate > 1.0f) {
        return (uint32_t)(source_rate * (float)multiplier + 0.5f);
    }
    return 0;
}

static void fg_engine_configure(FgPresenter* fg, uint32_t multiplier, uint32_t target_rate,
                                float flow_scale, float refresh_rate, float source_rate,
                                uint32_t dis_min_side) {
    if (fg->active_engine == FG_ENGINE_DIS) {
        if (!fg->dis) return;
        vkr_dis_configure(fg->dis, dis_min_side ? dis_min_side : FG_DIS_MIN_SIDE_DEFAULT,
                          fg_dis_target(multiplier, target_rate, source_rate), refresh_rate);
        return;
    }
    if (!fg->lsfg) return;
    vkr_lsfg_configure(fg->lsfg, multiplier ? multiplier : 2u, target_rate,
                       flow_scale > 0.0f ? flow_scale : 0.7f, refresh_rate, source_rate);
}

static void fg_engine_release(FgPresenter* fg) {
    if (fg->lsfg) {
        vkr_lsfg_destroy(fg->lsfg);
        fg->lsfg = NULL;
    }
    if (fg->dis) {
        vkr_dis_destroy(fg->dis);
        fg->dis = NULL;
    }
}

static bool fg_engine_acquire(FgPresenter* fg) {
    if (fg->active_engine == FG_ENGINE_DIS) {
        if (fg->dis) return true;
        fg->dis = vkr_dis_create(fg->device, fg->physical_device);
        if (!fg->dis) {
            FG_LOGW("DIS optical flow is unavailable on this device");
            return false;
        }
        return true;
    }
    if (fg->lsfg) return true;
    fg->lsfg = vkr_lsfg_create(fg->device, fg->physical_device, fg->cache_path);
    if (!fg->lsfg) {
        FG_LOGW("Lossless Scaling shaders unavailable at %s", fg->cache_path);
        return false;
    }
    return true;
}

static bool fg_engine_ready(const FgPresenter* fg) {
    return fg->active_engine == FG_ENGINE_DIS ? fg->dis != NULL : fg->lsfg != NULL;
}

static const char* fg_engine_name(uint32_t engine) {
    return engine == FG_ENGINE_DIS ? "DIS" : "LSFG";
}

static void fg_apply_config(FgPresenter* fg) {
    pthread_mutex_lock(&fg->lock);
    bool dirty = fg->config_dirty;
    uint32_t multiplier = fg->multiplier;
    uint32_t target_rate = fg->target_rate;
    float source_rate = fg->source_rate;
    float flow_scale = fg->flow_scale;
    float refresh_rate = fg->refresh_rate;
    uint32_t engine = fg->engine;
    uint32_t dis_min_side = fg->dis_min_side;
    fg->config_dirty = false;
    pthread_mutex_unlock(&fg->lock);

    if (!dirty) return;

    if (engine != fg->active_engine) {
        const uint32_t previous = fg->active_engine;
        vkDeviceWaitIdle(fg->device);
        fg_engine_release(fg);
        fg->active_engine = engine;
        if (!fg_engine_acquire(fg)) {
            fg->active_engine = previous;
            if (!fg_engine_acquire(fg)) {
                FG_LOGE("no frame generation engine could be started");
                return;
            }
            FG_LOGW("%s could not start; staying on %s", fg_engine_name(engine),
                    fg_engine_name(previous));
            pthread_mutex_lock(&fg->lock);
            fg->engine = previous;
            pthread_mutex_unlock(&fg->lock);
        } else {
            FG_LOGI("frame generation engine -> %s", fg_engine_name(fg->active_engine));
        }
    }

    fg_engine_configure(fg, multiplier, target_rate, flow_scale, refresh_rate, source_rate,
                        dis_min_side);
}

static void fg_renew_semaphore(FgPresenter* fg, VkSemaphore* handle) {
    if (fg->retired_count >= FG_MAX_RETIRED) return;
    VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore fresh = VK_NULL_HANDLE;
    if (vkCreateSemaphore(fg->device, &sci, NULL, &fresh) != VK_SUCCESS) return;
    if (*handle) fg->retired[fg->retired_count++] = *handle;
    *handle = fresh;
}

// VkrDisFlushFn: DIS needs this frame's pixels on the CPU mid-frame for the hardware motion
// estimator. Everything recorded so far - the source blit and DIS's copies - touches no
// swapchain image and waits on no semaphore, so it goes out on its own; the same buffer is then
// begun again for the rest of the frame, which keeps every semaphore on the final submit.
static VkCommandBuffer fg_dis_flush(void* user, VkCommandBuffer cmd) {
    FgPresenter* fg = (FgPresenter*)user;
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkResetFences(fg->device, 1, &fg->flush_fence);
    if (vkQueueSubmit(fg->queue, 1, &si, fg->flush_fence) == VK_SUCCESS) {
        vkWaitForFences(fg->device, 1, &fg->flush_fence, VK_TRUE, UINT64_MAX);
    } else {
        FG_LOGW("DIS mid-frame submit failed");
        vkDeviceWaitIdle(fg->device);
    }
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

static void fg_record_and_present(FgPresenter* fg, FgImport* source, AImage* image) {
    FgFrame* f = &fg->frames[fg->frame_index];

    uint32_t capacity = 0;
    if (fg_engine_ready(fg) && fg->swapchain_image_count > 2) {
        capacity = fg->swapchain_image_count - 2;
        if (capacity > FG_MAX_GENERATIONS) capacity = FG_MAX_GENERATIONS;
    }

    uint32_t planned = 0;
    if (fg->active_engine == FG_ENGINE_DIS) {
        planned = vkr_dis_plan(fg->dis, capacity, fg->source_frames);
    } else if (fg->lsfg) {
        vkr_lsfg_set_guest_extent(fg->lsfg, source->width, source->height);
        planned = vkr_lsfg_plan(fg->lsfg, capacity, fg->source_frames);
    }

    uint32_t image_index = 0;
    VkResult acq = vkAcquireNextImageKHR(fg->device, fg->swapchain, UINT64_MAX, f->acquire,
                                         VK_NULL_HANDLE, &image_index);
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        AImage_delete(image);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR) fg_reset_swapchain(fg);
        return;
    }

    uint64_t gen_timeout = FG_ACQUIRE_TIMEOUT_MAX_NS;
    if (fg->refresh_rate > 1.0f) {
        gen_timeout = (uint64_t)(2000000000.0f / fg->refresh_rate);
        if (gen_timeout < FG_ACQUIRE_TIMEOUT_NS) gen_timeout = FG_ACQUIRE_TIMEOUT_NS;
        if (gen_timeout > FG_ACQUIRE_TIMEOUT_MAX_NS) gen_timeout = FG_ACQUIRE_TIMEOUT_MAX_NS;
    }

    uint32_t gen_count = 0;
    uint32_t gen_index[FG_MAX_GENERATIONS] = {0};
    for (uint32_t g = 0; g < planned; g++) {
        uint32_t index = 0;
        VkResult ga = vkAcquireNextImageKHR(fg->device, fg->swapchain, gen_timeout,
                                            f->acquire_gen[g], VK_NULL_HANDLE, &index);
        if (ga != VK_SUCCESS && ga != VK_SUBOPTIMAL_KHR) {
            if (fg->acquire_misses++ % 240 == 0) {
                FG_LOGW("generated frame %u/%u dropped: acquire -> %d", g + 1, planned, (int)ga);
            }
            fg_renew_semaphore(fg, &f->acquire_gen[g]);
            break;
        }
        gen_index[gen_count++] = index;
    }

    vkResetFences(fg->device, 1, &f->fence);
    vkResetCommandBuffer(f->cmd, 0);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f->cmd, &bi);

    FgTarget* composite = &fg->targets[fg->frame_index];

    fg_barrier(f->cmd, source->image, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_READ_BIT);
    fg_barrier(f->cmd, composite->image, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    fg_blit(f->cmd, source->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, source->width,
            source->height, composite->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            fg->extent.width, fg->extent.height);
    fg_barrier(f->cmd, composite->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                   | VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);

    if (fg->active_engine == FG_ENGINE_DIS) {
        vkr_dis_process_ex(fg->dis, f->cmd, composite->image, fg->extent.width,
                           fg->extent.height, gen_count, fg_dis_flush, fg);
    } else if (fg->lsfg) {
        vkr_lsfg_process(fg->lsfg, f->cmd, composite->image, fg->extent.width, fg->extent.height,
                         gen_count);
    }

    for (uint32_t g = 0; g < gen_count; g++) {
        FgTarget* generated = &fg->targets[FG_FRAMES_IN_FLIGHT + g];
        if (fg->active_engine == FG_ENGINE_DIS) {
            vkr_dis_generate_into(fg->dis, f->cmd, g, FG_FRAMES_IN_FLIGHT + g, generated->image,
                                  generated->view, fg->extent.width, fg->extent.height,
                                  VK_NULL_HANDLE);
            fg_barrier(f->cmd, generated->image, VK_IMAGE_LAYOUT_GENERAL,
                       VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT);
        } else {
            vkr_lsfg_generate_into(fg->lsfg, f->cmd, g, FG_FRAMES_IN_FLIGHT + g, generated->image,
                                   generated->view, fg->extent.width, fg->extent.height);
        }
        fg_barrier(f->cmd, fg->swapchain_images[gen_index[g]], VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        fg_blit(f->cmd, generated->image, VK_IMAGE_LAYOUT_GENERAL, fg->extent.width,
                fg->extent.height, fg->swapchain_images[gen_index[g]],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, fg->extent.width, fg->extent.height);
        fg_barrier(f->cmd, fg->swapchain_images[gen_index[g]],
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    }

    fg_barrier(f->cmd, composite->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
               VK_ACCESS_TRANSFER_READ_BIT);
    fg_barrier(f->cmd, fg->swapchain_images[image_index], VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    fg_blit(f->cmd, composite->image, VK_IMAGE_LAYOUT_GENERAL, fg->extent.width, fg->extent.height,
            fg->swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            fg->extent.width, fg->extent.height);
    fg_barrier(f->cmd, fg->swapchain_images[image_index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0);

    vkEndCommandBuffer(f->cmd);

    VkSemaphore wait[1 + FG_MAX_GENERATIONS];
    VkPipelineStageFlags stages[1 + FG_MAX_GENERATIONS];
    VkSemaphore signal[1 + FG_MAX_GENERATIONS];
    uint32_t wait_count = 0;
    uint32_t signal_count = 0;

    wait[wait_count] = f->acquire;
    stages[wait_count++] = VK_PIPELINE_STAGE_TRANSFER_BIT;
    signal[signal_count++] = fg->swapchain_ready[image_index];
    for (uint32_t g = 0; g < gen_count; g++) {
        wait[wait_count] = f->acquire_gen[g];
        stages[wait_count++] = VK_PIPELINE_STAGE_TRANSFER_BIT;
        signal[signal_count++] = fg->swapchain_ready[gen_index[g]];
    }

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = wait_count;
    si.pWaitSemaphores = wait;
    si.pWaitDstStageMask = stages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f->cmd;
    si.signalSemaphoreCount = signal_count;
    si.pSignalSemaphores = signal;

    if (vkQueueSubmit(fg->queue, 1, &si, f->fence) != VK_SUCCESS) {
        FG_LOGE("vkQueueSubmit failed; dropping this frame");
        vkDeviceWaitIdle(fg->device);
        AImage_delete(image);
        return;
    }
    f->submitted = true;
    f->held = image;

    bool out_of_date = false;
    for (uint32_t g = 0; g < gen_count; g++) {
        VkPresentInfoKHR gpi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        gpi.waitSemaphoreCount = 1;
        gpi.pWaitSemaphores = &fg->swapchain_ready[gen_index[g]];
        gpi.swapchainCount = 1;
        gpi.pSwapchains = &fg->swapchain;
        gpi.pImageIndices = &gen_index[g];
        VkResult pr = vkQueuePresentKHR(fg->queue, &gpi);
        if (pr == VK_ERROR_OUT_OF_DATE_KHR) out_of_date = true;
    }

    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &fg->swapchain_ready[image_index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &fg->swapchain;
    pi.pImageIndices = &image_index;
    VkResult pr = vkQueuePresentKHR(fg->queue, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR) out_of_date = true;

    fg->real_frames++;
    fg->generated_frames += gen_count;
    if ((fg->real_frames % FG_TELEMETRY_FRAMES) == 0) {
        const uint64_t d_real = fg->real_frames - fg->log_real;
        const uint64_t d_gen = fg->generated_frames - fg->log_generated;
        fg->log_real = fg->real_frames;
        fg->log_generated = fg->generated_frames;
        FG_LOGI("framegen[%s] real=%llu made=%llu ratio=%.2f planned=%u got=%u images=%u div=%u misses=%llu",
                fg_engine_name(fg->active_engine),
                (unsigned long long)fg->real_frames, (unsigned long long)fg->generated_frames,
                d_real ? (double)(d_real + d_gen) / (double)d_real : 0.0, planned, gen_count,
                fg->swapchain_image_count, fg->source_divisor,
                (unsigned long long)fg->acquire_misses);
    }

    fg->frame_index = (fg->frame_index + 1) % FG_FRAMES_IN_FLIGHT;

    if (out_of_date) fg_reset_swapchain(fg);
}

static void fg_reset_swapchain(FgPresenter* fg) {
    vkDeviceWaitIdle(fg->device);
    const VkExtent2D previous = fg->extent;
    fg_destroy_swapchain(fg);
    if (!fg_create_swapchain(fg)) return;
    if (previous.width != fg->extent.width || previous.height != fg->extent.height) {
        fg_destroy_targets(fg);
    }
}

static bool fg_prepare_chain(FgPresenter* fg) {
    if (fg->active_engine == FG_ENGINE_DIS) {
        if (!fg->dis) return false;
        const VkrDisContentRect content = {0, 0, fg->extent.width, fg->extent.height};
        if (!vkr_dis_needs_rebuild(fg->dis, fg->extent.width, fg->extent.height, fg->target_format,
                                   content)) {
            return true;
        }
        vkDeviceWaitIdle(fg->device);
        vkr_dis_forget_targets(fg->dis);
        return vkr_dis_prepare(fg->dis, fg->extent.width, fg->extent.height, fg->target_format,
                               content);
    }

    if (!fg->lsfg) return false;
    if (!vkr_lsfg_needs_rebuild(fg->lsfg, fg->extent.width, fg->extent.height, fg->target_format)) {
        return true;
    }
    vkDeviceWaitIdle(fg->device);
    vkr_lsfg_forget_targets(fg->lsfg);
    return vkr_lsfg_prepare(fg->lsfg, fg->extent.width, fg->extent.height, fg->target_format);
}

static float fg_elapsed_since(struct timespec* mark) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (mark->tv_sec == 0 && mark->tv_nsec == 0) {
        *mark = now;
        return 0.0f;
    }
    return (float)(now.tv_sec - mark->tv_sec) + (float)(now.tv_nsec - mark->tv_nsec) / 1.0e9f;
}

static void fg_track_arrivals(FgPresenter* fg) {
    fg->raw_frames++;
    const float elapsed = fg_elapsed_since(&fg->raw_mark);
    if (elapsed < FG_ARRIVAL_WINDOW) return;
    clock_gettime(CLOCK_MONOTONIC, &fg->raw_mark);
    const float rate = (float)fg->raw_frames / elapsed;
    fg->raw_frames = 0;
    fg->raw_rate = fg->raw_rate > 0.0f ? fg->raw_rate + (rate - fg->raw_rate) * 0.5f : rate;

    uint32_t divisor = 1;
    const float guest = fg->source_rate;
    if (guest > 1.0f && fg->raw_rate > guest) {
        const long ratio = lroundf(fg->raw_rate / guest);
        if (ratio > 1 && ratio <= (long)FG_MAX_GENERATIONS + 1) {
            const float folded = fg->raw_rate / (float)ratio;
            if (fabsf(folded - guest) <= guest * FG_ARRIVAL_TOLERANCE) divisor = (uint32_t)ratio;
        }
    }
    if (divisor != fg->source_divisor) {
        fg->source_divisor = divisor;
        fg->divisor_phase = 0;
        FG_LOGI("producer repeats every frame %u times at %.1f fps for a %.1f fps game",
                divisor, fg->raw_rate, guest);
    }
}

static void* fg_thread(void* arg) {
    FgPresenter* fg = (FgPresenter*)arg;

    while (true) {
        pthread_mutex_lock(&fg->lock);
        while (fg->running && !fg->image_pending) {
            pthread_cond_wait(&fg->signal, &fg->lock);
        }
        bool running = fg->running;
        fg->image_pending = false;
        pthread_mutex_unlock(&fg->lock);
        if (!running) break;

        if (!vkd_bind(fg->vulkan_handle, fg->instance)) break;
        fg_apply_config(fg);

        while (true) {
            fg_apply_config(fg);

            AImage* image = NULL;
            int fence_fd = -1;
            media_status_t status =
                AImageReader_acquireNextImageAsync(fg->reader, &image, &fence_fd);
            if (status != AMEDIA_OK || !image) break;

            fg_track_arrivals(fg);
            if (fg->source_divisor > 1) {
                fg->divisor_phase = (fg->divisor_phase + 1) % fg->source_divisor;
                if (fg->divisor_phase != 0) {
                    if (fence_fd >= 0) close(fence_fd);
                    AImage_delete(image);
                    continue;
                }
            }

            AHardwareBuffer* buffer = NULL;
            if (AImage_getHardwareBuffer(image, &buffer) != AMEDIA_OK || !buffer) {
                if (fence_fd >= 0) close(fence_fd);
                AImage_delete(image);
                continue;
            }

            fg_await_fence_fd(fence_fd);

            FgFrame* f = &fg->frames[fg->frame_index];
            if (f->submitted) {
                vkWaitForFences(fg->device, 1, &f->fence, VK_TRUE, UINT64_MAX);
            }
            if (f->held) {
                AImage_delete(f->held);
                f->held = NULL;
            }

            if (!fg->swapchain && !fg_create_swapchain(fg)) {
                AImage_delete(image);
                break;
            }
            if (!fg->targets_built && !fg_create_targets(fg)) {
                AImage_delete(image);
                break;
            }
            if (!fg_prepare_chain(fg)) {
                AImage_delete(image);
                break;
            }

            FgImport* source = fg_import_buffer(fg, buffer);
            if (!source) {
                AImage_delete(image);
                continue;
            }

            fg->source_frames++;
            fg_record_and_present(fg, source, image);
        }
    }

    return NULL;
}

FgPresenter* fg_create(JNIEnv* env, jobject context, const char* driver_name,
                       ANativeWindow* output, uint32_t width, uint32_t height,
                       const char* cache_path, uint32_t multiplier, uint32_t target_rate,
                       float flow_scale, float refresh_rate, float source_rate,
                       uint32_t engine, uint32_t dis_min_side) {
    if (!output || width == 0 || height == 0 || !cache_path) return NULL;

    FgPresenter* fg = calloc(1, sizeof(FgPresenter));
    if (!fg) return NULL;

    pthread_mutex_init(&fg->lock, NULL);
    pthread_cond_init(&fg->signal, NULL);
    fg->output = output;
    fg->extent.width = width;
    fg->extent.height = height;
    fg->multiplier = multiplier;
    fg->target_rate = target_rate;
    fg->flow_scale = flow_scale;
    fg->engine = engine == FG_ENGINE_DIS ? FG_ENGINE_DIS : FG_ENGINE_LSFG;
    fg->active_engine = fg->engine;
    fg->dis_min_side = dis_min_side ? dis_min_side : FG_DIS_MIN_SIDE_DEFAULT;
    fg->refresh_rate = refresh_rate;
    fg->source_rate = source_rate;
    fg->cache_path = strdup(cache_path);
    fg->source_divisor = 1;
    ANativeWindow_acquire(output);

    VkAndroidSurfaceCreateInfoKHR asi;
    memset(&asi, 0, sizeof(asi));
    asi.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    asi.window = output;

    fg->vulkan_handle = winlator_open_vulkan(env, context, driver_name);
    if (!fg->vulkan_handle) {
        FG_LOGE("no Vulkan driver for frame generation");
        goto fail;
    }
    if (!vkd_init(fg->vulkan_handle)) goto fail;
    if (!fg_create_instance(fg)) goto fail;

    if (vkCreateAndroidSurfaceKHR(fg->instance, &asi, NULL, &fg->surface) != VK_SUCCESS) {
        FG_LOGE("vkCreateAndroidSurfaceKHR failed");
        goto fail;
    }

    if (!fg_pick_physical_device(fg)) {
        FG_LOGE("no device supports frame generation");
        goto fail;
    }
    if (!fg_create_device(fg)) goto fail;
    if (!fg_create_frames(fg)) goto fail;
    if (!fg_create_swapchain(fg)) goto fail;
    if (!fg_create_targets(fg)) goto fail;

    if (!fg_engine_acquire(fg)) {
        const uint32_t fallback =
            fg->active_engine == FG_ENGINE_DIS ? FG_ENGINE_LSFG : FG_ENGINE_DIS;
        FG_LOGW("%s unavailable; trying %s", fg_engine_name(fg->active_engine),
                fg_engine_name(fallback));
        fg->active_engine = fallback;
        fg->engine = fallback;
        if (!fg_engine_acquire(fg)) goto fail;
    }
    fg_engine_configure(fg, multiplier, target_rate, flow_scale, refresh_rate, source_rate,
                        fg->dis_min_side);

    const uint64_t reader_usage =
        AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;
    if (AImageReader_newWithUsage((int32_t)fg->extent.width, (int32_t)fg->extent.height,
                                  AIMAGE_FORMAT_PRIVATE, reader_usage, FG_READER_IMAGES,
                                  &fg->reader) != AMEDIA_OK) {
        FG_LOGE("AImageReader_newWithUsage failed");
        goto fail;
    }
    if (AImageReader_getWindow(fg->reader, &fg->producer) != AMEDIA_OK || !fg->producer) {
        FG_LOGE("AImageReader_getWindow failed");
        goto fail;
    }
    ANativeWindow_acquire(fg->producer);

    fg->listener.context = fg;
    fg->listener.onImageAvailable = fg_image_available;
    AImageReader_setImageListener(fg->reader, &fg->listener);

    fg->running = true;
    if (pthread_create(&fg->thread, NULL, fg_thread, fg) != 0) {
        FG_LOGE("presenter thread could not start");
        fg->running = false;
        goto fail;
    }

    FG_LOGI("frame generation presenter ready %ux%u engine=%s multiplier=%u target=%u flow=%.2f "
            "dis_min_side=%u",
            fg->extent.width, fg->extent.height, fg_engine_name(fg->active_engine), multiplier,
            target_rate, (double)flow_scale, fg->dis_min_side);
    return fg;

fail:
    fg_destroy(fg);
    return NULL;
}

ANativeWindow* fg_producer_window(FgPresenter* fg) {
    return fg ? fg->producer : NULL;
}

void fg_configure(FgPresenter* fg, uint32_t multiplier, uint32_t target_rate, float flow_scale,
                  float refresh_rate, float source_rate, uint32_t engine, uint32_t dis_min_side) {
    if (!fg) return;
    pthread_mutex_lock(&fg->lock);
    fg->multiplier = multiplier;
    fg->target_rate = target_rate;
    fg->flow_scale = flow_scale;
    fg->engine = engine == FG_ENGINE_DIS ? FG_ENGINE_DIS : FG_ENGINE_LSFG;
    fg->dis_min_side = dis_min_side ? dis_min_side : FG_DIS_MIN_SIDE_DEFAULT;
    fg->refresh_rate = refresh_rate;
    fg->source_rate = source_rate;
    fg->config_dirty = true;
    pthread_mutex_unlock(&fg->lock);
}

void fg_stats(FgPresenter* fg, uint64_t* real_frames, uint64_t* generated_frames) {
    if (!fg) return;
    if (real_frames) *real_frames = __atomic_load_n(&fg->real_frames, __ATOMIC_RELAXED);
    if (generated_frames) {
        *generated_frames = __atomic_load_n(&fg->generated_frames, __ATOMIC_RELAXED);
    }
}

void fg_destroy(FgPresenter* fg) {
    if (!fg) return;

    if (fg->running) {
        pthread_mutex_lock(&fg->lock);
        fg->running = false;
        pthread_cond_broadcast(&fg->signal);
        pthread_mutex_unlock(&fg->lock);
        pthread_join(fg->thread, NULL);
    }

    if (fg->reader) AImageReader_setImageListener(fg->reader, NULL);

    if (fg->device) {
        vkd_bind(fg->vulkan_handle, fg->instance);
        vkDeviceWaitIdle(fg->device);
    }

    for (uint32_t i = 0; i < FG_FRAMES_IN_FLIGHT; i++) {
        FgFrame* f = &fg->frames[i];
        if (f->held) {
            AImage_delete(f->held);
            f->held = NULL;
        }
    }

    fg_engine_release(fg);
    if (fg->reader) {
        AImageReader_delete(fg->reader);
        fg->reader = NULL;
    }
    if (fg->producer) ANativeWindow_release(fg->producer);

    if (fg->device) {
        fg_release_imports(fg);
        fg_destroy_targets(fg);
        fg_destroy_swapchain(fg);
        for (uint32_t i = 0; i < FG_FRAMES_IN_FLIGHT; i++) {
            FgFrame* f = &fg->frames[i];
            if (f->fence) vkDestroyFence(fg->device, f->fence, NULL);
            if (f->acquire) vkDestroySemaphore(fg->device, f->acquire, NULL);
            for (uint32_t g = 0; g < FG_MAX_GENERATIONS; g++) {
                if (f->acquire_gen[g]) vkDestroySemaphore(fg->device, f->acquire_gen[g], NULL);
            }
        }
        for (uint32_t i = 0; i < fg->retired_count; i++) {
            vkDestroySemaphore(fg->device, fg->retired[i], NULL);
        }
        fg->retired_count = 0;
        if (fg->flush_fence) vkDestroyFence(fg->device, fg->flush_fence, NULL);
        if (fg->command_pool) vkDestroyCommandPool(fg->device, fg->command_pool, NULL);
        vkDestroyDevice(fg->device, NULL);
    }
    if (fg->surface) vkDestroySurfaceKHR(fg->instance, fg->surface, NULL);
    if (fg->instance) vkDestroyInstance(fg->instance, NULL);
    if (fg->output) ANativeWindow_release(fg->output);

    pthread_mutex_destroy(&fg->lock);
    pthread_cond_destroy(&fg->signal);
    free(fg->cache_path);
    free(fg);
}
