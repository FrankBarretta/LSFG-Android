#include "vk_dispatch.h"

#include <android/log.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#define LOG_TAG "VkDispatch"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

VkDispatch vkd;

static void* vkd_owner_handle;
static VkInstance vkd_owner_instance;

bool vkd_init(void* libvulkan_handle) {
    if (!libvulkan_handle) return false;
    memset(&vkd, 0, sizeof(vkd));
    vkd_owner_handle = libvulkan_handle;
    vkd_owner_instance = VK_NULL_HANDLE;

    vkd.GetInstanceProcAddr =
        (PFN_vkGetInstanceProcAddr)dlsym(libvulkan_handle, "vkGetInstanceProcAddr");
    if (!vkd.GetInstanceProcAddr) {
        LOGE("dlsym(vkGetInstanceProcAddr) failed: %s", dlerror());
        return false;
    }

    // Per spec, these three resolve with VK_NULL_HANDLE before any instance exists.
    vkd.CreateInstance = (PFN_vkCreateInstance)
        vkd.GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance");
    vkd.EnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)
        vkd.GetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
    vkd.EnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)
        vkd.GetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties");

    if (!vkd.CreateInstance) {
        LOGE("vkGetInstanceProcAddr returned NULL for vkCreateInstance");
        return false;
    }
    return true;
}

bool vkd_load_instance(VkInstance instance) {
    if (!vkd.GetInstanceProcAddr || instance == VK_NULL_HANDLE) return false;

    // Device entry points resolve via vkGetInstanceProcAddr too — the loader trampolines.
    // See vk_dispatch.h for the rationale.
    #define LOAD(name) \
        vkd.name = (PFN_vk##name)vkd.GetInstanceProcAddr(instance, "vk" #name)

    // Instance / physical-device
    LOAD(DestroyInstance);
    LOAD(EnumeratePhysicalDevices);
    LOAD(GetPhysicalDeviceProperties);
    LOAD(GetPhysicalDeviceMemoryProperties);
    LOAD(GetPhysicalDeviceFeatures2);
    LOAD(GetPhysicalDeviceQueueFamilyProperties);
    LOAD(GetPhysicalDeviceFormatProperties);
    LOAD(GetPhysicalDeviceImageFormatProperties);
    LOAD(GetPhysicalDeviceSurfaceCapabilitiesKHR);
    LOAD(GetPhysicalDeviceSurfaceFormatsKHR);
    LOAD(GetPhysicalDeviceSurfacePresentModesKHR);
    LOAD(GetPhysicalDeviceSurfaceSupportKHR);
    LOAD(CreateAndroidSurfaceKHR);
    LOAD(DestroySurfaceKHR);
    LOAD(CreateDevice);
    LOAD(EnumerateDeviceExtensionProperties);
    LOAD(CreateDebugUtilsMessengerEXT);
    LOAD(DestroyDebugUtilsMessengerEXT);

    // Device
    LOAD(GetDeviceProcAddr);
    LOAD(DestroyDevice);
    LOAD(GetDeviceQueue);
    LOAD(DeviceWaitIdle);

    // Memory
    LOAD(AllocateMemory);
    LOAD(FreeMemory);
    LOAD(MapMemory);
    LOAD(UnmapMemory);
    LOAD(FlushMappedMemoryRanges);
    LOAD(InvalidateMappedMemoryRanges);
    LOAD(GetAndroidHardwareBufferPropertiesANDROID);

    // Buffer
    LOAD(CreateBuffer);
    LOAD(DestroyBuffer);
    LOAD(BindBufferMemory);
    LOAD(GetBufferMemoryRequirements);

    // Image / image view
    LOAD(CreateImage);
    LOAD(DestroyImage);
    LOAD(BindImageMemory);
    LOAD(GetImageMemoryRequirements);
    LOAD(CreateImageView);
    LOAD(DestroyImageView);

    // Sampler / YCbCr
    LOAD(CreateSampler);
    LOAD(DestroySampler);
    LOAD(CreateSamplerYcbcrConversion);
    LOAD(DestroySamplerYcbcrConversion);
    LOAD(CreateSamplerYcbcrConversionKHR);
    LOAD(DestroySamplerYcbcrConversionKHR);

    // Descriptors
    LOAD(CreateDescriptorSetLayout);
    LOAD(DestroyDescriptorSetLayout);
    LOAD(CreateDescriptorPool);
    LOAD(DestroyDescriptorPool);
    LOAD(AllocateDescriptorSets);
    LOAD(FreeDescriptorSets);
    LOAD(UpdateDescriptorSets);

    // Pipeline
    LOAD(CreatePipelineLayout);
    LOAD(DestroyPipelineLayout);
    LOAD(CreateGraphicsPipelines);
    LOAD(CreateComputePipelines);
    LOAD(DestroyPipeline);
    LOAD(CreateShaderModule);
    LOAD(DestroyShaderModule);

    // RenderPass / Framebuffer
    LOAD(CreateRenderPass);
    LOAD(DestroyRenderPass);
    LOAD(CreateFramebuffer);
    LOAD(DestroyFramebuffer);

    // Sync
    LOAD(CreateFence);
    LOAD(DestroyFence);
    LOAD(ResetFences);
    LOAD(WaitForFences);
    LOAD(CreateSemaphore);
    LOAD(DestroySemaphore);

    // Commands
    LOAD(CreateCommandPool);
    LOAD(DestroyCommandPool);
    LOAD(ResetCommandPool);
    LOAD(AllocateCommandBuffers);
    LOAD(FreeCommandBuffers);
    LOAD(BeginCommandBuffer);
    LOAD(EndCommandBuffer);
    LOAD(ResetCommandBuffer);
    LOAD(CmdBeginRenderPass);
    LOAD(CmdEndRenderPass);
    LOAD(CmdBindPipeline);
    LOAD(CmdBindDescriptorSets);
    LOAD(CmdBindVertexBuffers);
    LOAD(CmdPushConstants);
    LOAD(CmdSetViewport);
    LOAD(CmdSetScissor);
    LOAD(CmdDraw);
    LOAD(CmdPipelineBarrier);
    LOAD(CmdCopyBufferToImage);
    LOAD(CmdBlitImage);
    LOAD(CmdCopyImage);
    LOAD(CmdDispatch);

    // Queue
    LOAD(QueueSubmit);
    LOAD(QueueWaitIdle);
    LOAD(QueuePresentKHR);

    // Swapchain
    LOAD(CreateSwapchainKHR);
    LOAD(DestroySwapchainKHR);
    LOAD(GetSwapchainImagesKHR);
    LOAD(AcquireNextImageKHR);

    #undef LOAD

    if (!vkd.DestroyInstance || !vkd.EnumeratePhysicalDevices || !vkd.CreateDevice) {
        LOGE("vkd_load_instance: required core entry points missing");
        return false;
    }
    vkd_owner_instance = instance;
    return true;
}

bool vkd_bind(void* libvulkan_handle, VkInstance instance) {
    if (!libvulkan_handle || instance == VK_NULL_HANDLE) return false;
    if (vkd_owner_handle == libvulkan_handle && vkd_owner_instance == instance
        && vkd.CreateInstance) {
        return true;
    }
    if (!vkd_init(libvulkan_handle)) return false;
    return vkd_load_instance(instance);
}

bool vkd_bind_proc(PFN_vkGetInstanceProcAddr gipa, VkInstance instance) {
    if (!gipa || instance == VK_NULL_HANDLE) return false;
    if (vkd.GetInstanceProcAddr == gipa && vkd_owner_instance == instance) return true;

    memset(&vkd, 0, sizeof(vkd));
    vkd_owner_handle = NULL;
    vkd_owner_instance = VK_NULL_HANDLE;
    vkd.GetInstanceProcAddr = gipa;
    vkd.CreateInstance = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    vkd.EnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)
        gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
    vkd.EnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)
        gipa(VK_NULL_HANDLE, "vkEnumerateInstanceLayerProperties");
    return vkd_load_instance(instance);
}

void vkd_unload(void) {
    memset(&vkd, 0, sizeof(vkd));
    vkd_owner_handle = NULL;
    vkd_owner_instance = VK_NULL_HANDLE;
}
