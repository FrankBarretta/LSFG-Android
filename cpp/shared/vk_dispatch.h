// Function-pointer dispatch for the compositor's Vulkan calls.
//
// Required because adrenotools-loaded drivers live in an isolated linker namespace and do
// not share global symbols with the system loader — every call must resolve through the
// libvulkan handle chosen at dlopen time.
//
// Init order: vkd_init(handle) -> vkCreateInstance(...) -> vkd_load_instance(instance).

#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <stdbool.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VkDispatch {
    // Loader-level (resolved via dlsym + vkGetInstanceProcAddr(NULL, ...))
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_vkEnumerateInstanceLayerProperties EnumerateInstanceLayerProperties;

    // Instance / physical-device
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties;
    PFN_vkGetPhysicalDeviceImageFormatProperties GetPhysicalDeviceImageFormatProperties;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR GetPhysicalDeviceSurfaceCapabilitiesKHR;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR GetPhysicalDeviceSurfaceFormatsKHR;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR GetPhysicalDeviceSurfacePresentModesKHR;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR GetPhysicalDeviceSurfaceSupportKHR;
    PFN_vkCreateAndroidSurfaceKHR CreateAndroidSurfaceKHR;
    PFN_vkDestroySurfaceKHR DestroySurfaceKHR;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
    PFN_vkCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT;
    PFN_vkDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT;

    // Device
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkDeviceWaitIdle DeviceWaitIdle;

    // Memory
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID GetAndroidHardwareBufferPropertiesANDROID;

    // Buffer
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;

    // Image / image view
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;

    // Sampler / YCbCr
    PFN_vkCreateSampler CreateSampler;
    PFN_vkDestroySampler DestroySampler;
    PFN_vkCreateSamplerYcbcrConversion CreateSamplerYcbcrConversion;
    PFN_vkDestroySamplerYcbcrConversion DestroySamplerYcbcrConversion;
    PFN_vkCreateSamplerYcbcrConversionKHR CreateSamplerYcbcrConversionKHR;
    PFN_vkDestroySamplerYcbcrConversionKHR DestroySamplerYcbcrConversionKHR;

    // Descriptors
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkFreeDescriptorSets FreeDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;

    // Pipeline
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;

    // RenderPass / Framebuffer
    PFN_vkCreateRenderPass CreateRenderPass;
    PFN_vkDestroyRenderPass DestroyRenderPass;
    PFN_vkCreateFramebuffer CreateFramebuffer;
    PFN_vkDestroyFramebuffer DestroyFramebuffer;

    // Sync
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkResetFences ResetFences;
    PFN_vkWaitForFences WaitForFences;
    PFN_vkCreateSemaphore CreateSemaphore;
    PFN_vkDestroySemaphore DestroySemaphore;

    // Commands
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkResetCommandPool ResetCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkFreeCommandBuffers FreeCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkResetCommandBuffer ResetCommandBuffer;
    PFN_vkCmdBeginRenderPass CmdBeginRenderPass;
    PFN_vkCmdEndRenderPass CmdEndRenderPass;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdSetViewport CmdSetViewport;
    PFN_vkCmdSetScissor CmdSetScissor;
    PFN_vkCmdDraw CmdDraw;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
    PFN_vkCmdBlitImage CmdBlitImage;
    PFN_vkCmdCopyImage CmdCopyImage;
    PFN_vkCmdDispatch CmdDispatch;

    // Queue
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkQueueWaitIdle QueueWaitIdle;
    PFN_vkQueuePresentKHR QueuePresentKHR;

    // Swapchain
    PFN_vkCreateSwapchainKHR CreateSwapchainKHR;
    PFN_vkDestroySwapchainKHR DestroySwapchainKHR;
    PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR;
    PFN_vkAcquireNextImageKHR AcquireNextImageKHR;
} VkDispatch;

extern VkDispatch vkd;

bool vkd_init(void* libvulkan_handle);

// Loads device-level pointers via vkGetInstanceProcAddr too — the loader trampolines, which
// costs a few ns per call but avoids partitioning instance vs. device scope.
bool vkd_load_instance(VkInstance instance);

// Must be called before dlclose so stale-pointer crashes fault on NULL.
void vkd_unload(void);

bool vkd_bind(void* libvulkan_handle, VkInstance instance);

// For callers that already hold the driver's vkGetInstanceProcAddr rather than a dlopen handle
// (the Wayland compositor resolves Turnip through adrenotools).
bool vkd_bind_proc(PFN_vkGetInstanceProcAddr gipa, VkInstance instance);

#ifdef __cplusplus
}
#endif

// Redirect bare `vkFoo` names to the dispatch table.

#define vkGetInstanceProcAddr vkd.GetInstanceProcAddr
#define vkCreateInstance vkd.CreateInstance
#define vkEnumerateInstanceExtensionProperties vkd.EnumerateInstanceExtensionProperties
#define vkEnumerateInstanceLayerProperties vkd.EnumerateInstanceLayerProperties

#define vkDestroyInstance vkd.DestroyInstance
#define vkEnumeratePhysicalDevices vkd.EnumeratePhysicalDevices
#define vkGetPhysicalDeviceProperties vkd.GetPhysicalDeviceProperties
#define vkGetPhysicalDeviceMemoryProperties vkd.GetPhysicalDeviceMemoryProperties
#define vkGetPhysicalDeviceFeatures2 vkd.GetPhysicalDeviceFeatures2
#define vkGetPhysicalDeviceQueueFamilyProperties vkd.GetPhysicalDeviceQueueFamilyProperties
#define vkGetPhysicalDeviceFormatProperties vkd.GetPhysicalDeviceFormatProperties
#define vkGetPhysicalDeviceImageFormatProperties vkd.GetPhysicalDeviceImageFormatProperties
#define vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkd.GetPhysicalDeviceSurfaceCapabilitiesKHR
#define vkGetPhysicalDeviceSurfaceFormatsKHR vkd.GetPhysicalDeviceSurfaceFormatsKHR
#define vkGetPhysicalDeviceSurfacePresentModesKHR vkd.GetPhysicalDeviceSurfacePresentModesKHR
#define vkGetPhysicalDeviceSurfaceSupportKHR vkd.GetPhysicalDeviceSurfaceSupportKHR
#define vkCreateAndroidSurfaceKHR vkd.CreateAndroidSurfaceKHR
#define vkDestroySurfaceKHR vkd.DestroySurfaceKHR
#define vkCreateDevice vkd.CreateDevice
#define vkEnumerateDeviceExtensionProperties vkd.EnumerateDeviceExtensionProperties
#define vkCreateDebugUtilsMessengerEXT vkd.CreateDebugUtilsMessengerEXT
#define vkDestroyDebugUtilsMessengerEXT vkd.DestroyDebugUtilsMessengerEXT

#define vkGetDeviceProcAddr vkd.GetDeviceProcAddr
#define vkDestroyDevice vkd.DestroyDevice
#define vkGetDeviceQueue vkd.GetDeviceQueue
#define vkDeviceWaitIdle vkd.DeviceWaitIdle

#define vkAllocateMemory vkd.AllocateMemory
#define vkFreeMemory vkd.FreeMemory
#define vkMapMemory vkd.MapMemory
#define vkUnmapMemory vkd.UnmapMemory
#define vkFlushMappedMemoryRanges vkd.FlushMappedMemoryRanges
#define vkInvalidateMappedMemoryRanges vkd.InvalidateMappedMemoryRanges
#define vkGetAndroidHardwareBufferPropertiesANDROID vkd.GetAndroidHardwareBufferPropertiesANDROID

#define vkCreateBuffer vkd.CreateBuffer
#define vkDestroyBuffer vkd.DestroyBuffer
#define vkBindBufferMemory vkd.BindBufferMemory
#define vkGetBufferMemoryRequirements vkd.GetBufferMemoryRequirements

#define vkCreateImage vkd.CreateImage
#define vkDestroyImage vkd.DestroyImage
#define vkBindImageMemory vkd.BindImageMemory
#define vkGetImageMemoryRequirements vkd.GetImageMemoryRequirements
#define vkCreateImageView vkd.CreateImageView
#define vkDestroyImageView vkd.DestroyImageView

#define vkCreateSampler vkd.CreateSampler
#define vkDestroySampler vkd.DestroySampler
#define vkCreateSamplerYcbcrConversion vkd.CreateSamplerYcbcrConversion
#define vkDestroySamplerYcbcrConversion vkd.DestroySamplerYcbcrConversion
#define vkCreateSamplerYcbcrConversionKHR vkd.CreateSamplerYcbcrConversionKHR
#define vkDestroySamplerYcbcrConversionKHR vkd.DestroySamplerYcbcrConversionKHR

#define vkCreateDescriptorSetLayout vkd.CreateDescriptorSetLayout
#define vkDestroyDescriptorSetLayout vkd.DestroyDescriptorSetLayout
#define vkCreateDescriptorPool vkd.CreateDescriptorPool
#define vkDestroyDescriptorPool vkd.DestroyDescriptorPool
#define vkAllocateDescriptorSets vkd.AllocateDescriptorSets
#define vkFreeDescriptorSets vkd.FreeDescriptorSets
#define vkUpdateDescriptorSets vkd.UpdateDescriptorSets

#define vkCreatePipelineLayout vkd.CreatePipelineLayout
#define vkDestroyPipelineLayout vkd.DestroyPipelineLayout
#define vkCreateGraphicsPipelines vkd.CreateGraphicsPipelines
#define vkDestroyPipeline vkd.DestroyPipeline
#define vkCreateShaderModule vkd.CreateShaderModule
#define vkDestroyShaderModule vkd.DestroyShaderModule

#define vkCreateRenderPass vkd.CreateRenderPass
#define vkDestroyRenderPass vkd.DestroyRenderPass
#define vkCreateFramebuffer vkd.CreateFramebuffer
#define vkDestroyFramebuffer vkd.DestroyFramebuffer

#define vkCreateFence vkd.CreateFence
#define vkDestroyFence vkd.DestroyFence
#define vkResetFences vkd.ResetFences
#define vkWaitForFences vkd.WaitForFences
#define vkCreateSemaphore vkd.CreateSemaphore
#define vkDestroySemaphore vkd.DestroySemaphore

#define vkCreateCommandPool vkd.CreateCommandPool
#define vkDestroyCommandPool vkd.DestroyCommandPool
#define vkResetCommandPool vkd.ResetCommandPool
#define vkAllocateCommandBuffers vkd.AllocateCommandBuffers
#define vkFreeCommandBuffers vkd.FreeCommandBuffers
#define vkBeginCommandBuffer vkd.BeginCommandBuffer
#define vkEndCommandBuffer vkd.EndCommandBuffer
#define vkResetCommandBuffer vkd.ResetCommandBuffer
#define vkCmdBeginRenderPass vkd.CmdBeginRenderPass
#define vkCmdEndRenderPass vkd.CmdEndRenderPass
#define vkCmdBindPipeline vkd.CmdBindPipeline
#define vkCmdBindDescriptorSets vkd.CmdBindDescriptorSets
#define vkCmdBindVertexBuffers vkd.CmdBindVertexBuffers
#define vkCmdPushConstants vkd.CmdPushConstants
#define vkCmdSetViewport vkd.CmdSetViewport
#define vkCmdSetScissor vkd.CmdSetScissor
#define vkCmdDraw vkd.CmdDraw
#define vkCmdPipelineBarrier vkd.CmdPipelineBarrier
#define vkCmdCopyBufferToImage vkd.CmdCopyBufferToImage
#define vkCmdBlitImage vkd.CmdBlitImage

#define vkQueueSubmit vkd.QueueSubmit
#define vkQueueWaitIdle vkd.QueueWaitIdle
#define vkQueuePresentKHR vkd.QueuePresentKHR

#define vkCreateSwapchainKHR vkd.CreateSwapchainKHR
#define vkDestroySwapchainKHR vkd.DestroySwapchainKHR
#define vkGetSwapchainImagesKHR vkd.GetSwapchainImagesKHR
#define vkAcquireNextImageKHR vkd.AcquireNextImageKHR
