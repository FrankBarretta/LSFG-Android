// SPDX-FileCopyrightText: Copyright 2026 qwertypower (DEVAR Entertainment LLC)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// DIS frame generation: a Vulkan compute realisation of Dense Inverse Search
// optical flow. The algorithm and its reference implementation come from
// OpenCV's DISOpticalFlow, which adopted Till Kroeger's original OF_DIS.
// See CREDITS.md for the full attribution.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "vk_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VKR_DIS_MAX_GENERATIONS 7u

typedef struct VkrDis VkrDis;

typedef struct VkrDisContentRect {
    int32_t x;
    int32_t y;
    uint32_t width;
    uint32_t height;
} VkrDisContentRect;

VkrDis* vkr_dis_create(VkDevice device, VkPhysicalDevice physical_device);
void vkr_dis_destroy(VkrDis* dis);

void vkr_dis_configure(VkrDis* dis, uint32_t flow_min_side, uint32_t target_fps,
                       float refresh_rate);

void vkr_dis_set_debug_flow(VkrDis* dis, bool debug_flow);

bool vkr_dis_needs_rebuild(const VkrDis* dis, uint32_t width, uint32_t height,
                           VkFormat format, VkrDisContentRect content);

bool vkr_dis_prepare(VkrDis* dis, uint32_t width, uint32_t height, VkFormat format,
                     VkrDisContentRect content);

uint32_t vkr_dis_plan(VkrDis* dis, uint32_t capacity, uint64_t source_frames);

void vkr_dis_process(VkrDis* dis, VkCommandBuffer cmd, VkImage source,
                     uint32_t width, uint32_t height, uint32_t generations);

// Submits `cmd` (ended by the callee, no semaphores), waits for it on the CPU, and returns a
// command buffer in the recording state for the rest of the frame - `cmd` itself reset and begun
// again is fine. VK_NULL_HANDLE on failure.
typedef VkCommandBuffer (*VkrDisFlushFn)(void* user, VkCommandBuffer cmd);

// As vkr_dis_process, and when the GLES driver offers GL_QCOM_motion_estimation, seeds the flow
// with a hardware estimate. That needs this frame's pixels mid-frame, so DIS calls `flush` once
// and records the rest into the buffer it returns; the caller continues with - and finally
// submits - the returned buffer. With flush NULL, or without the extension, this is exactly
// vkr_dis_process and returns `cmd`.
VkCommandBuffer vkr_dis_process_ex(VkrDis* dis, VkCommandBuffer cmd, VkImage source,
                                   uint32_t width, uint32_t height, uint32_t generations,
                                   VkrDisFlushFn flush, void* flush_user);

// Hardware motion hint on or off (default off, or debug.winnative.dis.hwme=1); takes effect at
// the next resource build.
void vkr_dis_set_hw_motion(VkrDis* dis, bool enabled);
bool vkr_dis_hw_motion_active(const VkrDis* dis);

void vkr_dis_generate_into(VkrDis* dis, VkCommandBuffer cmd, uint32_t generation,
                           uint32_t target_index, VkImage target_image,
                           VkImageView target_view, uint32_t width, uint32_t height,
                           VkImage base_image);

void vkr_dis_debug_into(VkrDis* dis, VkCommandBuffer cmd, VkImage target_image,
                        uint32_t width, uint32_t height);

void vkr_dis_forget_targets(VkrDis* dis);

void vkr_dis_reset(VkrDis* dis);

#ifdef __cplusplus
}
#endif
