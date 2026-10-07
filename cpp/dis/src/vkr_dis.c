// SPDX-FileCopyrightText: Copyright 2026 qwertypower (DEVAR Entertainment LLC)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// DIS frame generation: a Vulkan compute realisation of Dense Inverse Search
// optical flow. The algorithm and its reference implementation come from
// OpenCV's DISOpticalFlow, which adopted Till Kroeger's original OF_DIS.
// See CREDITS.md for the full attribution.

#include "vkr_dis.h"

#include "dis_qcom_me.h"
#include "vk_dispatch.h"
#include "shaders/dis_luma_r16_comp.spv.h"
#include "shaders/dis_luma_r32_comp.spv.h"
#include "shaders/dis_gradient_comp.spv.h"
#include "shaders/dis_inverse_search_comp.spv.h"
#include "shaders/dis_propagate_comp.spv.h"
#include "shaders/dis_densify_comp.spv.h"
#include "shaders/dis_interpolate_comp.spv.h"
#include "shaders/dis_hist_comp.spv.h"
#include "shaders/dis_side_comp.spv.h"
#include "shaders/dis_flow_pack_comp.spv.h"
#include "shaders/dis_me_luma_comp.spv.h"
#include "shaders/dis_vr_prep_comp.spv.h"
#include "shaders/dis_vr_d1_comp.spv.h"
#include "shaders/dis_vr_d2_comp.spv.h"
#include "shaders/dis_vr_w_comp.spv.h"
#include "shaders/dis_vr_coef_comp.spv.h"
#include "shaders/dis_vr_sor_comp.spv.h"
#include "shaders/dis_vr_add_comp.spv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <android/log.h>
#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#define DIS_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "VkrDis", __VA_ARGS__)
#define DIS_LOGW(...) __android_log_print(ANDROID_LOG_WARN, "VkrDis", __VA_ARGS__)

#define DIS_LOCAL_SIZE 8u
#define DIS_PATCH_STRIDE 3u
#define DIS_MIN_EXTENT 16u

#define DIS_DEFAULT_FLOW_MIN_SIDE 180u
#define DIS_FLOW_MIN_SIDE_FLOOR 64u
#define DIS_FLOW_MIN_SIDE_CEIL 1080u
#define DIS_MAX_LEVELS 8u
#define DIS_MAX_DESCRIPTOR_WRITES 256u

#define DIS_SLOTS 3u

// How the hardware motion field enters the flow: as a second starting candidate for the search on
// its level (0), or as that level's result outright, skipping the search above it (1).
#ifndef DIS_ME_PRIMARY
#define DIS_ME_PRIMARY 0
#endif

#define DIS_PROP_STEPS_MAX 4u

#define DIS_SRC_SMOOTHING 0.08f
#define DIS_SRC_STALE_NS 500000000ull
#define DIS_MIN_RATE_SAMPLES 12u

#define DIS_PLAN_LOG_NS 5000000000ull

#define DIS_RATIO_SLACK 0.12f

#define DIS_MIN_GEN_RATIO 1.45f

#define DIS_RATIO_HYST 0.10f

#define DIS_VR_ALPHA 20.0f
#define DIS_VR_DELTA 5.0f
#define DIS_VR_GAMMA 10.0f
#define DIS_VR_OMEGA 1.6f
#define DIS_VR_ZETA 0.1f
#define DIS_VR_EPS 0.001f

// Fewest SOR sweeps a level that still runs the solver gets.
#define DIS_VR_SOR_FLOOR 2u

// The variational refinement runs on the finest levels only. Above the third,
// the refined flow is upsampled into the next search anyway, so refining the
// coarse levels bought very little while their prep/add dispatches and solver
// sweeps dominated the pass count. Levels without refinement skip the VR stage
// entirely and the search reads the densified flow instead.
//
// Restored to every level: fast motion showed the coarse levels were carrying
// more of the flow than the dispatch saving was worth, and the refinement has
// to be there for large displacements to come out of the pyramid cleanly.
#define DIS_VR_LEVELS 8u


#define DIS_SET_SAMPLERS 5u
#define DIS_SET_STORAGE 1u
#define DIS_SHARED_SETS_PER_LEVEL 6u
#define DIS_VR_SHARED_SETS 7u
#define DIS_VR_SAMPLER_BINDINGS 8u
#define DIS_VR_STORAGE_BINDINGS 2u
#define DIS_VR_FIRST_STORAGE 8u

typedef struct {
    VkImage image;
    VkDeviceMemory memory;
    VkExtent2D extent;
    VkFormat format;
    uint32_t mip_levels;
} DisImage;

typedef struct {
    VkPipeline pipeline;
} DisPass;

typedef struct {
    VkWriteDescriptorSet w[DIS_MAX_DESCRIPTOR_WRITES];
    VkDescriptorImageInfo img[DIS_MAX_DESCRIPTOR_WRITES];
    uint32_t count;
} DisBatch;

struct VkrDis {
    VkDevice device;
    VkPhysicalDevice physical_device;
    VkPhysicalDeviceMemoryProperties mem_props;

    uint32_t flow_min_side;
    uint32_t target_fps;
    float refresh_rate;

    VkExtent2D built_extent;
    VkExtent2D built_full_extent;
    VkrDisContentRect content;
    uint32_t built_min_side;
    VkFormat built_format;
    uint32_t levels;
    bool built;
    bool unavailable;
    bool layouts_primed;
    bool formats_audited;
    bool manual_flow_filter;
    bool debug_flow;

    DisImage color[DIS_SLOTS];
    DisImage flow_color[DIS_SLOTS];
    DisImage grad;
    DisImage flow_luma[DIS_SLOTS];
    VkFormat luma_format;
    DisImage flow_sparse[DIS_MAX_LEVELS];
    DisImage flow_sparse_b[DIS_MAX_LEVELS];
    DisImage flow_dense;
    DisImage interp_out;

    DisImage vr_prep;
    DisImage vr_d1;
    DisImage vr_d2;
    DisImage vr_A;
    DisImage vr_B;
    DisImage vr_wt;
    DisImage vr_dw[2];
    DisImage flow_refined;
    DisImage hist[2];
    DisImage side;
    DisImage flow_out;
    DisImage me_field;

    VkImageView view_color[DIS_SLOTS];
    VkImageView view_flow_color[DIS_SLOTS][DIS_MAX_LEVELS];
    VkImageView view_flow_luma[DIS_SLOTS][DIS_MAX_LEVELS];
    VkImageView view_grad[DIS_MAX_LEVELS];
    VkImageView view_sparse[DIS_MAX_LEVELS];
    VkImageView view_sparse_b[DIS_MAX_LEVELS];
    VkImageView view_dense[DIS_MAX_LEVELS];
    VkImageView view_interp_out;
    VkImageView view_vr_prep[DIS_MAX_LEVELS];
    VkImageView view_vr_d1[DIS_MAX_LEVELS];
    VkImageView view_vr_d2[DIS_MAX_LEVELS];
    VkImageView view_vr_A[DIS_MAX_LEVELS];
    VkImageView view_vr_B[DIS_MAX_LEVELS];
    VkImageView view_vr_wt[DIS_MAX_LEVELS];
    VkImageView view_vr_dw[2][DIS_MAX_LEVELS];
    VkImageView view_flow_refined[DIS_MAX_LEVELS];
    VkImageView view_hist[2];
    VkImageView view_side;
    VkImageView view_flow_out;
    VkImageView view_me_field;

    VkSampler sampler;

    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipeline_layout;
    VkDescriptorPool pool;
    VkDescriptorSet luma_sets[DIS_SLOTS][DIS_MAX_LEVELS];
    VkDescriptorSet grad_sets[DIS_SLOTS][DIS_MAX_LEVELS];
    VkDescriptorSet inverse_sets[DIS_SLOTS][DIS_MAX_LEVELS];
    VkDescriptorSet densify_sets[DIS_SLOTS][DIS_MAX_LEVELS];
    VkDescriptorSet prop_ab_sets[DIS_SLOTS][DIS_MAX_LEVELS];
    VkDescriptorSet prop_ba_sets[DIS_SLOTS][DIS_MAX_LEVELS];
    VkDescriptorSet interp_sets[DIS_SLOTS][2];
    VkDescriptorSet hist_sets[DIS_SLOTS][2];
    VkDescriptorSet side_sets[DIS_SLOTS];
    VkDescriptorSet pack_set;

    VkDescriptorSetLayout vr_set_layout;
    VkPipelineLayout vr_pipeline_layout;
    VkDescriptorSet vr_prep_sets[DIS_SLOTS][DIS_MAX_LEVELS];
    VkDescriptorSet vr_d1_set[DIS_MAX_LEVELS];
    VkDescriptorSet vr_d2_set[DIS_MAX_LEVELS];
    VkDescriptorSet vr_w_set[DIS_MAX_LEVELS];
    VkDescriptorSet vr_coef_set[DIS_MAX_LEVELS];
    VkDescriptorSet vr_sor_ab_set[DIS_MAX_LEVELS];
    VkDescriptorSet vr_sor_ba_set[DIS_MAX_LEVELS];
    VkDescriptorSet vr_add_set[DIS_MAX_LEVELS];

    DisPass pass_luma;
    DisPass pass_gradient;
    DisPass pass_inverse;
    DisPass pass_propagate;
    DisPass pass_densify;
    DisPass pass_interp;
    DisPass pass_hist;
    DisPass pass_side;
    DisPass pass_pack;
    DisPass pass_me_luma;
    DisPass pass_vr_prep;
    DisPass pass_vr_d1;
    DisPass pass_vr_d2;
    DisPass pass_vr_w;
    DisPass pass_vr_coef;
    DisPass pass_vr_sor;
    DisPass pass_vr_add;

    uint64_t frame_count;
    int prev_idx;
    int next_idx;
    uint32_t active_slot;
    uint32_t last_generations;

    uint64_t src_sample_ns;
    uint64_t src_last_frames;
    float src_frame_accum;
    float src_time_accum;
    float src_interval;
    uint32_t src_samples;

    float smoothed_desired;

    int planned_gen;
    uint32_t gen_high_streak;
    uint32_t gen_low_streak;

    uint64_t plan_log_ns;
    int plan_log_gen;

    uint32_t hist_parity;
    bool hist_valid;

    // Hardware motion hint (GL_QCOM_motion_estimation). me is NULL whenever the hint is off:
    // disabled, unsupported, or the pyramid too shallow for the level it seeds.
    bool hw_motion;
    DisQcomMe* me;
    uint32_t me_level;
    uint32_t me_w, me_h;
    uint32_t me_field_w, me_field_h;
    VkBuffer me_luma_buf;
    VkDeviceMemory me_luma_mem;
    void* me_luma_map;
    bool me_luma_coherent;
    VkBuffer me_field_buf;
    VkDeviceMemory me_field_mem;
    void* me_field_map;
    bool me_field_coherent;
    float* me_xy;
    int hint_level;
    bool me_primary_frame;
    VkDescriptorSetLayout me_set_layout;
    VkPipelineLayout me_pipeline_layout;
    VkDescriptorPool me_pool;
    VkDescriptorSet me_sets[DIS_SLOTS];
    uint64_t me_hinted;
    uint64_t me_pairs;
};

typedef struct {
    float lesser;
    float upper;
    float normVal;
} DisGradientPC;

typedef struct {
    int level;
    int coarseLevel;
    int hintLevel;
} DisInversePC;

typedef struct {
    int dist;
} DisPropPC;

typedef struct {
    float t;
    int debugMode;
} DisInterpPC;

typedef struct {
    float alpha2;
    float eps2;
} DisVrWPC;

typedef struct {
    float delta2;
    float gamma2;
    float zeta2;
    float eps2;
} DisVrCoefPC;

typedef struct {
    float omega;
    int parity;
} DisVrSorPC;

static void dis_destroy_me(VkrDis* d);

static uint64_t dis_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint32_t dis_find_memory_type(VkrDis* d, uint32_t bits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < d->mem_props.memoryTypeCount; i++) {
        if ((bits & (1u << i)) &&
            (d->mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

// Per-stage GPU timestamps for the bench (tools/dis-bench); compiled out of the app.
#ifdef DIS_PROFILE
#define DIS_PROF_MAX 64u
static struct {
    VkQueryPool pool;
    PFN_vkCmdWriteTimestamp write;
    PFN_vkCmdResetQueryPool reset;
    PFN_vkGetQueryPoolResults get;
    float period_ns;
    const char* label[DIS_PROF_MAX];
    uint32_t n;
    const char* acc_label[DIS_PROF_MAX];
    double acc_ms[DIS_PROF_MAX];
    uint32_t acc_n;
    uint64_t frames;
} g_prof;

static void dis_prof_mark(VkCommandBuffer cmd, const char* label) {
    if (!g_prof.pool || g_prof.n >= DIS_PROF_MAX) return;
    g_prof.write(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_prof.pool, g_prof.n);
    g_prof.label[g_prof.n++] = label;
}

static void dis_prof_begin(VkrDis* d, VkCommandBuffer cmd) {
    if (!g_prof.pool) {
        g_prof.write = (PFN_vkCmdWriteTimestamp)vkd.GetDeviceProcAddr(d->device, "vkCmdWriteTimestamp");
        g_prof.reset = (PFN_vkCmdResetQueryPool)vkd.GetDeviceProcAddr(d->device, "vkCmdResetQueryPool");
        g_prof.get = (PFN_vkGetQueryPoolResults)vkd.GetDeviceProcAddr(d->device, "vkGetQueryPoolResults");
        PFN_vkCreateQueryPool create =
            (PFN_vkCreateQueryPool)vkd.GetDeviceProcAddr(d->device, "vkCreateQueryPool");
        VkPhysicalDeviceProperties props;
        vkd.GetPhysicalDeviceProperties(d->physical_device, &props);
        g_prof.period_ns = props.limits.timestampPeriod;
        VkQueryPoolCreateInfo qi;
        memset(&qi, 0, sizeof(qi));
        qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = DIS_PROF_MAX;
        if (!create || create(d->device, &qi, NULL, &g_prof.pool) != VK_SUCCESS) return;
    }
    g_prof.reset(cmd, g_prof.pool, 0, DIS_PROF_MAX);
    g_prof.n = 0;
    dis_prof_mark(cmd, "start");
}

// Call once the frame's last submit has completed.
void vkr_dis_profile_collect(VkrDis* d) {
    if (!g_prof.pool || g_prof.n < 2) return;
    uint64_t ts[DIS_PROF_MAX];
    if (g_prof.get(d->device, g_prof.pool, 0, g_prof.n, sizeof(ts), ts, sizeof(uint64_t),
                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) return;
    for (uint32_t i = 1; i < g_prof.n; i++) {
        uint32_t k = 0;
        while (k < g_prof.acc_n && strcmp(g_prof.acc_label[k], g_prof.label[i]) != 0) k++;
        if (k == g_prof.acc_n) {
            if (g_prof.acc_n >= DIS_PROF_MAX) continue;
            g_prof.acc_label[g_prof.acc_n] = g_prof.label[i];
            g_prof.acc_ms[g_prof.acc_n++] = 0.0;
        }
        g_prof.acc_ms[k] += (double)(ts[i] - ts[i - 1]) * g_prof.period_ns * 1e-6;
    }
    g_prof.frames++;
    g_prof.n = 0;
}

void vkr_dis_profile_report(VkrDis* d) {
    (void)d;
    if (!g_prof.frames) return;
    double total = 0;
    for (uint32_t k = 0; k < g_prof.acc_n; k++) total += g_prof.acc_ms[k];
    printf("PROFILE over %llu frames (ms per real frame):\n", (unsigned long long)g_prof.frames);
    for (uint32_t k = 0; k < g_prof.acc_n; k++) {
        printf("  %-18s %7.3f\n", g_prof.acc_label[k], g_prof.acc_ms[k] / (double)g_prof.frames);
    }
    printf("  %-18s %7.3f\n", "total", total / (double)g_prof.frames);
    memset(g_prof.acc_ms, 0, sizeof(g_prof.acc_ms));
    g_prof.acc_n = 0;
    g_prof.frames = 0;
}
#define DIS_PROF_BEGIN(d, cmd) dis_prof_begin((d), (cmd))
#define DIS_PROF(cmd, label) dis_prof_mark((cmd), (label))
#else
#define DIS_PROF_BEGIN(d, cmd) ((void)0)
#define DIS_PROF(cmd, label) ((void)0)
#endif

static void dis_compute_barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkd.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void dis_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
                        VkAccessFlags src_access, VkAccessFlags dst_access) {
    VkImageMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    b.subresourceRange.layerCount = 1;
    vkd.CmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &b);
}

static uint32_t dis_collect_images(VkrDis* d, DisImage** out, uint32_t cap) {
    uint32_t n = 0;
    #define DIS_PUSH(img) do { if (n < cap) out[n++] = (img); } while (0)
    for (uint32_t s = 0; s < DIS_SLOTS; s++) {
        DIS_PUSH(&d->color[s]);
        DIS_PUSH(&d->flow_color[s]);
        DIS_PUSH(&d->flow_luma[s]);
    }
    DIS_PUSH(&d->grad);
    for (uint32_t l = 0; l < DIS_MAX_LEVELS; l++) {
        DIS_PUSH(&d->flow_sparse[l]);
        DIS_PUSH(&d->flow_sparse_b[l]);
    }
    DIS_PUSH(&d->flow_dense);
    DIS_PUSH(&d->interp_out);
    DIS_PUSH(&d->vr_prep);
    DIS_PUSH(&d->vr_d1);
    DIS_PUSH(&d->vr_d2);
    DIS_PUSH(&d->vr_A);
    DIS_PUSH(&d->vr_B);
    DIS_PUSH(&d->vr_wt);
    DIS_PUSH(&d->vr_dw[0]);
    DIS_PUSH(&d->vr_dw[1]);
    DIS_PUSH(&d->flow_refined);
    DIS_PUSH(&d->hist[0]);
    DIS_PUSH(&d->hist[1]);
    DIS_PUSH(&d->side);
    DIS_PUSH(&d->flow_out);
    DIS_PUSH(&d->me_field);
    #undef DIS_PUSH
    return n;
}

#define DIS_MAX_OWNED_IMAGES 48u

static void dis_prime_layouts(VkrDis* d, VkCommandBuffer cmd) {
    if (d->layouts_primed) return;

    DisImage* imgs[DIS_MAX_OWNED_IMAGES];
    const uint32_t n = dis_collect_images(d, imgs, DIS_MAX_OWNED_IMAGES);

    VkImageMemoryBarrier bars[DIS_MAX_OWNED_IMAGES];
    uint32_t count = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!imgs[i]->image) continue;
        VkImageMemoryBarrier* b = &bars[count++];
        memset(b, 0, sizeof(*b));
        b->sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b->srcAccessMask = 0;
        b->dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b->oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b->newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b->image = imgs[i]->image;
        b->subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b->subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        b->subresourceRange.layerCount = 1;
    }
    if (count == 0) return;

    vkd.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 0, NULL, 0, NULL, count, bars);
    d->layouts_primed = true;
}

static VkFormat dis_pick_luma_format(VkrDis* d) {
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    const VkFormat candidates[2] = {VK_FORMAT_R16_SFLOAT, VK_FORMAT_R32_SFLOAT};
    for (uint32_t i = 0; i < 2; i++) {
        VkFormatProperties fp;
        memset(&fp, 0, sizeof(fp));
        vkd.GetPhysicalDeviceFormatProperties(d->physical_device, candidates[i], &fp);
        if ((fp.optimalTilingFeatures & need) == need) {
            if (i != 0) DIS_LOGI("R16F unusable for the luminance plane; using R32F");
            return candidates[i];
        }
    }
    return VK_FORMAT_UNDEFINED;
}

static bool dis_create_image(VkrDis* d, DisImage* out, uint32_t w, uint32_t h, VkFormat format,
                             uint32_t mip_levels, VkImageUsageFlags usage) {
    memset(out, 0, sizeof(*out));
    out->extent.width = w;
    out->extent.height = h;
    out->format = format;
    out->mip_levels = mip_levels;

    VkImageCreateInfo ic;
    memset(&ic, 0, sizeof(ic));
    ic.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = format;
    ic.extent.width = w;
    ic.extent.height = h;
    ic.extent.depth = 1;
    ic.mipLevels = mip_levels;
    ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = usage;
    ic.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkd.CreateImage(d->device, &ic, NULL, &out->image) != VK_SUCCESS) return false;

    VkMemoryRequirements mr;
    vkd.GetImageMemoryRequirements(d->device, out->image, &mr);
    uint32_t type = dis_find_memory_type(d, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) return false;
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    if (vkd.AllocateMemory(d->device, &ai, NULL, &out->memory) != VK_SUCCESS) return false;
    vkd.BindImageMemory(d->device, out->image, out->memory, 0);
    return true;
}

static void dis_destroy_image(VkrDis* d, DisImage* img) {
    if (img->image) vkd.DestroyImage(d->device, img->image, NULL);
    if (img->memory) vkd.FreeMemory(d->device, img->memory, NULL);
    memset(img, 0, sizeof(*img));
}

static bool dis_create_view(VkrDis* d, VkImage image, VkFormat format, uint32_t base_level,
                            uint32_t level_count, VkImageView* out) {
    VkImageViewCreateInfo vi;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.baseMipLevel = base_level;
    vi.subresourceRange.levelCount = level_count;
    vi.subresourceRange.layerCount = 1;
    return vkd.CreateImageView(d->device, &vi, NULL, out) == VK_SUCCESS;
}

static void dis_destroy_view(VkrDis* d, VkImageView* view) {
    if (*view) vkd.DestroyImageView(d->device, *view, NULL);
    *view = VK_NULL_HANDLE;
}

static VkPipeline dis_create_compute_pipeline_with_layout(VkrDis* d, const uint32_t* code,
                                                           size_t code_size, VkPipelineLayout layout,
                                                           const VkSpecializationInfo* spec) {
    VkShaderModuleCreateInfo smi;
    memset(&smi, 0, sizeof(smi));
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smi.codeSize = code_size;
    smi.pCode = code;
    VkShaderModule sm;
    if (vkd.CreateShaderModule(d->device, &smi, NULL, &sm) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stage;
    memset(&stage, 0, sizeof(stage));
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = sm;
    stage.pName = "main";
    stage.pSpecializationInfo = spec;

    VkComputePipelineCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage = stage;
    pci.layout = layout;

    VkPipeline pipeline;
    VkResult res = vkd.CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &pci, NULL, &pipeline);
    vkd.DestroyShaderModule(d->device, sm, NULL);
    return res == VK_SUCCESS ? pipeline : VK_NULL_HANDLE;
}

static VkPipeline dis_create_compute_pipeline(VkrDis* d, const uint32_t* code, size_t code_size) {
    return dis_create_compute_pipeline_with_layout(d, code, code_size, d->pipeline_layout, NULL);
}

static bool dis_create_pipelines(VkrDis* d) {
    VkDescriptorSetLayoutBinding bindings[6];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 5; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo li;
    memset(&li, 0, sizeof(li));
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 6;
    li.pBindings = bindings;
    if (vkd.CreateDescriptorSetLayout(d->device, &li, NULL, &d->set_layout) != VK_SUCCESS) {
        return false;
    }

    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 32;

    VkPipelineLayoutCreateInfo pli;
    memset(&pli, 0, sizeof(pli));
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &d->set_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkd.CreatePipelineLayout(d->device, &pli, NULL, &d->pipeline_layout) != VK_SUCCESS) {
        return false;
    }

    const uint32_t shared_sets = DIS_SLOTS * DIS_MAX_LEVELS * DIS_SHARED_SETS_PER_LEVEL
                               + DIS_SLOTS * 2u   // interpolation sets, one per history direction
                               + DIS_SLOTS        // side-map sets
                               + 1u;              // flow pack set
    // VR sets exist only for the levels the refinement actually runs on.
    const uint32_t vr_sets = (DIS_SLOTS
                           + DIS_VR_SHARED_SETS) * DIS_VR_LEVELS;
    const uint32_t hist_sets = DIS_SLOTS * 2u;
    const uint32_t vr_layout_sets = vr_sets + hist_sets;
    const uint32_t total_sets = shared_sets + vr_layout_sets;

    // The history sets are allocated from vr_set_layout, so they consume the full
    // VR footprint (8 samplers + 2 storage) per set, not just the bindings they
    // write. Counting them short here fails vkAllocateDescriptorSets with
    // VK_ERROR_OUT_OF_POOL_MEMORY, which disables DIS entirely.
    VkDescriptorPoolSize sizes[2];
    memset(sizes, 0, sizeof(sizes));
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = shared_sets * DIS_SET_SAMPLERS
                             + vr_layout_sets * DIS_VR_SAMPLER_BINDINGS;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = shared_sets * DIS_SET_STORAGE
                             + vr_layout_sets * DIS_VR_STORAGE_BINDINGS;
    VkDescriptorPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = total_sets;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = sizes;
    if (vkd.CreateDescriptorPool(d->device, &pci, NULL, &d->pool) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetLayoutBinding vr_bindings[DIS_VR_SAMPLER_BINDINGS + DIS_VR_STORAGE_BINDINGS];
    memset(vr_bindings, 0, sizeof(vr_bindings));
    for (uint32_t i = 0; i < DIS_VR_SAMPLER_BINDINGS; i++) {
        vr_bindings[i].binding = i;
        vr_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        vr_bindings[i].descriptorCount = 1;
        vr_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    for (uint32_t i = 0; i < DIS_VR_STORAGE_BINDINGS; i++) {
        vr_bindings[DIS_VR_SAMPLER_BINDINGS + i].binding = DIS_VR_FIRST_STORAGE + i;
        vr_bindings[DIS_VR_SAMPLER_BINDINGS + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        vr_bindings[DIS_VR_SAMPLER_BINDINGS + i].descriptorCount = 1;
        vr_bindings[DIS_VR_SAMPLER_BINDINGS + i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo vr_li;
    memset(&vr_li, 0, sizeof(vr_li));
    vr_li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    vr_li.bindingCount = DIS_VR_SAMPLER_BINDINGS + DIS_VR_STORAGE_BINDINGS;
    vr_li.pBindings = vr_bindings;
    if (vkd.CreateDescriptorSetLayout(d->device, &vr_li, NULL, &d->vr_set_layout) != VK_SUCCESS) {
        return false;
    }

    VkPushConstantRange vr_pcr;
    memset(&vr_pcr, 0, sizeof(vr_pcr));
    vr_pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    vr_pcr.offset = 0;
    vr_pcr.size = 32;

    VkPipelineLayoutCreateInfo vr_pli;
    memset(&vr_pli, 0, sizeof(vr_pli));
    vr_pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vr_pli.setLayoutCount = 1;
    vr_pli.pSetLayouts = &d->vr_set_layout;
    vr_pli.pushConstantRangeCount = 1;
    vr_pli.pPushConstantRanges = &vr_pcr;
    if (vkd.CreatePipelineLayout(d->device, &vr_pli, NULL, &d->vr_pipeline_layout) != VK_SUCCESS) {
        return false;
    }

    d->pass_luma.pipeline = d->luma_format == VK_FORMAT_R16_SFLOAT
        ? dis_create_compute_pipeline(d, dis_luma_r16_comp, dis_luma_r16_comp_size)
        : dis_create_compute_pipeline(d, dis_luma_r32_comp, dis_luma_r32_comp_size);
    d->pass_gradient.pipeline = dis_create_compute_pipeline(d, dis_gradient_comp, dis_gradient_comp_size);
    d->pass_inverse.pipeline = dis_create_compute_pipeline(d, dis_inverse_search_comp, dis_inverse_search_comp_size);
    d->pass_propagate.pipeline = dis_create_compute_pipeline(d, dis_propagate_comp, dis_propagate_comp_size);
    d->pass_densify.pipeline = dis_create_compute_pipeline(d, dis_densify_comp, dis_densify_comp_size);
    const VkBool32 manual_filter = d->manual_flow_filter ? 1u : 0u;
    VkSpecializationMapEntry spec_entry;
    memset(&spec_entry, 0, sizeof(spec_entry));
    spec_entry.constantID = 0;
    spec_entry.offset = 0;
    spec_entry.size = sizeof(manual_filter);
    VkSpecializationInfo spec;
    memset(&spec, 0, sizeof(spec));
    spec.mapEntryCount = 1;
    spec.pMapEntries = &spec_entry;
    spec.dataSize = sizeof(manual_filter);
    spec.pData = &manual_filter;
    d->pass_interp.pipeline = dis_create_compute_pipeline_with_layout(
        d, dis_interpolate_comp, dis_interpolate_comp_size, d->pipeline_layout, &spec);

    d->pass_vr_prep.pipeline = dis_create_compute_pipeline_with_layout(d, dis_vr_prep_comp, dis_vr_prep_comp_size, d->vr_pipeline_layout, NULL);
    d->pass_vr_d1.pipeline = dis_create_compute_pipeline_with_layout(d, dis_vr_d1_comp, dis_vr_d1_comp_size, d->vr_pipeline_layout, NULL);
    d->pass_vr_d2.pipeline = dis_create_compute_pipeline_with_layout(d, dis_vr_d2_comp, dis_vr_d2_comp_size, d->vr_pipeline_layout, NULL);
    d->pass_vr_w.pipeline = dis_create_compute_pipeline_with_layout(d, dis_vr_w_comp, dis_vr_w_comp_size, d->vr_pipeline_layout, NULL);
    d->pass_vr_coef.pipeline = dis_create_compute_pipeline_with_layout(d, dis_vr_coef_comp, dis_vr_coef_comp_size, d->vr_pipeline_layout, NULL);
    d->pass_vr_sor.pipeline = dis_create_compute_pipeline_with_layout(d, dis_vr_sor_comp, dis_vr_sor_comp_size, d->vr_pipeline_layout, NULL);
    d->pass_vr_add.pipeline = dis_create_compute_pipeline_with_layout(d, dis_vr_add_comp, dis_vr_add_comp_size, d->vr_pipeline_layout, NULL);
    d->pass_hist.pipeline = dis_create_compute_pipeline_with_layout(d, dis_hist_comp, dis_hist_comp_size, d->vr_pipeline_layout, NULL);
    d->pass_side.pipeline = dis_create_compute_pipeline(d, dis_side_comp, dis_side_comp_size);
    d->pass_pack.pipeline = dis_create_compute_pipeline(d, dis_flow_pack_comp, dis_flow_pack_comp_size);

    // The hardware-motion luminance pass writes a host-visible buffer, which neither shared
    // layout has, so it gets a small layout and pool of its own: the colour frame and the buffer.
    VkDescriptorSetLayoutBinding me_bindings[2];
    memset(me_bindings, 0, sizeof(me_bindings));
    me_bindings[0].binding = 0;
    me_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    me_bindings[0].descriptorCount = 1;
    me_bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    me_bindings[1].binding = 1;
    me_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    me_bindings[1].descriptorCount = 1;
    me_bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo me_li;
    memset(&me_li, 0, sizeof(me_li));
    me_li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    me_li.bindingCount = 2;
    me_li.pBindings = me_bindings;
    if (vkd.CreateDescriptorSetLayout(d->device, &me_li, NULL, &d->me_set_layout) != VK_SUCCESS) {
        return false;
    }
    VkPipelineLayoutCreateInfo me_pli;
    memset(&me_pli, 0, sizeof(me_pli));
    me_pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    me_pli.setLayoutCount = 1;
    me_pli.pSetLayouts = &d->me_set_layout;
    me_pli.pushConstantRangeCount = 1;
    me_pli.pPushConstantRanges = &pcr;
    if (vkd.CreatePipelineLayout(d->device, &me_pli, NULL, &d->me_pipeline_layout) != VK_SUCCESS) {
        return false;
    }
    VkDescriptorPoolSize me_sizes[2];
    memset(me_sizes, 0, sizeof(me_sizes));
    me_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    me_sizes[0].descriptorCount = DIS_SLOTS;
    me_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    me_sizes[1].descriptorCount = DIS_SLOTS;
    VkDescriptorPoolCreateInfo me_pci;
    memset(&me_pci, 0, sizeof(me_pci));
    me_pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    me_pci.maxSets = DIS_SLOTS;
    me_pci.poolSizeCount = 2;
    me_pci.pPoolSizes = me_sizes;
    if (vkd.CreateDescriptorPool(d->device, &me_pci, NULL, &d->me_pool) != VK_SUCCESS) {
        return false;
    }
    d->pass_me_luma.pipeline = dis_create_compute_pipeline_with_layout(
        d, dis_me_luma_comp, dis_me_luma_comp_size, d->me_pipeline_layout, NULL);

    if (!d->pass_gradient.pipeline || !d->pass_inverse.pipeline || !d->pass_propagate.pipeline ||
        !d->pass_densify.pipeline || !d->pass_interp.pipeline ||
        !d->pass_vr_prep.pipeline || !d->pass_vr_d1.pipeline || !d->pass_vr_d2.pipeline ||
        !d->pass_vr_w.pipeline || !d->pass_vr_coef.pipeline || !d->pass_vr_sor.pipeline ||
        !d->pass_vr_add.pipeline || !d->pass_hist.pipeline || !d->pass_side.pipeline ||
        !d->pass_pack.pipeline || !d->pass_me_luma.pipeline) {
        return false;
    }
    return true;
}

static bool dis_create_sampler(VkrDis* d) {
    VkSamplerCreateInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.minLod = 0.0f;
    si.maxLod = (float)DIS_MAX_LEVELS;
    return vkd.CreateSampler(d->device, &si, NULL, &d->sampler) == VK_SUCCESS;
}

static void dis_flow_extent(uint32_t min_side, uint32_t content_w, uint32_t content_h,
                            uint32_t* out_w, uint32_t* out_h) {
    const uint32_t minor = content_w < content_h ? content_w : content_h;
    if (minor == 0 || min_side == 0 || min_side >= minor) {
        *out_w = content_w;
        *out_h = content_h;
        return;
    }
    const double k = (double)min_side / (double)minor;
    *out_w = (uint32_t)((double)content_w * k + 0.5);
    *out_h = (uint32_t)((double)content_h * k + 0.5);
}

static uint32_t dis_levels_for(uint32_t w, uint32_t h) {
    uint32_t levels = 1;
    while ((w >> levels) >= 16 && (h >> levels) >= 16 && levels < DIS_MAX_LEVELS) {
        levels++;
    }
    return levels;
}

static uint32_t dis_sparse_extent(uint32_t extent) {
    return extent > 8u ? 1u + (extent - 8u) / DIS_PATCH_STRIDE : 1u;
}

static uint32_t dis_prop_steps_for(uint32_t level, uint32_t levels, uint32_t floor_steps) {
    static const uint32_t profile[DIS_PROP_STEPS_MAX] = {4u, 3u, 2u, 1u};
    const uint32_t from_coarse = (levels - 1u) - level;
    const uint32_t base = from_coarse < DIS_PROP_STEPS_MAX ? profile[from_coarse] : 1u;
    return base > floor_steps ? base : floor_steps;
}

typedef struct {
    uint32_t vr_fixed_point;
    uint32_t vr_sor;
    uint32_t prop_floor;
    uint32_t vr_levels;
} DisRefine;

static DisRefine dis_refine_for(uint32_t generations) {
    if (generations >= 3u) {
        // The 4x path gets the largest refinement budget in the ladder, because its
        // whole cost lands ONCE PER SOURCE PAIR while the interpolate pass runs three
        // times. At a 30 fps source that is a 33 ms budget for a 320x180 pyramid, so four
        // fixed-point passes and seven SOR sweeps on the finest level are a handful of
        // small dispatches - paid for by a field that is closer to right, which is the one
        // improvement that costs no softness anywhere.
        const DisRefine r = {4u, 7u, 2u, DIS_VR_LEVELS};
        return r;
    }
    if (generations == 2u) {
        const DisRefine r = {2u, 4u, 1u, DIS_VR_LEVELS};
        return r;
    }
    const DisRefine r = {1u, 3u, 1u, DIS_VR_LEVELS};
    return r;
}

static void dis_batch_flush(VkrDis* d, DisBatch* b) {
    if (b->count == 0) return;
    vkd.UpdateDescriptorSets(d->device, b->count, b->w, 0, NULL);
    b->count = 0;
}

static void dis_batch_sampled(VkrDis* d, DisBatch* b, VkDescriptorSet set, uint32_t binding,
                              VkImageView view, VkSampler sampler) {
    if (b->count == DIS_MAX_DESCRIPTOR_WRITES) dis_batch_flush(d, b);
    VkDescriptorImageInfo* info = &b->img[b->count];
    memset(info, 0, sizeof(*info));
    info->sampler = sampler;
    info->imageView = view;
    info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet* w = &b->w[b->count];
    memset(w, 0, sizeof(*w));
    w->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w->dstSet = set;
    w->dstBinding = binding;
    w->descriptorCount = 1;
    w->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w->pImageInfo = info;
    b->count++;
}

static void dis_batch_storage(VkrDis* d, DisBatch* b, VkDescriptorSet set, uint32_t binding,
                              VkImageView view) {
    if (b->count == DIS_MAX_DESCRIPTOR_WRITES) dis_batch_flush(d, b);
    VkDescriptorImageInfo* info = &b->img[b->count];
    memset(info, 0, sizeof(*info));
    info->imageView = view;
    info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet* w = &b->w[b->count];
    memset(w, 0, sizeof(*w));
    w->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w->dstSet = set;
    w->dstBinding = binding;
    w->descriptorCount = 1;
    w->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w->pImageInfo = info;
    b->count++;
}

static void dis_write_all_descriptors(VkrDis* d) {
    DisBatch b;
    memset(&b, 0, sizeof(b));
    const uint32_t L = d->levels;
    const uint32_t coarse = L - 1;
    const uint32_t vrL = L < DIS_VR_LEVELS ? L : DIS_VR_LEVELS;

    for (uint32_t s = 0; s < DIS_SLOTS; s++) {
        const uint32_t next = s;
        const uint32_t prev = (s + DIS_SLOTS - 1u) % DIS_SLOTS;

        for (uint32_t l = 0; l < L; l++) {
            dis_batch_sampled(d, &b, d->luma_sets[s][l], 0, d->view_flow_color[next][l], d->sampler);
            dis_batch_storage(d, &b, d->luma_sets[s][l], 5, d->view_flow_luma[next][l]);

            dis_batch_sampled(d, &b, d->grad_sets[s][l], 0, d->view_flow_luma[prev][l], d->sampler);
            dis_batch_storage(d, &b, d->grad_sets[s][l], 5, d->view_grad[l]);

            dis_batch_sampled(d, &b, d->inverse_sets[s][l], 0, d->view_flow_luma[prev][l], d->sampler);
            dis_batch_sampled(d, &b, d->inverse_sets[s][l], 1, d->view_flow_luma[next][l], d->sampler);
            dis_batch_sampled(d, &b, d->inverse_sets[s][l], 2, d->view_grad[l], d->sampler);
            // The coarse estimate is the refined flow where the level above was
            // refined and the densified flow where the VR stage was skipped.
            const uint32_t coarse_l = l + 1 < L ? l + 1 : coarse;
            const VkImageView coarse_view = l + 1 < DIS_VR_LEVELS
                ? d->view_flow_refined[coarse_l] : d->view_dense[coarse_l];
            dis_batch_sampled(d, &b, d->inverse_sets[s][l], 3, coarse_view, d->sampler);
            dis_batch_sampled(d, &b, d->inverse_sets[s][l], 4, d->view_me_field, d->sampler);
            dis_batch_storage(d, &b, d->inverse_sets[s][l], 5, d->view_sparse[l]);

            dis_batch_sampled(d, &b, d->prop_ab_sets[s][l], 0, d->view_flow_luma[prev][l], d->sampler);
            dis_batch_sampled(d, &b, d->prop_ab_sets[s][l], 1, d->view_flow_luma[next][l], d->sampler);
            dis_batch_sampled(d, &b, d->prop_ab_sets[s][l], 2, d->view_sparse[l], d->sampler);
            dis_batch_storage(d, &b, d->prop_ab_sets[s][l], 5, d->view_sparse_b[l]);

            dis_batch_sampled(d, &b, d->prop_ba_sets[s][l], 0, d->view_flow_luma[prev][l], d->sampler);
            dis_batch_sampled(d, &b, d->prop_ba_sets[s][l], 1, d->view_flow_luma[next][l], d->sampler);
            dis_batch_sampled(d, &b, d->prop_ba_sets[s][l], 2, d->view_sparse_b[l], d->sampler);
            dis_batch_storage(d, &b, d->prop_ba_sets[s][l], 5, d->view_sparse[l]);

            dis_batch_sampled(d, &b, d->densify_sets[s][l], 0, d->view_sparse[l], d->sampler);
            dis_batch_sampled(d, &b, d->densify_sets[s][l], 1, d->view_flow_luma[prev][l], d->sampler);
            dis_batch_sampled(d, &b, d->densify_sets[s][l], 2, d->view_flow_luma[next][l], d->sampler);
            dis_batch_storage(d, &b, d->densify_sets[s][l], 5, d->view_dense[l]);
        }

        for (uint32_t dir = 0; dir < 2u; dir++) {
            dis_batch_sampled(d, &b, d->interp_sets[s][dir], 0, d->view_color[prev], d->sampler);
            dis_batch_sampled(d, &b, d->interp_sets[s][dir], 1, d->view_color[next], d->sampler);
            dis_batch_sampled(d, &b, d->interp_sets[s][dir], 2, d->view_flow_out, d->sampler);
            dis_batch_sampled(d, &b, d->interp_sets[s][dir], 3, d->view_side, d->sampler);
            dis_batch_sampled(d, &b, d->interp_sets[s][dir], 4, d->view_hist[dir], d->sampler);
            dis_batch_storage(d, &b, d->interp_sets[s][dir], 5, d->view_interp_out);

            dis_batch_sampled(d, &b, d->hist_sets[s][dir], 0, d->view_color[prev], d->sampler);
            dis_batch_sampled(d, &b, d->hist_sets[s][dir], 1, d->view_color[next], d->sampler);
            dis_batch_sampled(d, &b, d->hist_sets[s][dir], 2, d->view_hist[1u - dir], d->sampler);
            dis_batch_storage(d, &b, d->hist_sets[s][dir], DIS_VR_FIRST_STORAGE, d->view_hist[dir]);
        }

        dis_batch_sampled(d, &b, d->side_sets[s], 0, d->view_flow_refined[0], d->sampler);
        dis_batch_storage(d, &b, d->side_sets[s], 5, d->view_side);

        for (uint32_t l = 0; l < vrL; l++) {
            dis_batch_sampled(d, &b, d->vr_prep_sets[s][l], 0, d->view_flow_color[prev][l], d->sampler);
            dis_batch_sampled(d, &b, d->vr_prep_sets[s][l], 1, d->view_flow_color[next][l], d->sampler);
            dis_batch_sampled(d, &b, d->vr_prep_sets[s][l], 2, d->view_dense[l], d->sampler);
            dis_batch_storage(d, &b, d->vr_prep_sets[s][l], DIS_VR_FIRST_STORAGE, d->view_vr_prep[l]);
            dis_batch_storage(d, &b, d->vr_prep_sets[s][l], DIS_VR_FIRST_STORAGE + 1, d->view_vr_dw[0][l]);
        }
    }

    for (uint32_t l = 0; l < vrL; l++) {
        dis_batch_sampled(d, &b, d->vr_d1_set[l], 0, d->view_vr_prep[l], d->sampler);
        dis_batch_storage(d, &b, d->vr_d1_set[l], DIS_VR_FIRST_STORAGE, d->view_vr_d1[l]);

        dis_batch_sampled(d, &b, d->vr_d2_set[l], 0, d->view_vr_d1[l], d->sampler);
        dis_batch_storage(d, &b, d->vr_d2_set[l], DIS_VR_FIRST_STORAGE, d->view_vr_d2[l]);

        dis_batch_sampled(d, &b, d->vr_w_set[l], 0, d->view_dense[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_w_set[l], 1, d->view_vr_dw[0][l], d->sampler);
        dis_batch_storage(d, &b, d->vr_w_set[l], DIS_VR_FIRST_STORAGE, d->view_vr_wt[l]);

        dis_batch_sampled(d, &b, d->vr_coef_set[l], 0, d->view_vr_prep[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_coef_set[l], 1, d->view_vr_d1[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_coef_set[l], 2, d->view_vr_d2[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_coef_set[l], 3, d->view_vr_dw[0][l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_coef_set[l], 4, d->view_dense[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_coef_set[l], 5, d->view_vr_wt[l], d->sampler);
        dis_batch_storage(d, &b, d->vr_coef_set[l], DIS_VR_FIRST_STORAGE, d->view_vr_A[l]);
        dis_batch_storage(d, &b, d->vr_coef_set[l], DIS_VR_FIRST_STORAGE + 1, d->view_vr_B[l]);

        dis_batch_sampled(d, &b, d->vr_sor_ab_set[l], 0, d->view_vr_A[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_sor_ab_set[l], 1, d->view_vr_B[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_sor_ab_set[l], 3, d->view_vr_dw[0][l], d->sampler);
        dis_batch_storage(d, &b, d->vr_sor_ab_set[l], DIS_VR_FIRST_STORAGE, d->view_vr_dw[1][l]);

        dis_batch_sampled(d, &b, d->vr_sor_ba_set[l], 0, d->view_vr_A[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_sor_ba_set[l], 1, d->view_vr_B[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_sor_ba_set[l], 3, d->view_vr_dw[1][l], d->sampler);
        dis_batch_storage(d, &b, d->vr_sor_ba_set[l], DIS_VR_FIRST_STORAGE, d->view_vr_dw[0][l]);

        dis_batch_sampled(d, &b, d->vr_add_set[l], 0, d->view_dense[l], d->sampler);
        dis_batch_sampled(d, &b, d->vr_add_set[l], 1, d->view_vr_dw[0][l], d->sampler);
        dis_batch_storage(d, &b, d->vr_add_set[l], DIS_VR_FIRST_STORAGE, d->view_flow_refined[l]);
    }

    // Not per slot: the input and the output are single images, and this set is written
    // once here and never updated, so sharing it across frames is safe.
    dis_batch_sampled(d, &b, d->pack_set, 0, d->view_flow_refined[0], d->sampler);
    dis_batch_storage(d, &b, d->pack_set, 5, d->view_flow_out);

    dis_batch_flush(d, &b);
}

static void dis_destroy_views(VkrDis* d) {
    for (uint32_t s = 0; s < DIS_SLOTS; s++) dis_destroy_view(d, &d->view_color[s]);
    dis_destroy_view(d, &d->view_interp_out);
    dis_destroy_view(d, &d->view_hist[0]);
    dis_destroy_view(d, &d->view_hist[1]);
    dis_destroy_view(d, &d->view_side);
    dis_destroy_view(d, &d->view_flow_out);
    dis_destroy_view(d, &d->view_me_field);
    for (uint32_t l = 0; l < DIS_MAX_LEVELS; l++) {
        dis_destroy_view(d, &d->view_vr_prep[l]);
        dis_destroy_view(d, &d->view_vr_d1[l]);
        dis_destroy_view(d, &d->view_vr_d2[l]);
        dis_destroy_view(d, &d->view_vr_A[l]);
        dis_destroy_view(d, &d->view_vr_B[l]);
        dis_destroy_view(d, &d->view_vr_wt[l]);
        dis_destroy_view(d, &d->view_vr_dw[0][l]);
        dis_destroy_view(d, &d->view_vr_dw[1][l]);
        dis_destroy_view(d, &d->view_flow_refined[l]);
        for (uint32_t s = 0; s < DIS_SLOTS; s++) {
            dis_destroy_view(d, &d->view_flow_color[s][l]);
            dis_destroy_view(d, &d->view_flow_luma[s][l]);
        }
        dis_destroy_view(d, &d->view_grad[l]);
        dis_destroy_view(d, &d->view_sparse[l]);
        dis_destroy_view(d, &d->view_sparse_b[l]);
        dis_destroy_view(d, &d->view_dense[l]);
    }
}

static void dis_destroy_images(VkrDis* d) {
    dis_destroy_views(d);
    for (uint32_t s = 0; s < DIS_SLOTS; s++) {
        dis_destroy_image(d, &d->color[s]);
        dis_destroy_image(d, &d->flow_color[s]);
        dis_destroy_image(d, &d->flow_luma[s]);
    }
    dis_destroy_image(d, &d->grad);
    for (uint32_t l = 0; l < DIS_MAX_LEVELS; l++) {
        dis_destroy_image(d, &d->flow_sparse[l]);
        dis_destroy_image(d, &d->flow_sparse_b[l]);
    }
    dis_destroy_image(d, &d->flow_dense);
    dis_destroy_image(d, &d->interp_out);
    dis_destroy_image(d, &d->vr_prep);
    dis_destroy_image(d, &d->vr_d1);
    dis_destroy_image(d, &d->vr_d2);
    dis_destroy_image(d, &d->vr_A);
    dis_destroy_image(d, &d->vr_B);
    dis_destroy_image(d, &d->vr_wt);
    dis_destroy_image(d, &d->vr_dw[0]);
    dis_destroy_image(d, &d->vr_dw[1]);
    dis_destroy_image(d, &d->flow_refined);
    dis_destroy_image(d, &d->hist[0]);
    dis_destroy_image(d, &d->hist[1]);
    dis_destroy_image(d, &d->side);
    dis_destroy_image(d, &d->flow_out);
    dis_destroy_image(d, &d->me_field);
    dis_destroy_me(d);
}

static void dis_destroy_buffer(VkrDis* d, VkBuffer* buf, VkDeviceMemory* mem, void** map) {
    if (*map) vkd.UnmapMemory(d->device, *mem);
    if (*buf) vkd.DestroyBuffer(d->device, *buf, NULL);
    if (*mem) vkd.FreeMemory(d->device, *mem, NULL);
    *buf = VK_NULL_HANDLE;
    *mem = VK_NULL_HANDLE;
    *map = NULL;
}

// Host-visible buffer, mapped for its whole life. Cached memory is preferred for a buffer the
// CPU reads back: uncached reads of a few hundred kilobytes cost far more than the invalidate.
static bool dis_create_host_buffer(VkrDis* d, VkDeviceSize size, VkBufferUsageFlags usage,
                                   bool readback, VkBuffer* buf, VkDeviceMemory* mem, void** map,
                                   bool* coherent) {
    VkBufferCreateInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkd.CreateBuffer(d->device, &bi, NULL, buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    vkd.GetBufferMemoryRequirements(d->device, *buf, &mr);
    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags HC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags CA = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    uint32_t type = UINT32_MAX;
    if (readback) {
        type = dis_find_memory_type(d, mr.memoryTypeBits, HV | CA | HC);
        if (type == UINT32_MAX) type = dis_find_memory_type(d, mr.memoryTypeBits, HV | CA);
    }
    if (type == UINT32_MAX) type = dis_find_memory_type(d, mr.memoryTypeBits, HV | HC);
    if (type == UINT32_MAX) type = dis_find_memory_type(d, mr.memoryTypeBits, HV);
    if (type == UINT32_MAX) return false;
    *coherent = (d->mem_props.memoryTypes[type].propertyFlags & HC) != 0;
    VkMemoryAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    if (vkd.AllocateMemory(d->device, &ai, NULL, mem) != VK_SUCCESS) return false;
    if (vkd.BindBufferMemory(d->device, *buf, *mem, 0) != VK_SUCCESS) return false;
    return vkd.MapMemory(d->device, *mem, 0, VK_WHOLE_SIZE, 0, map) == VK_SUCCESS;
}

static void dis_destroy_me(VkrDis* d) {
    dis_qcom_me_destroy(d->me);
    d->me = NULL;
    dis_destroy_buffer(d, &d->me_luma_buf, &d->me_luma_mem, &d->me_luma_map);
    dis_destroy_buffer(d, &d->me_field_buf, &d->me_field_mem, &d->me_field_map);
    free(d->me_xy);
    d->me_xy = NULL;
    d->hint_level = -1;
}

// The hint seeds the search on level me_level of a w x h pyramid: the estimator's field is that
// level's size, and its input is the field times the block size, which for the usual 2:1 levels
// is twice the flow extent - so the estimator sees finer detail than the level it seeds, and its
// fixed search range covers twice the motion it would at the flow extent.
static void dis_create_me(VkrDis* d, uint32_t w, uint32_t h) {
    dis_destroy_me(d);
    if (!d->hw_motion || d->levels < 3) return;
    uint32_t bx = 0, by = 0;
    if (!dis_qcom_me_supported(&bx, &by)) return;

    d->me_level = 2;
    d->me_field_w = w >> d->me_level;
    d->me_field_h = h >> d->me_level;
    d->me_w = d->me_field_w * bx;
    d->me_h = d->me_field_h * by;
    // Tiny inputs came back as NaN on Adreno 750 (320x176); stay well clear of that.
    if (d->me_w < 256 || d->me_h < 144 || (d->me_w & 3u)) return;

    const VkDeviceSize luma_bytes = (VkDeviceSize)d->me_w * d->me_h;
    const VkDeviceSize field_bytes = (VkDeviceSize)d->me_field_w * d->me_field_h * 2 * sizeof(float);
    if (!dis_create_host_buffer(d, luma_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true,
                                &d->me_luma_buf, &d->me_luma_mem, &d->me_luma_map,
                                &d->me_luma_coherent) ||
        !dis_create_host_buffer(d, field_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false,
                                &d->me_field_buf, &d->me_field_mem, &d->me_field_map,
                                &d->me_field_coherent)) {
        DIS_LOGW("DIS hardware motion: host buffers unavailable; using DIS alone");
        dis_destroy_me(d);
        return;
    }
    d->me_xy = (float*)malloc((size_t)d->me_field_w * d->me_field_h * 2 * sizeof(float));
    d->me = d->me_xy ? dis_qcom_me_create(d->me_w, d->me_h) : NULL;
    if (!d->me) {
        dis_destroy_me(d);
        return;
    }

    for (uint32_t s = 0; s < DIS_SLOTS; s++) {
        VkDescriptorImageInfo ii;
        memset(&ii, 0, sizeof(ii));
        ii.sampler = d->sampler;
        ii.imageView = d->view_color[s];
        ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo bi;
        memset(&bi, 0, sizeof(bi));
        bi.buffer = d->me_luma_buf;
        bi.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet w2[2];
        memset(w2, 0, sizeof(w2));
        w2[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w2[0].dstSet = d->me_sets[s];
        w2[0].dstBinding = 0;
        w2[0].descriptorCount = 1;
        w2[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w2[0].pImageInfo = &ii;
        w2[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w2[1].dstSet = d->me_sets[s];
        w2[1].dstBinding = 1;
        w2[1].descriptorCount = 1;
        w2[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w2[1].pBufferInfo = &bi;
        vkd.UpdateDescriptorSets(d->device, 2, w2, 0, NULL);
    }
    DIS_LOGI("DIS hardware motion hint: GL_QCOM_motion_estimation on %ux%u seeds level %u (%ux%u)",
             d->me_w, d->me_h, d->me_level, d->me_field_w, d->me_field_h);
}

static bool dis_create_resources(VkrDis* d, uint32_t w, uint32_t h, uint32_t full_w,
                                 uint32_t full_h, VkFormat format) {
    dis_destroy_images(d);
    d->layouts_primed = false;

    const uint32_t L = d->levels;
    for (uint32_t s = 0; s < DIS_SLOTS; s++) {
        if (!dis_create_image(d, &d->color[s], full_w, full_h, format, 1,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return false;
        if (!dis_create_image(d, &d->flow_color[s], w, h, format, L,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return false;
        if (!dis_create_image(d, &d->flow_luma[s], w, h, d->luma_format, L,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    }
    if (!dis_create_image(d, &d->grad, w, h, VK_FORMAT_R32G32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    for (uint32_t l = 0; l < L; l++) {
        const uint32_t spw = dis_sparse_extent(w >> l);
        const uint32_t sph = dis_sparse_extent(h >> l);
        if (!dis_create_image(d, &d->flow_sparse[l], spw, sph, VK_FORMAT_R32G32B32A32_SFLOAT, 1,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
        if (!dis_create_image(d, &d->flow_sparse_b[l], spw, sph, VK_FORMAT_R32G32B32A32_SFLOAT, 1,
                              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    }
    if (!dis_create_image(d, &d->flow_dense, w, h, VK_FORMAT_R32G32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->interp_out, full_w, full_h, VK_FORMAT_R8G8B8A8_UNORM, 1,
                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        return false;
    }

    if (!dis_create_image(d, &d->vr_prep, w, h, VK_FORMAT_R32G32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->vr_d1, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->vr_d2, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->vr_A, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->vr_B, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->vr_wt, w, h, VK_FORMAT_R32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->vr_dw[0], w, h, VK_FORMAT_R32G32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->vr_dw[1], w, h, VK_FORMAT_R32G32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->flow_refined, w, h, VK_FORMAT_R32G32_SFLOAT, L,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    // R32_SFLOAT at half the content resolution: the storage format is already
    // required by the flow images, so the history adds no device requirement.
    const uint32_t hist_w = full_w > 1u ? full_w / 2u : 1u;
    const uint32_t hist_h = full_h > 1u ? full_h / 2u : 1u;
    if (!dis_create_image(d, &d->hist[0], hist_w, hist_h, VK_FORMAT_R32_SFLOAT, 1,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->hist[1], hist_w, hist_h, VK_FORMAT_R32_SFLOAT, 1,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    if (!dis_create_image(d, &d->side, w, h, VK_FORMAT_R32_SFLOAT, 1,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    // The finished level-0 field as the interpolate pass reads it: RG16F is filterable on
    // every device, so its three lookups per output pixel stay single bilinear taps.
    if (!dis_create_image(d, &d->flow_out, w, h, VK_FORMAT_R16G16_SFLOAT, 1,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) return false;
    // Hardware motion hint, one vector per block of the level it seeds. It always exists so the
    // search's binding stays valid; the search reads it only on frames that uploaded one.
    const uint32_t hint_w = L >= 3 ? (w >> 2) : 1u;
    const uint32_t hint_h = L >= 3 ? (h >> 2) : 1u;
    if (!dis_create_image(d, &d->me_field, hint_w ? hint_w : 1u, hint_h ? hint_h : 1u,
                          VK_FORMAT_R32G32_SFLOAT, 1,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return false;

    for (uint32_t s = 0; s < DIS_SLOTS; s++) {
        if (!dis_create_view(d, d->color[s].image, format, 0, 1, &d->view_color[s])) return false;
    }
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t s = 0; s < DIS_SLOTS; s++) {
            if (!dis_create_view(d, d->flow_color[s].image, format, l, 1,
                                 &d->view_flow_color[s][l])) return false;
            if (!dis_create_view(d, d->flow_luma[s].image, d->luma_format, l, 1,
                                 &d->view_flow_luma[s][l])) return false;
        }
        if (!dis_create_view(d, d->grad.image, VK_FORMAT_R32G32_SFLOAT, l, 1, &d->view_grad[l])) return false;
        if (!dis_create_view(d, d->flow_sparse[l].image, VK_FORMAT_R32G32B32A32_SFLOAT, 0, 1, &d->view_sparse[l])) return false;
        if (!dis_create_view(d, d->flow_sparse_b[l].image, VK_FORMAT_R32G32B32A32_SFLOAT, 0, 1, &d->view_sparse_b[l])) return false;
        if (!dis_create_view(d, d->flow_dense.image, VK_FORMAT_R32G32_SFLOAT, l, 1, &d->view_dense[l])) return false;
        if (!dis_create_view(d, d->vr_prep.image, VK_FORMAT_R32G32_SFLOAT, l, 1, &d->view_vr_prep[l])) return false;
        if (!dis_create_view(d, d->vr_d1.image, VK_FORMAT_R32G32B32A32_SFLOAT, l, 1, &d->view_vr_d1[l])) return false;
        if (!dis_create_view(d, d->vr_d2.image, VK_FORMAT_R32G32B32A32_SFLOAT, l, 1, &d->view_vr_d2[l])) return false;
        if (!dis_create_view(d, d->vr_A.image, VK_FORMAT_R32G32B32A32_SFLOAT, l, 1, &d->view_vr_A[l])) return false;
        if (!dis_create_view(d, d->vr_B.image, VK_FORMAT_R32G32B32A32_SFLOAT, l, 1, &d->view_vr_B[l])) return false;
        if (!dis_create_view(d, d->vr_wt.image, VK_FORMAT_R32_SFLOAT, l, 1, &d->view_vr_wt[l])) return false;
        if (!dis_create_view(d, d->vr_dw[0].image, VK_FORMAT_R32G32_SFLOAT, l, 1, &d->view_vr_dw[0][l])) return false;
        if (!dis_create_view(d, d->vr_dw[1].image, VK_FORMAT_R32G32_SFLOAT, l, 1, &d->view_vr_dw[1][l])) return false;
        if (!dis_create_view(d, d->flow_refined.image, VK_FORMAT_R32G32_SFLOAT, l, 1, &d->view_flow_refined[l])) return false;
    }
    if (!dis_create_view(d, d->interp_out.image, VK_FORMAT_R8G8B8A8_UNORM, 0, 1, &d->view_interp_out)) return false;
    if (!dis_create_view(d, d->hist[0].image, VK_FORMAT_R32_SFLOAT, 0, 1, &d->view_hist[0])) return false;
    if (!dis_create_view(d, d->hist[1].image, VK_FORMAT_R32_SFLOAT, 0, 1, &d->view_hist[1])) return false;
    if (!dis_create_view(d, d->side.image, VK_FORMAT_R32_SFLOAT, 0, 1, &d->view_side)) return false;
    if (!dis_create_view(d, d->flow_out.image, VK_FORMAT_R16G16_SFLOAT, 0, 1, &d->view_flow_out)) return false;
    if (!dis_create_view(d, d->me_field.image, VK_FORMAT_R32G32_SFLOAT, 0, 1, &d->view_me_field)) return false;

    vkr_dis_reset(d);
    dis_write_all_descriptors(d);
    dis_create_me(d, w, h);
    return true;
}

static bool dis_alloc(VkrDis* d, VkDescriptorSetLayout layout, uint32_t count,
                      VkDescriptorSet* out) {
    VkDescriptorSetLayout layouts[8];
    if (count > 8) return false;
    for (uint32_t i = 0; i < count; i++) layouts[i] = layout;
    VkDescriptorSetAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = d->pool;
    ai.descriptorSetCount = count;
    ai.pSetLayouts = layouts;
    const VkResult res = vkd.AllocateDescriptorSets(d->device, &ai, out);
    if (res != VK_SUCCESS) {
        DIS_LOGW("DIS descriptor allocation failed (%d) asking for %u sets", (int)res, count);
        return false;
    }
    return true;
}

static bool dis_allocate_sets(VkrDis* d) {
    for (uint32_t s = 0; s < DIS_SLOTS; s++) {
        for (uint32_t l = 0; l < DIS_MAX_LEVELS; l++) {
            VkDescriptorSet sets[6];
            if (!dis_alloc(d, d->set_layout, 6, sets)) return false;
            d->grad_sets[s][l] = sets[0];
            d->inverse_sets[s][l] = sets[1];
            d->densify_sets[s][l] = sets[2];
            d->prop_ab_sets[s][l] = sets[3];
            d->prop_ba_sets[s][l] = sets[4];
            d->luma_sets[s][l] = sets[5];
        }
        if (!dis_alloc(d, d->set_layout, 1, &d->interp_sets[s][0])) return false;
        if (!dis_alloc(d, d->set_layout, 1, &d->interp_sets[s][1])) return false;
        if (!dis_alloc(d, d->set_layout, 1, &d->side_sets[s])) return false;
        if (!dis_alloc(d, d->vr_set_layout, 1, &d->hist_sets[s][0])) return false;
        if (!dis_alloc(d, d->vr_set_layout, 1, &d->hist_sets[s][1])) return false;
        for (uint32_t l = 0; l < DIS_VR_LEVELS; l++) {
            if (!dis_alloc(d, d->vr_set_layout, 1, &d->vr_prep_sets[s][l])) return false;
        }
    }

    if (!dis_alloc(d, d->set_layout, 1, &d->pack_set)) return false;

    for (uint32_t s = 0; s < DIS_SLOTS; s++) {
        VkDescriptorSetAllocateInfo ai;
        memset(&ai, 0, sizeof(ai));
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = d->me_pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &d->me_set_layout;
        if (vkd.AllocateDescriptorSets(d->device, &ai, &d->me_sets[s]) != VK_SUCCESS) return false;
    }

    for (uint32_t l = 0; l < DIS_VR_LEVELS; l++) {
        VkDescriptorSet vr_sets[DIS_VR_SHARED_SETS];
        if (!dis_alloc(d, d->vr_set_layout, DIS_VR_SHARED_SETS, vr_sets)) return false;
        d->vr_d1_set[l] = vr_sets[0];
        d->vr_d2_set[l] = vr_sets[1];
        d->vr_w_set[l] = vr_sets[2];
        d->vr_coef_set[l] = vr_sets[3];
        d->vr_sor_ab_set[l] = vr_sets[4];
        d->vr_sor_ba_set[l] = vr_sets[5];
        d->vr_add_set[l] = vr_sets[6];
    }
    return true;
}

static void dis_dispatch(VkrDis* d, VkCommandBuffer cmd, VkPipeline pipeline, VkDescriptorSet set,
                         uint32_t w, uint32_t h) {
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipeline_layout, 0, 1, &set,
                              0, NULL);
    vkd.CmdDispatch(cmd, (w + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE,
                    (h + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE, 1);
}

static void dis_blit_rect(VkCommandBuffer cmd,
                          VkImage src, int32_t sx, int32_t sy, uint32_t sw, uint32_t sh,
                          VkImage dst, int32_t dx, int32_t dy, uint32_t dw, uint32_t dh,
                          VkFilter filter) {
    VkImageBlit blit;
    memset(&blit, 0, sizeof(blit));
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[0].x = sx;
    blit.srcOffsets[0].y = sy;
    blit.srcOffsets[1].x = sx + (int32_t)sw;
    blit.srcOffsets[1].y = sy + (int32_t)sh;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[0].x = dx;
    blit.dstOffsets[0].y = dy;
    blit.dstOffsets[1].x = dx + (int32_t)dw;
    blit.dstOffsets[1].y = dy + (int32_t)dh;
    blit.dstOffsets[1].z = 1;
    vkd.CmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                     filter);
}

static void dis_blit_mip(VkCommandBuffer cmd, VkImage img, uint32_t src_level, uint32_t dst_level,
                         uint32_t src_w, uint32_t src_h, uint32_t dst_w, uint32_t dst_h) {
    VkImageBlit blit;
    memset(&blit, 0, sizeof(blit));
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = src_level;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1].x = (int32_t)src_w;
    blit.srcOffsets[1].y = (int32_t)src_h;
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = dst_level;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[1].x = (int32_t)dst_w;
    blit.dstOffsets[1].y = (int32_t)dst_h;
    blit.dstOffsets[1].z = 1;
    vkd.CmdBlitImage(cmd, img, VK_IMAGE_LAYOUT_GENERAL, img, VK_IMAGE_LAYOUT_GENERAL, 1, &blit,
                     VK_FILTER_LINEAR);
}

static const char* dis_format_name(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R16_SFLOAT: return "R16_SFLOAT";
        case VK_FORMAT_R32_SFLOAT: return "R32_SFLOAT";
        case VK_FORMAT_R32G32_SFLOAT: return "R32G32_SFLOAT";
        case VK_FORMAT_R16G16_SFLOAT: return "R16G16_SFLOAT";
        case VK_FORMAT_R32G32B32A32_SFLOAT: return "R32G32B32A32_SFLOAT";
        case VK_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
        case VK_FORMAT_UNDEFINED: return "none";
        default: return "format";
    }
}

static void dis_missing_features(VkFormatFeatureFlags missing, char* out, size_t cap) {
    out[0] = '\0';
    const struct { VkFormatFeatureFlags bit; const char* name; } names[] = {
        {VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, "STORAGE_IMAGE"},
        {VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT, "SAMPLED_IMAGE"},
        {VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT, "SAMPLED_IMAGE_FILTER_LINEAR"},
        {VK_FORMAT_FEATURE_BLIT_SRC_BIT, "BLIT_SRC"},
        {VK_FORMAT_FEATURE_BLIT_DST_BIT, "BLIT_DST"},
    };
    for (uint32_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        if (!(missing & names[i].bit)) continue;
        if (out[0]) strncat(out, "+", cap - strlen(out) - 1);
        strncat(out, names[i].name, cap - strlen(out) - 1);
    }
    if (!out[0]) strncat(out, "none", cap - 1);
}

static bool dis_audit_formats(VkrDis* d) {
    const VkFormatFeatureFlags STORE = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    const VkFormatFeatureFlags READ = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    const VkFormatFeatureFlags FILTER = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;

    if (d->luma_format == VK_FORMAT_UNDEFINED) {
        DIS_LOGW("DIS needs a single-channel float plane it can both write and filter; "
                 "neither R16_SFLOAT nor R32_SFLOAT qualifies on this device");
        return false;
    }

    const struct {
        VkFormat format;
        VkFormatFeatureFlags need;
        const char* purpose;
    } reqs[] = {
        {VK_FORMAT_R32G32_SFLOAT, STORE | READ, "optical flow"},
        {VK_FORMAT_R16G16_SFLOAT, STORE | READ, "interpolation flow"},
        {VK_FORMAT_R32G32B32A32_SFLOAT, STORE | READ, "sparse flow and refinement"},
        {VK_FORMAT_R32_SFLOAT, STORE | READ, "refinement weights"},
        {VK_FORMAT_R8G8B8A8_UNORM, STORE | VK_FORMAT_FEATURE_BLIT_SRC_BIT,
         "interpolated output"},
        {d->luma_format, STORE | READ | FILTER, "luminance plane"},
    };

    char line[512];
    line[0] = '\0';
    bool ok = true;

    for (uint32_t i = 0; i < sizeof(reqs) / sizeof(reqs[0]); i++) {
        VkFormatProperties fp;
        memset(&fp, 0, sizeof(fp));
        vkd.GetPhysicalDeviceFormatProperties(d->physical_device, reqs[i].format, &fp);
        const VkFormatFeatureFlags missing = reqs[i].need & ~fp.optimalTilingFeatures;

        char entry[96];
        snprintf(entry, sizeof(entry), "%s%s=%s", line[0] ? " " : "",
                 dis_format_name(reqs[i].format), missing ? "MISSING" : "ok");
        strncat(line, entry, sizeof(line) - strlen(line) - 1);

        if (missing) {
            char names[256];
            dis_missing_features(missing, names, sizeof(names));
            DIS_LOGW("DIS needs %s on %s for %s, and this device does not report it",
                     names, dis_format_name(reqs[i].format), reqs[i].purpose);
            ok = false;
        }
    }

    VkFormatProperties flow_fp;
    memset(&flow_fp, 0, sizeof(flow_fp));
    vkd.GetPhysicalDeviceFormatProperties(d->physical_device, VK_FORMAT_R16G16_SFLOAT, &flow_fp);
    d->manual_flow_filter = (flow_fp.optimalTilingFeatures & FILTER) == 0;

    DIS_LOGI("DIS format support: %s | flow filtering: %s", line,
             d->manual_flow_filter ? "in shader (driver cannot filter R16G16_SFLOAT)"
                                   : "sampler");
    return ok;
}

VkrDis* vkr_dis_create(VkDevice device, VkPhysicalDevice physical_device) {
    if (device == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE) return NULL;

    VkrDis* d = (VkrDis*)calloc(1, sizeof(VkrDis));
    d->device = device;
    d->physical_device = physical_device;
    d->flow_min_side = DIS_DEFAULT_FLOW_MIN_SIDE;
    d->target_fps = 0;
    d->refresh_rate = 0.0f;
    d->plan_log_gen = -1;
    // Off by default: on Adreno 750 the hint left quality unchanged and cost ~2.5 ms per real
    // frame, most of it the mid-frame submit it needs. Opt in on a device without a rebuild:
    //   adb shell setprop debug.winnative.dis.hwme 1
    d->hw_motion = false;
#ifdef __ANDROID__
    char prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("debug.winnative.dis.hwme", prop) > 0 && prop[0] == '1') {
        d->hw_motion = true;
        DIS_LOGI("DIS hardware motion hint enabled by debug.winnative.dis.hwme");
    }
#endif
    d->hint_level = -1;
    vkd.GetPhysicalDeviceMemoryProperties(physical_device, &d->mem_props);
    d->luma_format = dis_pick_luma_format(d);
    if (!dis_audit_formats(d)) {
        DIS_LOGW("DIS cannot run on this device's format support; frame generation stays off");
        vkr_dis_destroy(d);
        return NULL;
    }

    if (!dis_create_sampler(d) || !dis_create_pipelines(d)) {
        DIS_LOGW("DIS shaders could not be built; frame generation stays off");
        vkr_dis_destroy(d);
        return NULL;
    }
    if (!dis_allocate_sets(d)) {
        DIS_LOGW("DIS descriptor sets could not be allocated; frame generation stays off");
        vkr_dis_destroy(d);
        return NULL;
    }
    DIS_LOGI("DIS frame generation ready");
    return d;
}

void vkr_dis_destroy(VkrDis* d) {
    if (!d) return;
    dis_destroy_images(d);
    if (d->sampler) vkd.DestroySampler(d->device, d->sampler, NULL);
    if (d->pass_luma.pipeline) vkd.DestroyPipeline(d->device, d->pass_luma.pipeline, NULL);
    if (d->pass_gradient.pipeline) vkd.DestroyPipeline(d->device, d->pass_gradient.pipeline, NULL);
    if (d->pass_inverse.pipeline) vkd.DestroyPipeline(d->device, d->pass_inverse.pipeline, NULL);
    if (d->pass_propagate.pipeline) vkd.DestroyPipeline(d->device, d->pass_propagate.pipeline, NULL);
    if (d->pass_densify.pipeline) vkd.DestroyPipeline(d->device, d->pass_densify.pipeline, NULL);
    if (d->pass_interp.pipeline) vkd.DestroyPipeline(d->device, d->pass_interp.pipeline, NULL);
    if (d->pass_vr_prep.pipeline) vkd.DestroyPipeline(d->device, d->pass_vr_prep.pipeline, NULL);
    if (d->pass_vr_d1.pipeline) vkd.DestroyPipeline(d->device, d->pass_vr_d1.pipeline, NULL);
    if (d->pass_vr_d2.pipeline) vkd.DestroyPipeline(d->device, d->pass_vr_d2.pipeline, NULL);
    if (d->pass_vr_w.pipeline) vkd.DestroyPipeline(d->device, d->pass_vr_w.pipeline, NULL);
    if (d->pass_vr_coef.pipeline) vkd.DestroyPipeline(d->device, d->pass_vr_coef.pipeline, NULL);
    if (d->pass_vr_sor.pipeline) vkd.DestroyPipeline(d->device, d->pass_vr_sor.pipeline, NULL);
    if (d->pass_vr_add.pipeline) vkd.DestroyPipeline(d->device, d->pass_vr_add.pipeline, NULL);
    if (d->pass_hist.pipeline) vkd.DestroyPipeline(d->device, d->pass_hist.pipeline, NULL);
    if (d->pass_side.pipeline) vkd.DestroyPipeline(d->device, d->pass_side.pipeline, NULL);
    if (d->pass_pack.pipeline) vkd.DestroyPipeline(d->device, d->pass_pack.pipeline, NULL);
    if (d->pass_me_luma.pipeline) vkd.DestroyPipeline(d->device, d->pass_me_luma.pipeline, NULL);
    if (d->me_pool) vkd.DestroyDescriptorPool(d->device, d->me_pool, NULL);
    if (d->me_pipeline_layout) vkd.DestroyPipelineLayout(d->device, d->me_pipeline_layout, NULL);
    if (d->me_set_layout) vkd.DestroyDescriptorSetLayout(d->device, d->me_set_layout, NULL);
    if (d->pool) vkd.DestroyDescriptorPool(d->device, d->pool, NULL);
    if (d->pipeline_layout) vkd.DestroyPipelineLayout(d->device, d->pipeline_layout, NULL);
    if (d->vr_pipeline_layout) vkd.DestroyPipelineLayout(d->device, d->vr_pipeline_layout, NULL);
    if (d->set_layout) vkd.DestroyDescriptorSetLayout(d->device, d->set_layout, NULL);
    if (d->vr_set_layout) vkd.DestroyDescriptorSetLayout(d->device, d->vr_set_layout, NULL);
    free(d);
}

void vkr_dis_configure(VkrDis* d, uint32_t flow_min_side, uint32_t target_fps, float refresh_rate) {
    if (!d) return;
    uint32_t side = flow_min_side < DIS_FLOW_MIN_SIDE_FLOOR ? DIS_FLOW_MIN_SIDE_FLOOR
                  : (flow_min_side > DIS_FLOW_MIN_SIDE_CEIL ? DIS_FLOW_MIN_SIDE_CEIL
                                                            : flow_min_side);
    d->flow_min_side = side;
    d->target_fps = target_fps;
    d->refresh_rate = refresh_rate > 0.0f ? refresh_rate : 0.0f;
}

void vkr_dis_set_debug_flow(VkrDis* d, bool debug_flow) {
    if (!d) return;
    d->debug_flow = debug_flow;
}

bool vkr_dis_needs_rebuild(const VkrDis* d, uint32_t width, uint32_t height, VkFormat format,
                           VkrDisContentRect content) {
    if (!d || d->unavailable) return false;
    return !d->built || d->built_full_extent.width != width ||
           d->built_full_extent.height != height || d->built_format != format ||
           d->built_min_side != d->flow_min_side ||
           d->content.width != content.width || d->content.height != content.height;
}

bool vkr_dis_prepare(VkrDis* d, uint32_t width, uint32_t height, VkFormat format,
                     VkrDisContentRect content) {
    if (!d || d->unavailable) return false;
    if (width == 0 || height == 0 || format == VK_FORMAT_UNDEFINED) return false;

    if (content.width < DIS_MIN_EXTENT || content.height < DIS_MIN_EXTENT ||
        content.x < 0 || content.y < 0 ||
        (uint32_t)content.x + content.width > width ||
        (uint32_t)content.y + content.height > height) {
        content.x = 0;
        content.y = 0;
        content.width = width;
        content.height = height;
    }
    d->content = content;

    uint32_t w, h;
    dis_flow_extent(d->flow_min_side, content.width, content.height, &w, &h);
    if (w < DIS_MIN_EXTENT) w = DIS_MIN_EXTENT;
    if (h < DIS_MIN_EXTENT) h = DIS_MIN_EXTENT;

    const uint32_t levels = dis_levels_for(w, h);

    if (d->built && d->built_extent.width == w && d->built_extent.height == h &&
        d->built_full_extent.width == width && d->built_full_extent.height == height &&
        d->built_format == format && d->built_min_side == d->flow_min_side && d->levels == levels &&
        d->content.width == content.width && d->content.height == content.height) {
        d->content.x = content.x;
        d->content.y = content.y;
        return true;
    }

    if (!d->formats_audited) {
        d->formats_audited = true;
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                                          VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                          VK_FORMAT_FEATURE_BLIT_DST_BIT;
        VkFormatProperties fp;
        memset(&fp, 0, sizeof(fp));
        vkd.GetPhysicalDeviceFormatProperties(d->physical_device, format, &fp);
        const VkFormatFeatureFlags missing = need & ~fp.optimalTilingFeatures;
        if (missing) {
            char names[256];
            dis_missing_features(missing, names, sizeof(names));
            DIS_LOGW("DIS needs %s on the guest frame format (%d) and this device does not "
                     "report it; frame generation stays off", names, (int)format);
            d->unavailable = true;
            return false;
        }
    }

    d->levels = levels;

    if (!dis_create_resources(d, w, h, content.width, content.height, format)) {
        DIS_LOGW("DIS resource build failed at %ux%u; frame generation unavailable", w, h);
        dis_destroy_images(d);
        d->unavailable = true;
        return false;
    }

    d->built_extent.width = w;
    d->built_extent.height = h;
    d->built_full_extent.width = width;
    d->built_full_extent.height = height;
    d->built_format = format;
    d->built_min_side = d->flow_min_side;
    d->built = true;
    d->frame_count = 0;
    d->prev_idx = 0;
    d->next_idx = 0;
    d->active_slot = 0;
    d->last_generations = 0;
    DIS_LOGI("DIS resources built at %ux%u (flow min side %u, %u levels); content rect "
             "%dx%d+%d+%d inside a %ux%u composite",
             w, h, d->flow_min_side, levels, (int)content.width, (int)content.height,
             (int)content.x, (int)content.y, width, height);
    return true;
}

static void dis_track_source(VkrDis* d, uint64_t now, uint64_t source_frames) {
    if (d->src_sample_ns == 0) {
        d->src_sample_ns = now;
        d->src_last_frames = source_frames;
        return;
    }

    const uint64_t dt = now - d->src_sample_ns;
    if (dt == 0) return;
    // A burst of presents can land inside one millisecond (startup, alt-tab) and
    // a single such sample spikes the rate estimate to hundreds of fps, which
    // then walks the planner up and down the generation ladder. Let the window
    // span the burst instead.
    if (dt < 2000000ull) return;
    d->src_sample_ns = now;

    const uint64_t drawn =
        source_frames > d->src_last_frames ? source_frames - d->src_last_frames : 0;
    d->src_last_frames = source_frames;

    if (dt > DIS_SRC_STALE_NS) {
        d->src_frame_accum = 0.0f;
        d->src_time_accum = 0.0f;
        d->src_interval = 0.0f;
        d->src_samples = 0;
        return;
    }

    const float elapsed = (float)dt * 1.0e-9f;
    d->src_frame_accum += ((float)drawn - d->src_frame_accum) * DIS_SRC_SMOOTHING;
    d->src_time_accum += (elapsed - d->src_time_accum) * DIS_SRC_SMOOTHING;
    d->src_interval =
        d->src_frame_accum > 0.01f ? d->src_time_accum / d->src_frame_accum : 0.0f;
    if (d->src_samples < DIS_MIN_RATE_SAMPLES) d->src_samples++;
}

static int dis_gen_for_ratio(float ratio, uint32_t capacity) {
    float outputs = ceilf(ratio - DIS_RATIO_SLACK);
    if (outputs < 2.0f) outputs = 2.0f;
    int gen = (int)outputs - 1;
    if (gen < 1) gen = 1;
    if (gen > (int)capacity) gen = (int)capacity;
    return gen;
}

static void dis_log_plan(VkrDis* d, uint64_t now, float source_rate, float desired,
                         float ratio, uint32_t capacity) {
    if (d->planned_gen == d->plan_log_gen && now - d->plan_log_ns < DIS_PLAN_LOG_NS) return;
    d->plan_log_gen = d->planned_gen;
    d->plan_log_ns = now;
    DIS_LOGI("DIS plan: source %.1f fps, target %.1f fps, ratio %.2f -> %d generated "
             "(capacity %u, output %.1f fps)",
             (double)source_rate, (double)desired, (double)ratio, d->planned_gen, capacity,
             (double)(source_rate * (float)(d->planned_gen + 1)));
}

uint32_t vkr_dis_plan(VkrDis* d, uint32_t capacity, uint64_t source_frames) {
    if (!d || d->unavailable || !d->built) return 0;
    if (capacity > VKR_DIS_MAX_GENERATIONS) capacity = VKR_DIS_MAX_GENERATIONS;

    const uint64_t now = dis_now_ns();
    dis_track_source(d, now, source_frames);

    if (capacity == 0 || d->frame_count < 2 || d->src_samples < DIS_MIN_RATE_SAMPLES ||
        d->src_interval <= 0.0f) {
        d->planned_gen = 0;
        d->gen_high_streak = 0;
        d->gen_low_streak = 0;
        return 0;
    }

    const float source_rate = 1.0f / d->src_interval;
    float desired = d->target_fps > 0 ? (float)d->target_fps : d->refresh_rate;
    if (d->refresh_rate > 0.0f && desired > d->refresh_rate) desired = d->refresh_rate;
    if (desired <= 0.0f) return 0;

    if (d->smoothed_desired <= 0.0f) {
        d->smoothed_desired = desired;
    } else {
        d->smoothed_desired += (desired - d->smoothed_desired) * 0.25f;
    }
    const float eff_desired = d->smoothed_desired;

    const float ratio = eff_desired / source_rate;

    if (ratio <= 1.0f) {
        d->planned_gen = 0;
        d->gen_high_streak = 0;
        d->gen_low_streak = 0;
        dis_log_plan(d, now, source_rate, eff_desired, ratio, capacity);
        return 0;
    }

    const int cur = d->planned_gen > (int)capacity ? (int)capacity : d->planned_gen;
    const bool generate = cur > 0 ? (ratio >= DIS_MIN_GEN_RATIO - DIS_RATIO_HYST)
                                  : (ratio >= DIS_MIN_GEN_RATIO);

    int raw = 0;
    if (generate) {
        const int want_up = dis_gen_for_ratio(ratio - DIS_RATIO_HYST, capacity);
        const int want_down = dis_gen_for_ratio(ratio + DIS_RATIO_HYST, capacity);
        raw = cur;
        if (want_up > cur) {
            raw = want_up;
        } else if (want_down < cur) {
            raw = want_down;
        }
        if (raw < 1) raw = 1;
    }

    if (raw > d->planned_gen) {
        d->gen_low_streak = 0;
        d->gen_high_streak++;
        if (d->gen_high_streak >= 2) {
            d->planned_gen = raw;
            d->gen_high_streak = 0;
        }
    } else if (raw < d->planned_gen) {
        d->gen_high_streak = 0;
        d->gen_low_streak++;
        if (d->gen_low_streak >= 3) {
            d->planned_gen = raw;
            d->gen_low_streak = 0;
        }
    } else {
        d->gen_high_streak = 0;
        d->gen_low_streak = 0;
    }

    dis_log_plan(d, now, source_rate, eff_desired, ratio, capacity);
    return (uint32_t)d->planned_gen;
}

// Solver budget for one level. The refinement is a red-black SOR, and the
// number of sweeps a SOR needs scales with how far information has to travel
// across the grid - a level is half the size per axis, so it reaches the same
// relative distance in fewer sweeps. Spending the finest level's sweep count on
// every level buys nothing numerically and costs a dispatch and a barrier each.
static void dis_vr_budget(const DisRefine* refine, uint32_t l, uint32_t* fixed_point,
                          uint32_t* sor) {
    *fixed_point = l == 0 ? refine->vr_fixed_point : 1u;
    // vr_sor is the level-0 budget and coarser levels fall off twice as fast as the grid
    // shrinks. A flat slope spent the finest level's raised count on every level: with the
    // 4x budget at 7 that would be 7, 6, 5, 4 - eight extra dispatches and barriers per
    // pair for sweeps that buy nothing, since a half-size level crosses the same relative
    // distance in fewer of them. This gives 7, 5, 3, 2 instead.
    const uint32_t drop = l * 2u;
    const uint32_t s = refine->vr_sor > drop ? refine->vr_sor - drop : DIS_VR_SOR_FLOOR;
    *sor = s < DIS_VR_SOR_FLOOR ? DIS_VR_SOR_FLOOR : s;
}

static void dis_vr_level(VkrDis* d, VkCommandBuffer cmd, uint32_t slot, uint32_t l,
                         uint32_t lw, uint32_t lh, const DisRefine* refine, bool full) {
    const uint32_t gw = (lw + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE;
    const uint32_t gh = (lh + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE;

    uint32_t vr_fixed_point = 0;
    uint32_t vr_sor = 0;
    dis_vr_budget(refine, l, &vr_fixed_point, &vr_sor);

    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_vr_prep.pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->vr_pipeline_layout, 0, 1,
                              &d->vr_prep_sets[slot][l], 0, NULL);
    vkd.CmdDispatch(cmd, gw, gh, 1);
    dis_compute_barrier(cmd);

    if (full) {
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_vr_d1.pipeline);
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->vr_pipeline_layout, 0, 1,
                                  &d->vr_d1_set[l], 0, NULL);
        vkd.CmdDispatch(cmd, gw, gh, 1);
        dis_compute_barrier(cmd);

        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_vr_d2.pipeline);
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->vr_pipeline_layout, 0, 1,
                                  &d->vr_d2_set[l], 0, NULL);
        vkd.CmdDispatch(cmd, gw, gh, 1);
        dis_compute_barrier(cmd);
        if (l == 0) DIS_PROF(cmd, "L0 vr prep+deriv");

        for (uint32_t k = 0; k < vr_fixed_point; k++) {
            DisVrWPC wpc;
            wpc.alpha2 = DIS_VR_ALPHA * 0.5f;
            wpc.eps2 = DIS_VR_EPS * DIS_VR_EPS;
            vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_vr_w.pipeline);
            vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->vr_pipeline_layout,
                                      0, 1, &d->vr_w_set[l], 0, NULL);
            vkd.CmdPushConstants(cmd, d->vr_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(wpc), &wpc);
            vkd.CmdDispatch(cmd, gw, gh, 1);
            dis_compute_barrier(cmd);
            if (l == 0) DIS_PROF(cmd, "L0 vr w");

            DisVrCoefPC cpc;
            cpc.delta2 = DIS_VR_DELTA * 0.5f;
            cpc.gamma2 = DIS_VR_GAMMA * 0.5f;
            cpc.zeta2 = DIS_VR_ZETA * DIS_VR_ZETA;
            cpc.eps2 = DIS_VR_EPS * DIS_VR_EPS;
            vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_vr_coef.pipeline);
            vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->vr_pipeline_layout,
                                      0, 1, &d->vr_coef_set[l], 0, NULL);
            vkd.CmdPushConstants(cmd, d->vr_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(cpc), &cpc);
            vkd.CmdDispatch(cmd, gw, gh, 1);
            dis_compute_barrier(cmd);
            if (l == 0) DIS_PROF(cmd, "L0 vr coef");

            // ab then ba per iteration, so every iteration ends in vr_dw[0]. Fusing iterations
            // into shared-memory tiles was measured and lost on Adreno 750 (1.9 ms against 1.45
            // at level 0 with a third of the dispatches), so the half-sweeps stay separate.
            vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_vr_sor.pipeline);
            for (uint32_t it = 0; it < vr_sor; it++) {
                for (int parity = 0; parity < 2; parity++) {
                    DisVrSorPC spc;
                    spc.omega = DIS_VR_OMEGA;
                    spc.parity = parity;
                    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                              d->vr_pipeline_layout, 0, 1,
                                              parity ? &d->vr_sor_ba_set[l] : &d->vr_sor_ab_set[l],
                                              0, NULL);
                    vkd.CmdPushConstants(cmd, d->vr_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                         0, sizeof(spc), &spc);
                    vkd.CmdDispatch(cmd, gw, gh, 1);
                    dis_compute_barrier(cmd);
                }
            }
            if (l == 0) DIS_PROF(cmd, "L0 vr sor");
        }
    }

    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_vr_add.pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->vr_pipeline_layout, 0, 1,
                              &d->vr_add_set[l], 0, NULL);
    vkd.CmdDispatch(cmd, gw, gh, 1);
    dis_compute_barrier(cmd);
}

// Hardware motion hint for the pair ending in slot: the newest frame's luminance goes to the
// GLES estimator and its field comes back as the starting candidate of the search on me_level.
//
// The estimator needs this frame's pixels, which the caller has only just recorded, so the
// command buffer is handed back through lush to be submitted and waited on first. What was
// recorded so far - the caller's composite and the copies above - then runs ahead of the rest
// of the frame; the caller submits the returned buffer for everything after.
static VkCommandBuffer dis_hardware_motion(VkrDis* d, VkCommandBuffer cmd, uint32_t slot,
                                           VkrDisFlushFn flush, void* flush_user) {
    const int32_t me_pc[2] = {(int32_t)d->me_w, (int32_t)d->me_h};
    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_me_luma.pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->me_pipeline_layout, 0, 1,
                              &d->me_sets[slot], 0, NULL);
    vkd.CmdPushConstants(cmd, d->me_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(me_pc), me_pc);
    vkd.CmdDispatch(cmd, (d->me_w / 4u + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE,
                    (d->me_h + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE, 1);
    VkMemoryBarrier hb;
    memset(&hb, 0, sizeof(hb));
    hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkd.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                           0, 1, &hb, 0, NULL, 0, NULL);

    cmd = flush(flush_user, cmd);
    if (cmd == VK_NULL_HANDLE) return cmd;

    if (!d->me_luma_coherent) {
        VkMappedMemoryRange r;
        memset(&r, 0, sizeof(r));
        r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        r.memory = d->me_luma_mem;
        r.size = VK_WHOLE_SIZE;
        vkd.InvalidateMappedMemoryRanges(d->device, 1, &r);
    }
    d->me_pairs++;
    if (!dis_qcom_me_push(d->me, (const uint8_t*)d->me_luma_map, d->me_xy)) return cmd;

    // Pixels of the estimator's input -> normalised uv, which is what every flow image in the
    // chain stores. Vectors the estimator could not have found - non-finite, or past half the
    // frame - are marked invalid rather than clamped, so the search simply ignores them.
    const uint32_t n = d->me_field_w * d->me_field_h;
    float* dst = (float*)d->me_field_map;
    const float inv_w = 1.0f / (float)d->me_w;
    const float inv_h = 1.0f / (float)d->me_h;
    for (uint32_t i = 0; i < n; i++) {
        const float vx = d->me_xy[i * 2];
        const float vy = d->me_xy[i * 2 + 1];
        const bool ok = isfinite(vx) && isfinite(vy) &&
                        fabsf(vx) < 0.5f * (float)d->me_w && fabsf(vy) < 0.5f * (float)d->me_h;
        dst[i * 2] = ok ? vx * inv_w : 1.0e7f;
        dst[i * 2 + 1] = ok ? vy * inv_h : 1.0e7f;
    }
    if (DIS_ME_PRIMARY) {
        // As the level's result the field has no search behind it to reject a bad block, so
        // outliers are taken out here: a 3x3 component median over the valid neighbours, and
        // zero where there are none.
        const int fw = (int)d->me_field_w, fh = (int)d->me_field_h;
        float* med = d->me_xy;  // reused: the raw pixels are no longer needed
        for (int y = 0; y < fh; y++) {
            for (int x = 0; x < fw; x++) {
                for (int c = 0; c < 2; c++) {
                    float v[9];
                    int k = 0;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            const int xx = x + dx, yy = y + dy;
                            if (xx < 0 || yy < 0 || xx >= fw || yy >= fh) continue;
                            const float s = dst[(yy * fw + xx) * 2 + c];
                            if (fabsf(s) < 1.0e6f) v[k++] = s;
                        }
                    }
                    for (int a = 1; a < k; a++) {
                        const float key = v[a];
                        int b = a - 1;
                        while (b >= 0 && v[b] > key) { v[b + 1] = v[b]; b--; }
                        v[b + 1] = key;
                    }
                    med[(y * fw + x) * 2 + c] = k ? v[k / 2] : 0.0f;
                }
            }
        }
        memcpy(dst, med, (size_t)n * 2 * sizeof(float));
    }
    if (!d->me_field_coherent) {
        VkMappedMemoryRange r;
        memset(&r, 0, sizeof(r));
        r.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        r.memory = d->me_field_mem;
        r.size = VK_WHOLE_SIZE;
        vkd.FlushMappedMemoryRanges(d->device, 1, &r);
    }

    dis_barrier(cmd, d->me_field.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region;
    memset(&region, 0, sizeof(region));
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = d->me_field_w;
    region.imageExtent.height = d->me_field_h;
    region.imageExtent.depth = 1;
    vkd.CmdCopyBufferToImage(cmd, d->me_field_buf, d->me_field.image, VK_IMAGE_LAYOUT_GENERAL, 1,
                             &region);
    dis_barrier(cmd, d->me_field.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    if (DIS_ME_PRIMARY) {
        // The field is exactly the size of level me_level, so it drops into that mip of the
        // refined flow, where the next finer level's search picks it up as its coarse estimate.
        VkImageCopy ic;
        memset(&ic, 0, sizeof(ic));
        ic.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ic.srcSubresource.layerCount = 1;
        ic.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ic.dstSubresource.mipLevel = d->me_level;
        ic.dstSubresource.layerCount = 1;
        ic.extent.width = d->me_field_w;
        ic.extent.height = d->me_field_h;
        ic.extent.depth = 1;
        dis_barrier(cmd, d->flow_refined.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        vkd.CmdCopyImage(cmd, d->me_field.image, VK_IMAGE_LAYOUT_GENERAL, d->flow_refined.image,
                         VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
        dis_barrier(cmd, d->flow_refined.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        d->me_primary_frame = true;
    } else {
        d->hint_level = (int)d->me_level;
    }
    d->me_hinted++;
    if ((d->me_hinted % 600u) == 1u) {
        DIS_LOGI("DIS hardware motion hint: %llu of %llu pairs seeded",
                 (unsigned long long)d->me_hinted, (unsigned long long)d->me_pairs);
    }
    return cmd;
}

VkCommandBuffer vkr_dis_process_ex(VkrDis* d, VkCommandBuffer cmd, VkImage source,
                                   uint32_t width, uint32_t height, uint32_t generations,
                                   VkrDisFlushFn flush, void* flush_user) {
    if (!d || !d->built || d->unavailable) return cmd;

    d->last_generations = generations;
    d->hint_level = -1;
    d->me_primary_frame = false;

    dis_prime_layouts(d, cmd);
    DIS_PROF_BEGIN(d, cmd);

    const DisRefine refine = dis_refine_for(generations);
    const uint32_t L = d->levels;
    const uint32_t coarse = L - 1;
    const uint32_t w = d->built_extent.width;
    const uint32_t h = d->built_extent.height;
    const uint32_t full_w = d->content.width;
    const uint32_t full_h = d->content.height;
    const int32_t cx = d->content.x;
    const int32_t cy = d->content.y;
    (void)width; (void)height;

    const uint32_t slot = (uint32_t)(d->frame_count % DIS_SLOTS);
    DisImage* full_dst = &d->color[slot];
    DisImage* flow_dst = &d->flow_color[slot];

    dis_barrier(cmd, full_dst->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    dis_blit_rect(cmd, source, cx, cy, full_w, full_h,
                  full_dst->image, 0, 0, full_w, full_h, VK_FILTER_LINEAR);
    dis_barrier(cmd, full_dst->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    dis_barrier(cmd, flow_dst->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    dis_blit_rect(cmd, source, cx, cy, full_w, full_h,
                  flow_dst->image, 0, 0, w, h, VK_FILTER_LINEAR);
    for (uint32_t l = 1; l < L; l++) {
        dis_barrier(cmd, flow_dst->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        dis_blit_mip(cmd, flow_dst->image, l - 1, l, w >> (l - 1), h >> (l - 1), w >> l, h >> l);
    }
    dis_barrier(cmd, flow_dst->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    d->next_idx = (int)slot;
    d->prev_idx = (int)((slot + DIS_SLOTS - 1u) % DIS_SLOTS);
    d->active_slot = slot;
    d->frame_count++;

    for (uint32_t l = 0; l < L; l++) {
        const uint32_t lw = w >> l;
        const uint32_t lh = h >> l;
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_luma.pipeline);
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipeline_layout, 0, 1,
                                  &d->luma_sets[slot][l], 0, NULL);
        vkd.CmdDispatch(cmd, (lw + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE,
                        (lh + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE, 1);
    }

    const bool wants_flow = generations > 0 || d->debug_flow;
    if (d->me && flush && wants_flow) {
        cmd = dis_hardware_motion(d, cmd, slot, flush, flush_user);
    } else if (d->me) {
        // Without this pair's estimate the held frame no longer precedes the next one.
        dis_qcom_me_invalidate(d->me);
    }

    dis_compute_barrier(cmd);
    DIS_PROF(cmd, "copy+luma");

    if (!wants_flow) return cmd;
    DisGradientPC gpc;
    gpc.lesser = 3.0f;
    gpc.upper = 10.0f;
    gpc.normVal = 1.0f / (2.0f * 10.0f + 4.0f * 3.0f);

    for (uint32_t l = 0; l < L; l++) {
        const uint32_t lw = w >> l;
        const uint32_t lh = h >> l;
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_gradient.pipeline);
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipeline_layout, 0, 1,
                                  &d->grad_sets[slot][l], 0, NULL);
        vkd.CmdPushConstants(cmd, d->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpc), &gpc);
        vkd.CmdDispatch(cmd, (lw + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE,
                        (lh + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE, 1);
    }

    dis_compute_barrier(cmd);
    DIS_PROF(cmd, "gradient");

    static const char* const k_search[DIS_MAX_LEVELS] = {"L0 search", "L1 search", "L2 search", "L3 search", "L4 search", "L5 search", "L6 search", "L7 search"};
    static const char* const k_vr[DIS_MAX_LEVELS] = {"L0 refine", "L1 refine", "L2 refine", "L3 refine", "L4 refine", "L5 refine", "L6 refine", "L7 refine"};
    static const char* const k_inv[DIS_MAX_LEVELS] = {"L0 inverse", "L1 inverse", "L2 inverse", "L3 inverse", "L4 inverse", "L5 inverse", "L6 inverse", "L7 inverse"};
    static const char* const k_prop[DIS_MAX_LEVELS] = {"L0 propagate", "L1 propagate", "L2 propagate", "L3 propagate", "L4 propagate", "L5 propagate", "L6 propagate", "L7 propagate"};
    (void)k_search; (void)k_vr; (void)k_inv; (void)k_prop;
    for (uint32_t li = 0; li < L; li++) {
        const uint32_t l = coarse - li;
        // The hardware field already stands in for this level and everything above it.
        if (d->me_primary_frame && l >= d->me_level) continue;
        const uint32_t lw = w >> l;
        const uint32_t lh = h >> l;
        const uint32_t spw = dis_sparse_extent(lw);
        const uint32_t sph = dis_sparse_extent(lh);

        DisInversePC ipc;
        ipc.level = (int)l;
        ipc.coarseLevel = (int)coarse;
        ipc.hintLevel = d->hint_level;
        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_inverse.pipeline);
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipeline_layout, 0, 1,
                                  &d->inverse_sets[slot][l], 0, NULL);
        vkd.CmdPushConstants(cmd, d->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ipc), &ipc);
        vkd.CmdDispatch(cmd, (spw + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE,
                        (sph + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE, 1);

        dis_compute_barrier(cmd);
        DIS_PROF(cmd, k_inv[l]);

        uint32_t prop_passes = dis_prop_steps_for(l, L, refine.prop_floor);
        const uint32_t prop_doubling = prop_passes;
        if (prop_passes & 1u) prop_passes++;

        for (uint32_t k = 0; k < prop_passes; k++) {
            VkDescriptorSet prop_set =
                (k & 1u) ? d->prop_ba_sets[slot][l] : d->prop_ab_sets[slot][l];
            DisPropPC ppc;
            ppc.dist = k < prop_doubling ? (int)(1u << k) : 1;
            vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_propagate.pipeline);
            vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipeline_layout, 0,
                                      1, &prop_set, 0, NULL);
            vkd.CmdPushConstants(cmd, d->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(ppc), &ppc);
            vkd.CmdDispatch(cmd, (spw + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE,
                            (sph + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE, 1);

            dis_compute_barrier(cmd);
        }
        DIS_PROF(cmd, k_prop[l]);

        dis_dispatch(d, cmd, d->pass_densify.pipeline, d->densify_sets[slot][l], lw, lh);

        dis_compute_barrier(cmd);
        DIS_PROF(cmd, k_search[l]);

        if (l < refine.vr_levels) {
            dis_vr_level(d, cmd, slot, l, lw, lh, &refine, true);
            DIS_PROF(cmd, k_vr[l]);
        }
    }

    if (generations > 0 || d->debug_flow) {
        const uint32_t hist_w = full_w > 1u ? full_w / 2u : 1u;
        const uint32_t hist_h = full_h > 1u ? full_h / 2u : 1u;
        const uint32_t hist_dir = 1u - d->hist_parity;
        const int hist_reset = d->hist_valid ? 0 : 1;

        // The three per-pair products the interpolate pass reads are independent of each
        // other - the packed field and the side map read the refined field, the overlay
        // history reads the colour pair - so they share one barrier.
        dis_dispatch(d, cmd, d->pass_pack.pipeline, d->pack_set, w, h);

        vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_hist.pipeline);
        vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->vr_pipeline_layout, 0, 1,
                                  &d->hist_sets[slot][hist_dir], 0, NULL);
        vkd.CmdPushConstants(cmd, d->vr_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(hist_reset), &hist_reset);
        vkd.CmdDispatch(cmd, (hist_w + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE,
                        (hist_h + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE, 1);
        d->hist_parity = hist_dir;
        d->hist_valid = true;

        // Which real frame a true occlusion takes, and how sure, per level-0 texel.
        dis_dispatch(d, cmd, d->pass_side.pipeline, d->side_sets[slot], w, h);
        dis_compute_barrier(cmd);
        DIS_PROF(cmd, "pack+hist+side");
    }
    return cmd;
}

void vkr_dis_process(VkrDis* d, VkCommandBuffer cmd, VkImage source, uint32_t width,
                     uint32_t height, uint32_t generations) {
    (void)vkr_dis_process_ex(d, cmd, source, width, height, generations, NULL, NULL);
}

static void dis_render_into(VkrDis* d, VkCommandBuffer cmd, float t, int debug_mode,
                            VkImage target_image, uint32_t width, uint32_t height,
                            VkImage base_image) {
    const uint32_t w = d->content.width;
    const uint32_t h = d->content.height;

    DisInterpPC ipc;
    ipc.t = t;
    ipc.debugMode = debug_mode;

    vkd.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pass_interp.pipeline);
    vkd.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipeline_layout, 0, 1,
                              &d->interp_sets[d->active_slot][d->hist_parity], 0, NULL);
    vkd.CmdPushConstants(cmd, d->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ipc), &ipc);
    vkd.CmdDispatch(cmd, (w + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE,
                    (h + DIS_LOCAL_SIZE - 1) / DIS_LOCAL_SIZE, 1);
    DIS_PROF(cmd, "interp shader");

    dis_barrier(cmd, d->interp_out.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    dis_barrier(cmd, target_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT);

    const uint32_t comp_w = d->built_full_extent.width;
    const uint32_t comp_h = d->built_full_extent.height;
    const float tsx = comp_w > 0 ? (float)width / (float)comp_w : 1.0f;
    const float tsy = comp_h > 0 ? (float)height / (float)comp_h : 1.0f;
    const int32_t tx = (int32_t)((float)d->content.x * tsx + 0.5f);
    const int32_t ty = (int32_t)((float)d->content.y * tsy + 0.5f);
    uint32_t tw = (uint32_t)((float)w * tsx + 0.5f);
    uint32_t th = (uint32_t)((float)h * tsy + 0.5f);
    if (tw == 0) tw = 1;
    if (th == 0) th = 1;
    if ((uint32_t)tx + tw > width) tw = width - (uint32_t)tx;
    if ((uint32_t)ty + th > height) th = height - (uint32_t)ty;

    if (base_image != VK_NULL_HANDLE) {
        const int32_t cx = d->content.x;
        const int32_t cy = d->content.y;
        const int32_t cr = cx + (int32_t)w;
        const int32_t cb = cy + (int32_t)h;
        const int32_t trx = tx + (int32_t)tw;
        const int32_t tby = ty + (int32_t)th;
        const int32_t src[4][4] = {
            {0,  0,  cx,                    (int32_t)comp_h},
            {cr, 0,  (int32_t)comp_w - cr,  (int32_t)comp_h},
            {cx, 0,  (int32_t)w,            cy},
            {cx, cb, (int32_t)w,            (int32_t)comp_h - cb},
        };
        const int32_t dst[4][4] = {
            {0,   0,   tx,                     (int32_t)height},
            {trx, 0,   (int32_t)width - trx,   (int32_t)height},
            {tx,  0,   (int32_t)tw,            ty},
            {tx,  tby, (int32_t)tw,            (int32_t)height - tby},
        };
        bool any_strip = false;
        for (uint32_t i = 0; i < 4; i++) {
            if (src[i][2] <= 0 || src[i][3] <= 0 || dst[i][2] <= 0 || dst[i][3] <= 0) continue;
            dis_blit_rect(cmd, base_image, src[i][0], src[i][1],
                          (uint32_t)src[i][2], (uint32_t)src[i][3],
                          target_image, dst[i][0], dst[i][1],
                          (uint32_t)dst[i][2], (uint32_t)dst[i][3], VK_FILTER_LINEAR);
            any_strip = true;
        }
        if (any_strip) {
            dis_barrier(cmd, target_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        }
    }

    dis_blit_rect(cmd, d->interp_out.image, 0, 0, w, h,
                  target_image, tx, ty, tw, th, VK_FILTER_LINEAR);

    dis_barrier(cmd, d->interp_out.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    DIS_PROF(cmd, "interp blit");
}

void vkr_dis_generate_into(VkrDis* d, VkCommandBuffer cmd, uint32_t generation,
                           uint32_t target_index, VkImage target_image, VkImageView target_view,
                           uint32_t width, uint32_t height, VkImage base_image) {
    (void)target_index;
    (void)target_view;
    if (!d || !d->built || d->unavailable) return;
    if (d->last_generations == 0) return;

    const float t = (float)(generation + 1) / (float)(d->last_generations + 1);
    dis_render_into(d, cmd, t, d->debug_flow ? 1 : 0, target_image, width, height, base_image);
}

void vkr_dis_debug_into(VkrDis* d, VkCommandBuffer cmd, VkImage target_image, uint32_t width,
                        uint32_t height) {
    if (!d || !d->built || d->unavailable || !d->debug_flow) return;
    if (d->frame_count < 2) return;
    dis_render_into(d, cmd, 0.5f, 1, target_image, width, height, VK_NULL_HANDLE);
}

void vkr_dis_forget_targets(VkrDis* d) {
    (void)d;
}

void vkr_dis_reset(VkrDis* d) {
    if (!d) return;
    d->frame_count = 0;
    d->last_generations = 0;
    d->prev_idx = 0;
    d->next_idx = 0;
    d->active_slot = 0;
    d->src_sample_ns = 0;
    d->src_last_frames = 0;
    d->src_frame_accum = 0.0f;
    d->src_time_accum = 0.0f;
    d->src_interval = 0.0f;
    d->src_samples = 0;
    d->smoothed_desired = 0.0f;
    d->planned_gen = 0;
    d->gen_high_streak = 0;
    d->gen_low_streak = 0;
    d->plan_log_gen = -1;
    d->plan_log_ns = 0;
    d->hist_parity = 0;
    d->hist_valid = false;
    d->hint_level = -1;
    if (d->me) dis_qcom_me_invalidate(d->me);
}

void vkr_dis_set_hw_motion(VkrDis* d, bool enabled) {
    if (!d || d->hw_motion == enabled) return;
    d->hw_motion = enabled;
    // Takes effect at the next resource build; force one.
    d->built = false;
}

bool vkr_dis_hw_motion_active(const VkrDis* d) {
    return d && d->me != NULL;
}
