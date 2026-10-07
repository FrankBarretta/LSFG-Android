/* Stand-in for framegen_engine.cpp when the compositor is built on its own (standalone/, no
 * winlator tree, no dxbc). Every engine is reported unavailable; nothing is generated. */
#include "framegen_engine.h"

const void *fge_probe(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd,
                      VkFormat ring_fmt) {
    (void)gipa; (void)inst; (void)pd; (void)ring_fmt;
    return NULL;
}
void fge_device_ready(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd,
                      VkDevice dev, VkQueue queue, uint32_t qfam, int features_enabled) {
    (void)gipa; (void)inst; (void)pd; (void)dev; (void)queue; (void)qfam; (void)features_enabled;
}
int fge_caps_ok(int kind) { (void)kind; return 0; }
const char *fge_caps_reason(void) { return "frame-generation engines not built into this compositor"; }
int fge_format_ok(VkFormat fmt) { (void)fmt; return 0; }
int fge_start(int kind, const char *cache_path) { (void)kind; (void)cache_path; return -1; }
void fge_stop(void) {}
int fge_unavailable(void) { return 1; }
void fge_configure(uint32_t multiplier, float flow_scale, float refresh_hz, int flow_min_side,
                   int target_fps) {
    (void)multiplier; (void)flow_scale; (void)refresh_hz; (void)flow_min_side; (void)target_fps;
}
int fge_prepare(uint32_t w, uint32_t h, VkFormat fmt) { (void)w; (void)h; (void)fmt; return 0; }
uint32_t fge_plan(uint32_t capacity, uint64_t source_frames, float presented_rate) {
    (void)capacity; (void)source_frames; (void)presented_rate; return 0;
}
void fge_process(VkCommandBuffer cmd, VkImage source, uint32_t w, uint32_t h, uint32_t gens) {
    (void)cmd; (void)source; (void)w; (void)h; (void)gens;
}
void fge_generate(VkCommandBuffer cmd, uint32_t g, uint32_t count, VkImage img, VkImageView view,
                  uint32_t w, uint32_t h) {
    (void)cmd; (void)g; (void)count; (void)img; (void)view; (void)w; (void)h;
}
void fge_forget_targets(void) {}
void fge_telemetry(float *accepted, float *source_rate, float *thermal) {
    *accepted = 0.0f; *source_rate = 0.0f; *thermal = -1.0f;
}
const char *fge_engine_name(int kind) { return kind == 1 ? "DIS Native" : "LSFG Native"; }
const char *fge_build_info(void) { return "stub"; }
