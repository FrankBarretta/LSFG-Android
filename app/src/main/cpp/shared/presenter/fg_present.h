#pragma once

#include <android/native_window.h>
#include <jni.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FgPresenter FgPresenter;

typedef enum FgEngine {
    FG_ENGINE_LSFG = 0,
    FG_ENGINE_DIS = 1,
} FgEngine;

#define FG_DIS_MIN_SIDE_DEFAULT 180u

FgPresenter* fg_create(JNIEnv* env, jobject context, const char* driver_name,
                       ANativeWindow* output, uint32_t width, uint32_t height,
                       const char* cache_path, uint32_t multiplier, uint32_t target_rate,
                       float flow_scale, float refresh_rate, float source_rate,
                       uint32_t engine, uint32_t dis_min_side);

ANativeWindow* fg_producer_window(FgPresenter* fg);

void fg_configure(FgPresenter* fg, uint32_t multiplier, uint32_t target_rate, float flow_scale,
                  float refresh_rate, float source_rate, uint32_t engine, uint32_t dis_min_side);

void fg_stats(FgPresenter* fg, uint64_t* real_frames, uint64_t* generated_frames);

void fg_destroy(FgPresenter* fg);

#ifdef __cplusplus
}
#endif
