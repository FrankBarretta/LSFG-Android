#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <jni.h>
#include <stdlib.h>
#include <string.h>

#include "fg_present.h"

#define FG_FN(name) Java_com_winlator_cmod_shared_framegen_FrameGenNative_##name

#define FG_DIS_QUALITY_TAG 1000

static void fg_split_quality(jint quality, uint32_t* engine, float* flow, uint32_t* dis_min_side) {
    if (quality >= FG_DIS_QUALITY_TAG) {
        *engine = FG_ENGINE_DIS;
        *flow = 0.7f;
        const jint side = quality - FG_DIS_QUALITY_TAG;
        *dis_min_side = side > 0 ? (uint32_t)side : FG_DIS_MIN_SIDE_DEFAULT;
        return;
    }
    *engine = FG_ENGINE_LSFG;
    *flow = quality > 0 ? (float)quality / 100.0f : 0.7f;
    *dis_min_side = FG_DIS_MIN_SIDE_DEFAULT;
}

static char* fg_copy_utf(JNIEnv* env, jstring value) {
    if (!value) return NULL;
    const char* chars = (*env)->GetStringUTFChars(env, value, NULL);
    if (!chars) return NULL;
    char* copy = strdup(chars);
    (*env)->ReleaseStringUTFChars(env, value, chars);
    return copy;
}

JNIEXPORT jlong JNICALL FG_FN(nativeCreate)(JNIEnv* env, jclass clazz, jobject context,
                                            jobject output, jint width, jint height,
                                            jstring cachePath, jstring driverName, jint multiplier,
                                            jint targetRate, jint flowScale, jfloat refreshRate,
                                            jfloat sourceRate) {
    (void)clazz;
    if (!output || width <= 0 || height <= 0) return 0;

    ANativeWindow* window = ANativeWindow_fromSurface(env, output);
    if (!window) return 0;

    char* cache = fg_copy_utf(env, cachePath);
    char* driver = fg_copy_utf(env, driverName);

    uint32_t engine = FG_ENGINE_LSFG;
    float flow = 0.7f;
    uint32_t dis_min_side = FG_DIS_MIN_SIDE_DEFAULT;
    fg_split_quality(flowScale, &engine, &flow, &dis_min_side);

    FgPresenter* fg = fg_create(env, context, driver, window, (uint32_t)width, (uint32_t)height,
                                cache, (uint32_t)multiplier, (uint32_t)targetRate, flow,
                                refreshRate, sourceRate, engine, dis_min_side);

    ANativeWindow_release(window);
    free(cache);
    free(driver);
    return (jlong)(intptr_t)fg;
}

JNIEXPORT jobject JNICALL FG_FN(nativeProducerSurface)(JNIEnv* env, jclass clazz, jlong handle) {
    (void)clazz;
    FgPresenter* fg = (FgPresenter*)(intptr_t)handle;
    ANativeWindow* window = fg_producer_window(fg);
    if (!window) return NULL;
    return ANativeWindow_toSurface(env, window);
}

JNIEXPORT void JNICALL FG_FN(nativeConfigure)(JNIEnv* env, jclass clazz, jlong handle,
                                              jint multiplier, jint targetRate, jint flowScale,
                                              jfloat refreshRate, jfloat sourceRate) {
    (void)env;
    (void)clazz;
    FgPresenter* fg = (FgPresenter*)(intptr_t)handle;
    uint32_t engine = FG_ENGINE_LSFG;
    float flow = 0.7f;
    uint32_t dis_min_side = FG_DIS_MIN_SIDE_DEFAULT;
    fg_split_quality(flowScale, &engine, &flow, &dis_min_side);
    fg_configure(fg, (uint32_t)multiplier, (uint32_t)targetRate, flow, refreshRate, sourceRate,
                 engine, dis_min_side);
}

JNIEXPORT jlong JNICALL FG_FN(nativeRealFrames)(JNIEnv* env, jclass clazz, jlong handle) {
    (void)env;
    (void)clazz;
    uint64_t real = 0;
    fg_stats((FgPresenter*)(intptr_t)handle, &real, NULL);
    return (jlong)real;
}

JNIEXPORT jlong JNICALL FG_FN(nativeGeneratedFrames)(JNIEnv* env, jclass clazz, jlong handle) {
    (void)env;
    (void)clazz;
    uint64_t generated = 0;
    fg_stats((FgPresenter*)(intptr_t)handle, NULL, &generated);
    return (jlong)generated;
}

JNIEXPORT void JNICALL FG_FN(nativeDestroy)(JNIEnv* env, jclass clazz, jlong handle) {
    (void)env;
    (void)clazz;
    fg_destroy((FgPresenter*)(intptr_t)handle);
}
