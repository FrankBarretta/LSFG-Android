#include <jni.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "lsfg/lsfg_dll.h"

#define CACHE_FN(name) Java_app_framegen_LsfgCache_##name

static char* copy_utf(JNIEnv* env, jstring value) {
    if (!value) return NULL;
    const char* chars = (*env)->GetStringUTFChars(env, value, NULL);
    if (!chars) return NULL;
    char* copy = strdup(chars);
    (*env)->ReleaseStringUTFChars(env, value, chars);
    return copy;
}

/* LsfgStatus: 0 ok, 1 not installed, 2 unreadable, 3 not PE, 4 missing shaders, 5 translation failed */
JNIEXPORT jint JNICALL CACHE_FN(nativeValidateDll)(JNIEnv* env, jclass clazz, jstring dll) {
    (void)clazz;
    char* path = copy_utf(env, dll);
    if (!path) return LSFG_NOT_INSTALLED;
    jint st = (jint)lsfg_validate_dll(path);
    free(path);
    return st;
}

JNIEXPORT jint JNICALL CACHE_FN(nativeBuildCache)(JNIEnv* env, jclass clazz, jstring dll,
                                                  jstring cache, jboolean preferFp16) {
    (void)clazz;
    char* d = copy_utf(env, dll);
    char* c = copy_utf(env, cache);
    jint st = LSFG_NOT_INSTALLED;
    if (d && c) st = (jint)lsfg_build_cache(d, c, preferFp16 == JNI_TRUE);
    free(d);
    free(c);
    return st;
}

/* 0 ok when every shader module loads from the cache file */
JNIEXPORT jint JNICALL CACHE_FN(nativeInspectCache)(JNIEnv* env, jclass clazz, jstring cache) {
    (void)clazz;
    char* c = copy_utf(env, cache);
    if (!c) return LSFG_NOT_INSTALLED;
    LsfgModuleSet set;
    jint st = (jint)lsfg_load_modules(c, &set);
    if (st == LSFG_OK) lsfg_release_modules(&set);
    free(c);
    return st;
}

/* 1 = FP16, 2 = FP32, 3 = DXBC-translated, 0 = cache unusable */
JNIEXPORT jint JNICALL CACHE_FN(nativeCacheVariant)(JNIEnv* env, jclass clazz, jstring cache) {
    (void)clazz;
    char* c = copy_utf(env, cache);
    if (!c) return LSFG_VARIANT_NONE;
    LsfgModuleSet set;
    jint variant = LSFG_VARIANT_NONE;
    if (lsfg_load_modules(c, &set) == LSFG_OK) {
        variant = (jint)set.variant;
        lsfg_release_modules(&set);
    }
    free(c);
    return variant;
}
