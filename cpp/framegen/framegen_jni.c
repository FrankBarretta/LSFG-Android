/* JNI for frame generation on the Wayland backend (WaylandCompositor.nativeFrameGen* /
 * nativeSet*FrameGen* / nativeSetLsfgCachePath / nativeSetEngineTuning). Every call only stores
 * a value in framegen_bridge.c; the compositor thread picks it up on its next frame, so these
 * are safe from any thread and before the compositor is up (the app arms the engine from its
 * launch code while the SurfaceView is still coming). */
#include <jni.h>
#include "framegen_bridge.h"

JNIEXPORT void JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeSetFrameGenEngine(JNIEnv *env, jclass clazz, jint kind) {
    (void)env; (void)clazz;
    vkp_framegen_set_engine((int)kind);
}

JNIEXPORT void JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeSetFrameGenArmed(JNIEnv *env, jclass clazz,
                                                                        jboolean armed, jint multiplier) {
    (void)env; (void)clazz;
    vkp_framegen_set_armed(armed ? 1 : 0, (int)multiplier);
}

JNIEXPORT void JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeSetLsfgCachePath(JNIEnv *env, jclass clazz, jstring path) {
    (void)clazz;
    if (!path) { vkp_framegen_set_lsfg_cache_path(NULL); return; }
    const char *c = (*env)->GetStringUTFChars(env, path, NULL);
    vkp_framegen_set_lsfg_cache_path(c);
    if (c) (*env)->ReleaseStringUTFChars(env, path, c);
}

JNIEXPORT void JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeSetFrameGenTuning(JNIEnv *env, jclass clazz,
                                                                         jfloat flowScale, jfloat refreshHz) {
    (void)env; (void)clazz;
    vkp_framegen_set_tuning((float)flowScale, (float)refreshHz);
}

JNIEXPORT void JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeSetEngineTuning(JNIEnv *env, jclass clazz,
                                                                 jint flowMinSide, jint targetFps) {
    (void)env; (void)clazz;
    vkp_framegen_set_engine_tuning((int)flowMinSide, (int)targetFps);
}

JNIEXPORT jint JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeFrameGenProblem(JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return (jint)vkp_framegen_problem();
}

JNIEXPORT jstring JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeFrameGenCapsReason(JNIEnv *env, jclass clazz) {
    (void)clazz;
    const char *r = vkp_framegen_caps_reason();
    return (*env)->NewStringUTF(env, r ? r : "");
}

JNIEXPORT jfloatArray JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeFrameGenStats(JNIEnv *env, jclass clazz) {
    (void)clazz;
    float st[6] = {0.f, 0.f, 0.f, 0.f, -1.f, -1.f};
    vkp_framegen_stats(st);
    jfloatArray arr = (*env)->NewFloatArray(env, 6);
    if (arr) (*env)->SetFloatArrayRegion(env, arr, 0, 6, st);
    return arr;
}

JNIEXPORT jlong JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeFrameGenPresentedFrames(JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return (jlong)vkp_framegen_total_presented();
}

JNIEXPORT jlong JNICALL
Java_com_winlator_cmod_runtime_display_wayland_WaylandCompositor_nativeFrameGenGeneratedFrames(JNIEnv *env, jclass clazz) {
    (void)env; (void)clazz;
    return (jlong)vkp_framegen_total_generated();
}
