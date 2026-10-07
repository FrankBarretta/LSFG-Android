// SPDX-FileCopyrightText: Copyright 2026 qwertypower (DEVAR Entertainment LLC)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Hardware motion estimation through GL_QCOM_motion_estimation. See dis_qcom_me.h.
//
// The estimator lives in the platform GLES driver, while DIS runs on whichever Vulkan driver the
// compositor loaded (often Turnip), so nothing is shared between the two APIs: the luminance goes
// in with a texture upload and the field comes back with a read, both a few hundred kilobytes at
// most. EGL and GLES are opened with dlopen so the libraries linking DIS take no hard dependency
// on them, and each instance owns a surfaceless context that is made current only for the
// duration of a call, restoring whatever the calling thread had current before.

#include "dis_qcom_me.h"

#ifdef __ANDROID__

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#include <android/log.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define ME_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "VkrDis", __VA_ARGS__)
#define ME_LOGW(...) __android_log_print(ANDROID_LOG_WARN, "VkrDis", __VA_ARGS__)

#define GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM 0x8C90
#define GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM 0x8C91

// The NDK's EGL headers only declare the core entry points as prototypes, so their pointer types
// are spelled out here for the dlsym'd table below.
typedef EGLDisplay (*MeGetDisplay)(EGLNativeDisplayType);
typedef EGLBoolean (*MeInitialize)(EGLDisplay, EGLint*, EGLint*);
typedef EGLBoolean (*MeChooseConfig)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*);
typedef EGLBoolean (*MeBindAPI)(EGLenum);
typedef EGLContext (*MeCreateContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint*);
typedef EGLBoolean (*MeDestroyContext)(EGLDisplay, EGLContext);
typedef EGLSurface (*MeCreatePbufferSurface)(EGLDisplay, EGLConfig, const EGLint*);
typedef EGLBoolean (*MeDestroySurface)(EGLDisplay, EGLSurface);
typedef EGLBoolean (*MeMakeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
typedef EGLContext (*MeGetCurrentContext)(void);
typedef EGLDisplay (*MeGetCurrentDisplay)(void);
typedef EGLSurface (*MeGetCurrentSurface)(EGLint);
typedef const char* (*MeQueryString)(EGLDisplay, EGLint);
typedef void (*(*MeGetProcAddress)(const char*))(void);
typedef struct {
    bool loaded;
    bool ok;
    MeGetDisplay GetDisplay;
    MeInitialize Initialize;
    MeChooseConfig ChooseConfig;
    MeBindAPI BindAPI;
    MeCreateContext CreateContext;
    MeDestroyContext DestroyContext;
    MeCreatePbufferSurface CreatePbufferSurface;
    MeDestroySurface DestroySurface;
    MeMakeCurrent MakeCurrent;
    MeGetCurrentContext GetCurrentContext;
    MeGetCurrentDisplay GetCurrentDisplay;
    MeGetCurrentSurface GetCurrentSurface;
    MeQueryString QueryString;
    MeGetProcAddress GetProcAddress;

    const GLubyte* (*GetString)(GLenum);
    void (*GetIntegerv)(GLenum, GLint*);
    GLenum (*GetError)(void);
    void (*GenTextures)(GLsizei, GLuint*);
    void (*DeleteTextures)(GLsizei, const GLuint*);
    void (*BindTexture)(GLenum, GLuint);
    void (*TexStorage2D)(GLenum, GLsizei, GLenum, GLsizei, GLsizei);
    void (*TexSubImage2D)(GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void*);
    void (*TexParameteri)(GLenum, GLenum, GLint);
    void (*PixelStorei)(GLenum, GLint);
    void (*GenFramebuffers)(GLsizei, GLuint*);
    void (*DeleteFramebuffers)(GLsizei, const GLuint*);
    void (*BindFramebuffer)(GLenum, GLuint);
    void (*FramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint);
    GLenum (*CheckFramebufferStatus)(GLenum);
    void (*ReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);
    void (*EstimateMotion)(GLuint, GLuint, GLuint);
} MeApi;

static MeApi g_api;
static pthread_mutex_t g_api_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_probe = -1;  // -1 not probed, 0 unsupported, 1 supported
static uint32_t g_block_x, g_block_y;

struct DisQcomMe {
    EGLDisplay display;
    EGLContext context;
    EGLSurface surface;  // EGL_NO_SURFACE when surfaceless
    uint32_t width, height;
    uint32_t field_w, field_h;
    GLuint luma[2];
    GLuint field;
    GLuint fbo;
    uint32_t newest;  // index of the texture holding the newest frame
    bool have_prev;
};

static bool me_load_api(void) {
    if (g_api.loaded) return g_api.ok;
    g_api.loaded = true;
    void* egl = dlopen("libEGL.so", RTLD_NOW | RTLD_LOCAL);
    void* gles = dlopen("libGLESv3.so", RTLD_NOW | RTLD_LOCAL);
    if (!egl || !gles) return false;
#define EGLFN(field, name) g_api.field = (void*)dlsym(egl, name); if (!g_api.field) return false
#define GLFN(field, name) g_api.field = (void*)dlsym(gles, name); if (!g_api.field) return false
    EGLFN(GetDisplay, "eglGetDisplay");
    EGLFN(Initialize, "eglInitialize");
    EGLFN(ChooseConfig, "eglChooseConfig");
    EGLFN(BindAPI, "eglBindAPI");
    EGLFN(CreateContext, "eglCreateContext");
    EGLFN(DestroyContext, "eglDestroyContext");
    EGLFN(CreatePbufferSurface, "eglCreatePbufferSurface");
    EGLFN(DestroySurface, "eglDestroySurface");
    EGLFN(MakeCurrent, "eglMakeCurrent");
    EGLFN(GetCurrentContext, "eglGetCurrentContext");
    EGLFN(GetCurrentDisplay, "eglGetCurrentDisplay");
    EGLFN(GetCurrentSurface, "eglGetCurrentSurface");
    EGLFN(QueryString, "eglQueryString");
    EGLFN(GetProcAddress, "eglGetProcAddress");
    GLFN(GetString, "glGetString");
    GLFN(GetIntegerv, "glGetIntegerv");
    GLFN(GetError, "glGetError");
    GLFN(GenTextures, "glGenTextures");
    GLFN(DeleteTextures, "glDeleteTextures");
    GLFN(BindTexture, "glBindTexture");
    GLFN(TexStorage2D, "glTexStorage2D");
    GLFN(TexSubImage2D, "glTexSubImage2D");
    GLFN(TexParameteri, "glTexParameteri");
    GLFN(PixelStorei, "glPixelStorei");
    GLFN(GenFramebuffers, "glGenFramebuffers");
    GLFN(DeleteFramebuffers, "glDeleteFramebuffers");
    GLFN(BindFramebuffer, "glBindFramebuffer");
    GLFN(FramebufferTexture2D, "glFramebufferTexture2D");
    GLFN(CheckFramebufferStatus, "glCheckFramebufferStatus");
    GLFN(ReadPixels, "glReadPixels");
#undef EGLFN
#undef GLFN
    g_api.EstimateMotion = (void*)g_api.GetProcAddress("glTexEstimateMotionQCOM");
    g_api.ok = g_api.EstimateMotion != NULL;
    return g_api.ok;
}

typedef struct {
    EGLDisplay display;
    EGLContext context;
    EGLSurface draw, read;
} MeSaved;

static void me_save(MeSaved* s) {
    s->display = g_api.GetCurrentDisplay();
    s->context = g_api.GetCurrentContext();
    s->draw = g_api.GetCurrentSurface(EGL_DRAW);
    s->read = g_api.GetCurrentSurface(EGL_READ);
}

static void me_restore(const MeSaved* s, EGLDisplay own) {
    if (s->context != EGL_NO_CONTEXT && s->display != EGL_NO_DISPLAY) {
        g_api.MakeCurrent(s->display, s->draw, s->read, s->context);
    } else {
        g_api.MakeCurrent(own, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }
}

// Display, context and (only without surfaceless support) a 1x1 pbuffer.
static bool me_context(EGLDisplay* out_dpy, EGLContext* out_ctx, EGLSurface* out_surf) {
    EGLDisplay dpy = g_api.GetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY || !g_api.Initialize(dpy, NULL, NULL)) return false;
    const char* ext = g_api.QueryString(dpy, EGL_EXTENSIONS);
    const bool surfaceless = ext && strstr(ext, "EGL_KHR_surfaceless_context") != NULL;
    const EGLint cfg_attr[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
                               EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
    EGLConfig cfg;
    EGLint n = 0;
    if (!g_api.ChooseConfig(dpy, cfg_attr, &cfg, 1, &n) || n < 1) return false;
    g_api.BindAPI(EGL_OPENGL_ES_API);
    const EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    EGLContext ctx = g_api.CreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attr);
    if (ctx == EGL_NO_CONTEXT) return false;
    EGLSurface surf = EGL_NO_SURFACE;
    if (!surfaceless) {
        const EGLint pb_attr[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        surf = g_api.CreatePbufferSurface(dpy, cfg, pb_attr);
        if (surf == EGL_NO_SURFACE) {
            g_api.DestroyContext(dpy, ctx);
            return false;
        }
    }
    *out_dpy = dpy;
    *out_ctx = ctx;
    *out_surf = surf;
    return true;
}

bool dis_qcom_me_supported(uint32_t* block_x, uint32_t* block_y) {
    pthread_mutex_lock(&g_api_lock);
    if (g_probe < 0) {
        g_probe = 0;
        EGLDisplay dpy;
        EGLContext ctx;
        EGLSurface surf;
        if (me_load_api() && me_context(&dpy, &ctx, &surf)) {
            MeSaved saved;
            me_save(&saved);
            if (g_api.MakeCurrent(dpy, surf, surf, ctx)) {
                const char* ext = (const char*)g_api.GetString(GL_EXTENSIONS);
                GLint bx = 0, by = 0;
                if (ext && strstr(ext, "GL_QCOM_motion_estimation")) {
                    g_api.GetIntegerv(GL_MOTION_ESTIMATION_SEARCH_BLOCK_X_QCOM, &bx);
                    g_api.GetIntegerv(GL_MOTION_ESTIMATION_SEARCH_BLOCK_Y_QCOM, &by);
                }
                if (bx > 0 && by > 0) {
                    g_block_x = (uint32_t)bx;
                    g_block_y = (uint32_t)by;
                    g_probe = 1;
                }
                ME_LOGI("GL_QCOM_motion_estimation: %s (block %dx%d, %s)",
                        g_probe ? "available" : "not available", bx, by,
                        (const char*)g_api.GetString(GL_RENDERER));
            }
            me_restore(&saved, dpy);
            if (surf != EGL_NO_SURFACE) g_api.DestroySurface(dpy, surf);
            g_api.DestroyContext(dpy, ctx);
        }
    }
    const bool ok = g_probe == 1;
    if (ok) {
        if (block_x) *block_x = g_block_x;
        if (block_y) *block_y = g_block_y;
    }
    pthread_mutex_unlock(&g_api_lock);
    return ok;
}

static GLuint me_texture(GLenum format, uint32_t w, uint32_t h) {
    GLuint t = 0;
    g_api.GenTextures(1, &t);
    g_api.BindTexture(GL_TEXTURE_2D, t);
    g_api.TexStorage2D(GL_TEXTURE_2D, 1, format, (GLsizei)w, (GLsizei)h);
    g_api.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    g_api.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

DisQcomMe* dis_qcom_me_create(uint32_t width, uint32_t height) {
    uint32_t bx = 0, by = 0;
    if (!dis_qcom_me_supported(&bx, &by)) return NULL;
    if (width == 0 || height == 0 || width % bx || height % by) return NULL;

    DisQcomMe* me = (DisQcomMe*)calloc(1, sizeof(DisQcomMe));
    if (!me) return NULL;
    me->width = width;
    me->height = height;
    me->field_w = width / bx;
    me->field_h = height / by;
    if (!me_context(&me->display, &me->context, &me->surface)) {
        free(me);
        return NULL;
    }

    MeSaved saved;
    me_save(&saved);
    bool ok = g_api.MakeCurrent(me->display, me->surface, me->surface, me->context);
    if (ok) {
        me->luma[0] = me_texture(GL_R8, width, height);
        me->luma[1] = me_texture(GL_R8, width, height);
        me->field = me_texture(GL_RGBA16F, me->field_w, me->field_h);
        g_api.GenFramebuffers(1, &me->fbo);
        g_api.BindFramebuffer(GL_FRAMEBUFFER, me->fbo);
        g_api.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, me->field, 0);
        ok = g_api.CheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE &&
             g_api.GetError() == GL_NO_ERROR;
    }
    me_restore(&saved, me->display);
    if (!ok) {
        ME_LOGW("GL_QCOM_motion_estimation: could not set up %ux%u; using DIS alone", width, height);
        dis_qcom_me_destroy(me);
        return NULL;
    }
    ME_LOGI("GL_QCOM_motion_estimation: %ux%u luminance -> %ux%u field", width, height,
            me->field_w, me->field_h);
    return me;
}

void dis_qcom_me_destroy(DisQcomMe* me) {
    if (!me) return;
    if (me->context != EGL_NO_CONTEXT) {
        MeSaved saved;
        me_save(&saved);
        if (g_api.MakeCurrent(me->display, me->surface, me->surface, me->context)) {
            if (me->fbo) g_api.DeleteFramebuffers(1, &me->fbo);
            GLuint tex[3] = {me->luma[0], me->luma[1], me->field};
            g_api.DeleteTextures(3, tex);
        }
        me_restore(&saved, me->display);
        if (me->surface != EGL_NO_SURFACE) g_api.DestroySurface(me->display, me->surface);
        g_api.DestroyContext(me->display, me->context);
    }
    free(me);
}

void dis_qcom_me_invalidate(DisQcomMe* me) {
    if (me) me->have_prev = false;
}

bool dis_qcom_me_push(DisQcomMe* me, const uint8_t* luma, float* out_xy) {
    if (!me || !luma) return false;
    MeSaved saved;
    me_save(&saved);
    if (!g_api.MakeCurrent(me->display, me->surface, me->surface, me->context)) {
        me_restore(&saved, me->display);
        return false;
    }

    const uint32_t cur = me->have_prev ? 1u - me->newest : me->newest;
    g_api.BindTexture(GL_TEXTURE_2D, me->luma[cur]);
    g_api.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    g_api.TexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (GLsizei)me->width, (GLsizei)me->height, GL_RED,
                        GL_UNSIGNED_BYTE, luma);

    bool estimated = false;
    if (me->have_prev && out_xy) {
        g_api.EstimateMotion(me->luma[me->newest], me->luma[cur], me->field);
        const uint32_t n = me->field_w * me->field_h;
        float* rgba = (float*)malloc((size_t)n * 4 * sizeof(float));
        if (rgba) {
            g_api.BindFramebuffer(GL_FRAMEBUFFER, me->fbo);
            g_api.ReadPixels(0, 0, (GLsizei)me->field_w, (GLsizei)me->field_h, GL_RGBA, GL_FLOAT,
                             rgba);
            if (g_api.GetError() == GL_NO_ERROR) {
                for (uint32_t i = 0; i < n; i++) {
                    out_xy[i * 2] = rgba[i * 4];
                    out_xy[i * 2 + 1] = rgba[i * 4 + 1];
                }
                estimated = true;
            }
            free(rgba);
        }
    }
    me->newest = cur;
    me->have_prev = true;

    me_restore(&saved, me->display);
    return estimated;
}

#else  // !__ANDROID__: no GLES driver to ask; DIS runs alone.

bool dis_qcom_me_supported(uint32_t* block_x, uint32_t* block_y) {
    (void)block_x;
    (void)block_y;
    return false;
}
DisQcomMe* dis_qcom_me_create(uint32_t width, uint32_t height) {
    (void)width;
    (void)height;
    return NULL;
}
void dis_qcom_me_destroy(DisQcomMe* me) { (void)me; }
bool dis_qcom_me_push(DisQcomMe* me, const uint8_t* luma, float* out_xy) {
    (void)me;
    (void)luma;
    (void)out_xy;
    return false;
}
void dis_qcom_me_invalidate(DisQcomMe* me) { (void)me; }

#endif
