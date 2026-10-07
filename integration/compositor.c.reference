/*
 * The embedded compositor of the Wayland display path.
 *
 * Clients are the Wine processes of a container (winewayland.drv) and, through them,
 * Mesa's Vulkan WSI. Each process is its own client, so a Windows virtual desktop is
 * described with banner_desktop_v1: the desktop owner marks one surface as the desktop,
 * and every process reports its top-level windows' desktop positions and the Windows
 * stacking order.
 *
 * The scene is the desktop surface at the bottom, then every mapped xdg_toplevel at its
 * reported position in stacking order, each with its subsurface tree (a game's Vulkan
 * swapchain is a subsurface of its window) and wp_viewport crop/scale applied. It is
 * redrawn once per event-loop pass after any commit or layout change, stretched to the
 * Android surface.
 *
 * Input goes to the desktop surface in desktop coordinates: the Wine server routes it to
 * the window under the pointer (or the focus window) in whichever process owns it, the
 * same as winex11's virtual desktop. Without a desktop (a client that doesn't speak
 * banner_desktop_v1) the topmost window under the pointer gets it.
 */
#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>
#include <fcntl.h>
#include <time.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/timerfd.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <android/log.h>
#include <wayland-server.h>

#include "xdg-shell-server-protocol.h"
#include "linux-dmabuf-v1-server-protocol.h"
#include "sc_layer.h"
#include "ahb_swapchain.h"
#include "viewporter-server-protocol.h"
#include "desktop-v1-server-protocol.h"
#include "presentation-time-server-protocol.h"
#include "pointer-constraints-unstable-v1-server-protocol.h"
#include "relative-pointer-unstable-v1-server-protocol.h"
#include "vk_present.h"
#include "framegen_bridge.h"
#include <pthread.h>
#include "effects_chain.h"
#include "desktop_ext.h"
#include "color_mgmt.h"

#define WLOGI(...) __android_log_print(ANDROID_LOG_INFO, "WNWayland", __VA_ARGS__)
#define WLOGE(...) __android_log_print(ANDROID_LOG_ERROR, "WNWayland", __VA_ARGS__)

/* ------------------------------------------------------------------ session log
 * One readable, time-stamped file per Wayland session in Download/Wayland-logs, also
 * mirrored to logcat: which programs connect, the desktop, windows opening and closing,
 * Vulkan frames arriving, a frame-rate summary every 10 seconds, and errors. */

static FILE *g_log;
static char g_log_dir[512];     /* set by the app (nativeSetLogDir); empty = logcat only */

/* Lines logged before the session file exists are kept here and written into it, in order, the
 * moment it opens — the compositor runs on its own thread, so anything the app reports as that
 * thread starts (the display's HDR capability, for one) would otherwise only reach logcat. The
 * lock also serialises the file writes, which now come from the app's threads too. */
#define PRELOG_MAX 32
static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;
static char *g_prelog[PRELOG_MAX];
static int g_prelog_n;

void wnc_log(const char *tag, const char *fmt, ...) {
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    WLOGI("[%s] %s", tag, msg);
    struct timespec ts;
    struct tm tm;
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm);
    char line[640];
    snprintf(line, sizeof(line), "%02d:%02d:%02d.%03ld  %-9s %s\n", tm.tm_hour, tm.tm_min, tm.tm_sec,
             ts.tv_nsec / 1000000, tag, msg);
    pthread_mutex_lock(&g_log_lock);
    if (g_log) fputs(line, g_log);
    else if (g_prelog_n < PRELOG_MAX) g_prelog[g_prelog_n++] = strdup(line);
    pthread_mutex_unlock(&g_log_lock);
}

void wnc_wayland_set_log_dir(const char *dir) {
    pthread_mutex_lock(&g_log_lock);
    snprintf(g_log_dir, sizeof(g_log_dir), "%s", dir ? dir : "");
    pthread_mutex_unlock(&g_log_lock);
}

static void close_session_log(void) {
    pthread_mutex_lock(&g_log_lock);
    if (g_log) fclose(g_log);
    g_log = NULL;
    for (int i = 0; i < g_prelog_n; i++) { free(g_prelog[i]); g_prelog[i] = NULL; }
    g_prelog_n = 0;
    pthread_mutex_unlock(&g_log_lock);
}

static void open_session_log(void) {
    char dir[512], path[640], stamp[32];
    time_t now = time(NULL);
    struct tm tm;
    FILE *f;

    pthread_mutex_lock(&g_log_lock);
    snprintf(dir, sizeof(dir), "%s", g_log_dir);
    pthread_mutex_unlock(&g_log_lock);
    if (!dir[0]) return;
    localtime_r(&now, &tm);
    strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H-%M-%S", &tm);
    if (mkdir(dir, 0775) != 0 && errno != EEXIST)
        WLOGE("can't create %s: %s", dir, strerror(errno));
    snprintf(path, sizeof(path), "%s/wayland-%s.log", dir, stamp);
    if (!(f = fopen(path, "w"))) {
        WLOGE("can't open session log %s: %s", path, strerror(errno));
        return;
    }
    setvbuf(f, NULL, _IOLBF, 0);
    strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm);
    fprintf(f,
            "WinNative Wayland session\n"
            "=========================\n"
            "Started   %s\n"
            "Display   Wayland: Windows programs draw through the embedded compositor\n"
            "          No X server is used for this session.\n"
            "Log       %s\n\n"
            "Time          Area      Event\n"
            "------------  --------  -----------------------------------------------------\n",
            stamp, path);
    /* Publish the file and flush anything logged before it existed, in one critical section, so a
     * line from another thread can never land ahead of the header or be dropped between the two. */
    pthread_mutex_lock(&g_log_lock);
    for (int i = 0; i < g_prelog_n; i++) { fputs(g_prelog[i], f); free(g_prelog[i]); g_prelog[i] = NULL; }
    g_prelog_n = 0;
    g_log = f;
    pthread_mutex_unlock(&g_log_lock);
    WLOGI("session log: %s", path);
}

/* At most ~10 lines a second for chatty events (window moves), so a drag can't flood logcat. */
static int log_budget(void) {
    static struct timespec window;
    static int used;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (now.tv_sec != window.tv_sec) { window = now; used = 0; }
    return used++ < 10;
}

#ifndef BTN_LEFT
#define BTN_LEFT 0x110  /* linux/input-event-codes.h */
#endif
#ifndef BTN_RIGHT
#define BTN_RIGHT 0x111
#endif

/* Java sends pointer coordinates in this space, covering the whole output surface. */
#define INPUT_SPACE_W 1920
#define INPUT_SPACE_H 1080

static struct wl_display *g_display;

/* Which program each Wayland client is (from /proc/<pid>/cmdline), for the session log. */
struct client_info {
    struct wl_client *client;
    struct wl_listener destroy;
    pid_t pid;
    char name[64];
    /* An OpenGL program whose EGL gave up on the GPU: it asked for dma-buf feedback (EGL's
     * Wayland GPU path always does), never made a dma-buf buffer, and draws wl_shm frames. */
    unsigned asked_feedback : 1, shm_gl_said : 1;
    /* It has declared an opaque region, so where it declares none its alpha channel is meant. Programs
     * that never do (Wine's windows, whose alpha is whatever the game left there) stay opaque. */
    unsigned declares_opaque : 1;
    unsigned dmabuf_buffers, shm_frames;
    struct client_info *next;
};
static struct client_info *g_clients;

static struct client_info *client_info_of(struct wl_client *client) {
    for (struct client_info *ci = g_clients; ci; ci = ci->next)
        if (ci->client == client) return ci;
    return NULL;
}

static const char *client_name(struct wl_client *client) {
    struct client_info *ci = client_info_of(client);
    return ci ? ci->name : "a program";
}

static void on_client_destroyed(struct wl_listener *l, void *data) {
    struct client_info *ci = wl_container_of(l, ci, destroy), **pp;
    wnc_log("program", "disconnected: %s (pid %d)", ci->name, (int)ci->pid);
    wnc_color_client_gone(ci->client); /* HDR summary for a program that presented HDR (gate open only) */
    for (pp = &g_clients; *pp; pp = &(*pp)->next)
        if (*pp == ci) { *pp = ci->next; break; }
    free(ci);
}

static void on_client_created(struct wl_listener *l, void *data) {
    struct wl_client *client = data;
    struct client_info *ci = calloc(1, sizeof(*ci));
    char path[64], buf[256];
    uid_t uid; gid_t gid;
    int fd;
    ssize_t n;

    if (!ci) return;
    ci->client = client;
    wl_client_get_credentials(client, &ci->pid, &uid, &gid);
    snprintf(ci->name, sizeof(ci->name), "pid %d", (int)ci->pid);
    snprintf(path, sizeof(path), "/proc/%d/cmdline", (int)ci->pid);
    if ((fd = open(path, O_RDONLY | O_CLOEXEC)) >= 0) {
        if ((n = read(fd, buf, sizeof(buf) - 1)) > 0) {
            char *base = buf, *p;
            buf[n] = 0;  /* argv[0] only; Wine puts the Windows program path there */
            for (p = buf; *p; p++) if (*p == '\\' || *p == '/') base = p + 1;
            if (*base) snprintf(ci->name, sizeof(ci->name), "%s", base);
        }
        close(fd);
    }
    ci->destroy.notify = on_client_destroyed;
    wl_client_add_destroy_listener(client, &ci->destroy);
    ci->next = g_clients;
    g_clients = ci;
    wnc_log("program", "connected over Wayland: %s (pid %d)", ci->name, (int)ci->pid);
}

/* ------------------------------------------------------------------ surfaces */

enum surface_role { ROLE_NONE, ROLE_TOPLEVEL, ROLE_SUBSURFACE, ROLE_DESKTOP };

struct surface {
    struct wl_resource *resource;
    struct wl_list link;                    /* g_surfaces */

    /* Pending state, applied on commit. */
    struct wl_resource *pending_buffer;
    struct wl_listener pending_buffer_destroy;
    int pending_attach;
    int pending_src_set, pending_dst_set;
    float pending_src[4];                   /* x, y, w, h; w < 0 = unset */
    int pending_dst[2];                     /* w, h; w < 0 = unset */
    struct wl_list pending_frames;
    struct wl_list pending_feedback;        /* wp_presentation feedback asked for before commit */

    /* Current content. */
    struct vkp_image *shm_img;              /* our copy of the last wl_shm buffer */
    struct wl_resource *dmabuf;             /* current dmabuf wl_buffer (NULL once the client destroyed it) */
    struct dmabuf_buffer *dmabuf_buf;       /* its imported image, kept until replaced or the surface goes */
    struct wl_listener dmabuf_destroy;
    int buf_w, buf_h, has_content;
    int buf_alpha;                          /* the buffer's format has an alpha channel */
    int opaque[4], pending_opaque[4], pending_opaque_set; /* x, y, w, h; w = 0: none */
    int src_set, dst_set;
    float src[4];
    int dst[2];
    struct wl_list frames;                  /* frame callbacks for the next redraw */
    struct wl_list feedback;                /* presentation feedback for the current content */
    int drawn;                              /* part of the last rendered scene */
    int64_t next_release_ns;                /* FPS limiter: when this surface's last buffer goes back */
    int releases_pending;                   /* FPS limiter: buffers of this surface still to be released */

    enum surface_role role;
    struct wl_resource *xdg_surface, *xdg_toplevel;
    struct wl_resource *viewport;

    /* Toplevel placement on the scene. */
    int mapped;                             /* in g_toplevels */
    struct wl_list toplevel_link;
    int x, y, placed;
    uint32_t hwnd;

    /* Subsurface tree. */
    struct surface *parent;
    struct wl_resource *subsurface;
    struct wl_list children;                /* bottom to top */
    struct wl_list child_link;
    int sub_x, sub_y, sub_pending_x, sub_pending_y, sub_pending;
    int sub_sync;                           /* wl_subsurface mode: its frames reach the screen with the parent's commit */
    int hud_frame;                          /* a GPU frame arrived in this window since the HUD last counted one */
    int below_parent;
    int fullscreen;                         /* xdg_toplevel.set_fullscreen: no client decorations */

    char *title;                            /* xdg_toplevel title, for the session log */
    int announced_vulkan;                   /* logged its first dmabuf frame */
    uint32_t hdr_fmt_logged;                /* HDR session: the buffer format last named in the log */
};

static struct wl_list g_surfaces;           /* every surface */
static struct wl_list g_toplevels;          /* mapped toplevels, bottom to top */
static struct surface *g_desktop;
static uint32_t *g_zorder;                  /* last reported Windows stacking order, top first */
static size_t g_zorder_count;
static struct wl_event_source *g_render_idle, *g_frame_timer;

/* Rendering is driven by the screen's vsync (Java's Choreographer ticks, nativeVsync): one scene
 * per refresh showing the newest buffers, so the event loop is never parked in the swapchain and
 * clients get their buffers back as soon as they're replaced. Without ticks (an older app, no
 * window) a fallback timer renders instead. */
static int g_dirty;                         /* something changed since the last render */
static int64_t g_last_vsync_ns;             /* last tick, 0 = none yet */
static int64_t g_refresh_ns = 16666667;     /* screen refresh interval, from the ticks */
static struct wl_event_source *g_fallback_timer;
static int g_fallback_armed;

/* FPS limiter (the in-game drawer's): frames per second, 0 = unlimited. Set from the app thread,
 * read here. Like the X11 path, which delays the "your buffer is free" notice, the limit paces
 * when a replaced buffer is released to its client, so the game itself slows to the cap. */
volatile int g_fps_limit;

/* Shortcut launches: explorer's windows (the desktop, taskbar, Start menu) aren't drawn, matching
 * the X11 renderer's unviewable "explorer.exe". They still exist for input routing. */
volatile int g_hide_shell;

/* Experimental layer mode (BANNER_WAYLAND_ZERO_COPY=1 in the container's environment): a single
 * fullscreen window is shown on its own Android layer (sc_layer.c) instead of being blitted into the
 * screen swapchain. Set from the app before the compositor starts; read per frame. */
volatile int g_zero_copy;
static int g_zero_copy_paused;        /* effects hold the layer path off (render_scene) */
static int g_zero_copy_fx_skip_said;

/* Compressed (UBWC) game buffers: zwp_linux_dmabuf_v1 advertises DRM_FORMAT_MOD_QCOM_COMPRESSED next
 * to LINEAR for every format the renderer's own driver can import that way, so Turnip's Wayland WSI
 * in the game allocates UBWC swapchain images instead of resolving each frame to a linear copy.
 * BANNER_WAYLAND_UBWC=0 in the container's environment turns the advertisement off (A/B switch). Set
 * from the app before the compositor starts; read at every zwp_linux_dmabuf_v1 bind. */
volatile int g_ubwc = 1;

/* Debug (BANNER_WAYLAND_NO_RENDER_NODE=1): name no DRM device in the dma-buf feedback, as a phone
 * that exposes no /dev/dri node to apps does, to reproduce its OpenGL path here. Set from the app
 * before the compositor starts. */
volatile int g_no_render_node;

/* The panel's refresh rate in mHz, from the app; wl_output advertises it so Wine's display modes
 * carry the real rate (games pick their saved 144 Hz mode, as on X11). 0 = 60 Hz. */
volatile int g_output_refresh_mhz;

/* The advertised output size: the container's screen (what the app's X server reports on X11),
 * so Wine's display-mode list stops at the desktop size. 0 = 1920x1080. */
volatile int g_output_w, g_output_h;
struct pending_release {
    struct wl_resource *buffer;
    struct wl_listener destroy;
    struct surface *surface;                /* NULL once the surface is gone */
    int64_t at_ns;
    int64_t since_ns;                       /* when the compositor let go of it (the perf line) */
    struct wl_list link;
};
static struct wl_list g_pending_releases;
static int g_release_timer_fd = -1;
static struct wl_event_source *g_release_source;
static int g_scene_w, g_scene_h;            /* size of the last drawn scene */
static int g_desktop_w, g_desktop_h;        /* last size the desktop had content at */
static unsigned g_stat_frames, g_stat_dmabuf, g_stat_shm; /* since the last 10 s summary */

/* The 10 s `perf` line (on_stats_timer): where the compositor thread's time goes and how long games
 * wait for their buffers back. The driver-side times (acquire, present, fence waits) are counted in
 * vk_present.c (vkp_perf_take). Compositor thread only. */
static struct {
    unsigned ticks;                          /* screen refreshes (Choreographer ticks) */
    unsigned scenes;                         /* render_scene() calls ... */
    int64_t scene_ns, scene_max_ns;          /* ... and their wall time */
    unsigned copy_scenes;                    /* scenes drawn into the screen swapchain (the copy path) */
    unsigned releases, releases_held;        /* wl_buffer.release after a replacement; held = 1 ms or more */
    int64_t release_ns, release_max_ns;      /* replaced (or taken off the layer) -> released to the game */
} g_perf;

static void schedule_render(void);

static int64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/* A game got a replaced buffer back; since_ns = when the compositor let go of it (0 = not counted). */
static void perf_note_release(int64_t since_ns) {
    if (since_ns <= 0) return;
    int64_t d = now_ns() - since_ns;
    if (d < 0) d = 0;
    g_perf.releases++;
    g_perf.release_ns += d;
    if (d > g_perf.release_max_ns) g_perf.release_max_ns = d;
    if (d >= 1000000LL) g_perf.releases_held++;
}

/* ------------------------------------------------------------------ buffer release pacing */

static void pending_release_free(struct pending_release *pr) {
    if (pr->surface && pr->surface->releases_pending > 0) pr->surface->releases_pending--;
    wl_list_remove(&pr->destroy.link);
    wl_list_remove(&pr->link);
    free(pr);
}

/* A surface is going away: its queued releases still go out on time, they just stop counting. */
static void pending_releases_forget_surface(struct surface *s) {
    struct pending_release *pr;
    wl_list_for_each(pr, &g_pending_releases, link)
        if (pr->surface == s) pr->surface = NULL;
}

static void on_pending_release_buffer_destroyed(struct wl_listener *l, void *data) {
    struct pending_release *pr = wl_container_of(l, pr, destroy);
    pending_release_free(pr);
}

static void arm_release_timer(void) {
    struct pending_release *pr;
    int64_t earliest = 0;
    if (g_release_timer_fd < 0) return;
    wl_list_for_each(pr, &g_pending_releases, link)
        if (!earliest || pr->at_ns < earliest) earliest = pr->at_ns;
    struct itimerspec its = {0};
    if (earliest) { its.it_value.tv_sec = earliest / 1000000000LL; its.it_value.tv_nsec = earliest % 1000000000LL; }
    timerfd_settime(g_release_timer_fd, TFD_TIMER_ABSTIME, &its, NULL);
}

static int on_release_timer(int fd, uint32_t mask, void *data) {
    struct pending_release *pr, *tmp;
    uint64_t expirations;
    int64_t now = now_ns() + 300000; /* anything due within 0.3 ms goes now */
    if (read(fd, &expirations, sizeof(expirations)) < 0 && errno != EAGAIN) return 0;
    wl_list_for_each_safe(pr, tmp, &g_pending_releases, link) {
        if (pr->at_ns <= now) {
            wl_buffer_send_release(pr->buffer);
            perf_note_release(pr->since_ns);
            pending_release_free(pr);
        }
    }
    wl_display_flush_clients(g_display);
    arm_release_timer();
    return 0;
}

/* Give a replaced buffer back to its client: now, or on the limiter's cadence.
 *
 * Each release takes the next slot of a per-surface cadence (one slot per interval), so a game
 * gets one buffer back per interval however fast it commits. The schedule is kept honest at both
 * ends: a slot in the past is brought to now (the game was slower than the cap, no catch-up burst
 * is owed), and it never runs further ahead than the releases actually queued — with p releases
 * still pending the new one lands at most (p + 1) intervals out, nothing pending means at most
 * one interval. Without that bound the schedule kept slots that were consumed but never
 * delivered (a swapchain rebuild destroys buffers with queued releases) or that a coarser earlier
 * limit had spaced out, and the game's first frames after that waited on empty slots. */
static void release_buffer(struct surface *s, struct wl_resource *buffer, int64_t since_ns) {
    const int limit = g_fps_limit;
    /* Without a limit of its own a game is paced to the screen. A swapchain that does not wait for
     * frame callbacks (MAILBOX, IMMEDIATE) is held back by nothing else here, so handing a replaced
     * buffer straight back let it draw as fast as the GPU allowed and everything above the refresh
     * rate was thrown away - 178 frames a second for 91 on screen, measured. X11 never did that:
     * there a buffer comes back only once the scene that used it has been composed. One release per
     * refresh is the same back-pressure, and the screen shows the same frames either way. */
    const int64_t interval = limit > 0 ? 1000000000LL / limit : g_refresh_ns;
    if (interval <= 0 || g_release_timer_fd < 0) { wl_buffer_send_release(buffer); perf_note_release(since_ns); return; }
    int64_t now = now_ns();
    int64_t at = s->next_release_ns + interval;
    int64_t latest = now + interval * (int64_t)(s->releases_pending + 1);
    if (at < now) at = now;
    if (at > latest) at = latest;
    s->next_release_ns = at;
    struct pending_release *pr = calloc(1, sizeof(*pr));
    if (!pr) { wl_buffer_send_release(buffer); perf_note_release(since_ns); return; }
    pr->buffer = buffer;
    pr->surface = s;
    pr->at_ns = at;
    pr->since_ns = since_ns;
    pr->destroy.notify = on_pending_release_buffer_destroyed;
    wl_resource_add_destroy_listener(buffer, &pr->destroy);
    wl_list_insert(g_pending_releases.prev, &pr->link);
    s->releases_pending++;
    arm_release_timer();
}

/* ------------------------------------------------------------------ seat state */

/* Each Wine process is a separate client with its own wl_pointer / wl_keyboard. */
struct seat_pointer { struct wl_resource *ptr; struct wl_resource *focus; };
struct seat_keyboard { struct wl_resource *kb; struct wl_resource *focus; };
#define MAX_PTRS 32
static struct seat_pointer g_ptrs[MAX_PTRS];
static int g_nptrs;
static struct seat_keyboard g_kbs[MAX_PTRS];
static int g_nkbs;
static struct surface *g_grab;              /* no-desktop fallback: surface holding the button */
static struct surface *g_key_target;        /* no-desktop fallback: last clicked surface */
static struct surface *g_ime_click;         /* last clicked program window: where text input goes */
static int g_input_pipe[2] = {-1, -1};
/* type 0 = pointer (p1=action 0down/1move/2up, p2=x, p3=y); type 1 = key (p1=evdev, p2=state 1down/0up) */
struct input_msg { int type; int p1; int p2; int p3; };

/* Pointer constraints (zwp_pointer_constraints_v1) and relative pointers
 * (zwp_relative_pointer_manager_v1); see the "pointer constraints" section. */
struct surface;
static void constraints_surface_gone(struct surface *s);
static void constraints_surface_commit(struct surface *s);
static void constraints_pointer_gone(struct wl_resource *pointer);
static void relative_pointers_pointer_gone(struct wl_resource *pointer);
static void constraints_focus_entered(struct wl_resource *target, struct wl_client *client);

/* ------------------------------------------------------------------ dmabuf buffers */

#define FOURCC(a, b, c, d) \
    ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))
#define DRM_ARGB8888 FOURCC('A', 'R', '2', '4')
#define DRM_XRGB8888 FOURCC('X', 'R', '2', '4')
#define DRM_ABGR8888 FOURCC('A', 'B', '2', '4')
#define DRM_XBGR8888 FOURCC('X', 'B', '2', '4')
#define MOD_LINEAR VKP_MOD_LINEAR
#define MOD_INVALID VKP_MOD_INVALID
#define MAX_PLANES 4

struct dmabuf_params {
    int fd[MAX_PLANES];
    uint32_t offset[MAX_PLANES], stride[MAX_PLANES];
    uint64_t modifier[MAX_PLANES];
    int n_planes;
};
struct dmabuf_buffer {
    int fd[MAX_PLANES];
    uint32_t offset[MAX_PLANES], stride[MAX_PLANES];
    int n_planes;
    int32_t width, height;
    uint32_t format;
    uint64_t modifier;
    struct vkp_image *img;                  /* imported once, reused for every frame */
    int import_failed;
    /* One reference for the wl_buffer resource, one per surface showing the buffer. Mesa destroys
     * a swapchain's wl_buffers the moment the game rebuilds its swapchain, i.e. while the last
     * committed one is still what is on screen: the import (and the dma-buf memory it pins) stays
     * until the surface commits something newer, so the picture never blinks to black. */
    int refs;
    void *ahb_state;                        /* zero-copy: ahb_swapchain.c's record (the game's AHardwareBuffer) */
};

static void dmabuf_buffer_unref(struct dmabuf_buffer *b) {
    if (!b || --b->refs > 0) return;
    vkp_image_destroy(b->img);
    for (int i = 0; i < b->n_planes; i++)
        if (b->fd[i] >= 0) close(b->fd[i]);
    free(b);
}

static void dbuf_buffer_destroy_req(struct wl_client *c, struct wl_resource *r) {
    wl_resource_destroy(r);
}
static const struct wl_buffer_interface dbuf_buffer_impl = {
    .destroy = dbuf_buffer_destroy_req,
};

static struct dmabuf_buffer *get_dmabuf(struct wl_resource *buffer) {
    if (buffer && wl_resource_instance_of(buffer, &wl_buffer_interface, &dbuf_buffer_impl))
        return wl_resource_get_user_data(buffer);
    return NULL;
}

/* ------------------------------------------------------------------ surface state */

static void surface_size(const struct surface *s, int *w, int *h) {
    if (s->dst_set) { *w = s->dst[0]; *h = s->dst[1]; }
    else if (s->src_set) { *w = (int)(s->src[2] + 0.5f); *h = (int)(s->src[3] + 0.5f); }
    else { *w = s->buf_w; *h = s->buf_h; }
}

/* "Title" (program) for the log; subsurfaces are named after the window they belong to. */
static void describe(const struct surface *s, char *out, size_t size) {
    const struct surface *w = s;
    while (w->parent) w = w->parent;
    const char *prog = client_name(wl_resource_get_client(w->resource));
    if (w->title && *w->title) snprintf(out, size, "\"%s\" (%s)", w->title, prog);
    else if (w->role == ROLE_DESKTOP) snprintf(out, size, "the desktop (%s)", prog);
    else snprintf(out, size, "window %#x (%s)", w->hwnd, prog);
}

static struct surface *toplevel_by_hwnd(uint32_t hwnd) {
    struct surface *s;
    wl_list_for_each(s, &g_toplevels, toplevel_link)
        if (s->hwnd == hwnd) return s;
    return NULL;
}

/* Reorder the mapped toplevels to match the last reported Windows z-order. Windows the
 * report doesn't mention keep their relative order below the ones it does. */
static void apply_zorder(void) {
    for (size_t i = g_zorder_count; i-- > 0;) {
        struct surface *s = toplevel_by_hwnd(g_zorder[i]);
        if (!s) continue;
        wl_list_remove(&s->toplevel_link);
        wl_list_insert(g_toplevels.prev, &s->toplevel_link);
    }
}

static void map_toplevel(struct surface *s) {
    int w, h;
    if (s->mapped) return;
    s->mapped = 1;
    surface_size(s, &w, &h);
    {
        char name[160];
        describe(s, name, sizeof(name));
        wnc_log("window", "opened %s %dx%d at %d,%d%s", name, w, h, s->x, s->y,
                   s->placed ? "" : " (no desktop position yet)");
    }
    wl_list_insert(g_toplevels.prev, &s->toplevel_link); /* new windows start on top */
    apply_zorder();
}

static void unmap_toplevel(struct surface *s) {
    if (!s->mapped) return;
    s->mapped = 0;
    {
        char name[160];
        describe(s, name, sizeof(name));
        wnc_log("window", "closed %s", name);
    }
    wl_list_remove(&s->toplevel_link);
    wl_list_init(&s->toplevel_link);
    if (g_ime_click == s) g_ime_click = NULL;
    wnc_text_input_refocus();
}

/* Let go of the surface's dmabuf content. paced = 1: the buffer was replaced, give it back on the
 * limiter's cadence; paced = 0: the surface is going away, give it back at once (a buffer of a
 * destroyed surface is not shown again and must not sit unreleased with the client). */
static void drop_dmabuf(struct surface *s, int paced) {
    if (s->dmabuf) {
        wl_list_remove(&s->dmabuf_destroy.link);
        /* A buffer on the zero-copy layer is the display's until SurfaceFlinger says otherwise:
         * ahb_swapchain.c releases it then. */
        if (!ahb_swapchain_defer_release(s->dmabuf_buf, s->dmabuf, s, paced)) {
            if (paced) release_buffer(s, s->dmabuf, now_ns());
            else wl_buffer_send_release(s->dmabuf);
        }
        s->dmabuf = NULL;
    }
    dmabuf_buffer_unref(s->dmabuf_buf);
    s->dmabuf_buf = NULL;
}

/* The client destroyed the wl_buffer we are showing (a swapchain rebuild): keep showing its
 * image (s->dmabuf_buf holds it) until the next commit replaces it; only the resource is gone. */
static void on_dmabuf_destroyed(struct wl_listener *l, void *data) {
    struct surface *s = wl_container_of(l, s, dmabuf_destroy);
    wl_list_remove(&s->dmabuf_destroy.link);
    s->dmabuf = NULL;
}

static void on_pending_buffer_destroyed(struct wl_listener *l, void *data) {
    struct surface *s = wl_container_of(l, s, pending_buffer_destroy);
    wl_list_remove(&s->pending_buffer_destroy.link);
    wl_list_init(&s->pending_buffer_destroy.link);
    s->pending_buffer = NULL;
}

static void fire_frames(struct wl_list *frames) {
    struct wl_resource *cb, *tmp;
    uint32_t t;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t = (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
    wl_resource_for_each_safe(cb, tmp, frames) {
        wl_callback_send_done(cb, t);
        wl_resource_destroy(cb);
    }
}

/* Copy a committed wl_shm buffer into the surface's image and release the buffer. */
static void take_shm(struct surface *s, struct wl_shm_buffer *shm, struct wl_resource *buffer) {
    int32_t w = wl_shm_buffer_get_width(shm), h = wl_shm_buffer_get_height(shm);
    int32_t stride = wl_shm_buffer_get_stride(shm);

    if (s->shm_img && (vkp_image_width(s->shm_img) != w || vkp_image_height(s->shm_img) != h)) {
        vkp_image_destroy(s->shm_img);
        s->shm_img = NULL;
    }
    if (!s->shm_img) s->shm_img = vkp_image_create_shm(w, h);
    if (s->shm_img) {
        wl_shm_buffer_begin_access(shm);
        vkp_image_upload_shm(s->shm_img, wl_shm_buffer_get_data(shm), stride);
        wl_shm_buffer_end_access(shm);
    }
    wl_buffer_send_release(buffer);
    s->buf_w = w;
    s->buf_h = h;
    s->buf_alpha = wl_shm_buffer_get_format(shm) == WL_SHM_FORMAT_ARGB8888;
    s->has_content = s->shm_img != NULL;
    g_stat_shm++;
    struct client_info *ci = client_info_of(wl_resource_get_client(s->resource));
    if (ci && ci->asked_feedback) ci->shm_frames++;  /* since it asked for GPU buffers */
}

/* The window the app's performance HUD follows: the latest one to start presenting GPU frames
 * (X11 binds the HUD to the _MESA_DRV window and counts X presents instead). JNI upcalls.
 *
 * A synchronized subsurface belongs to its parent's frame: gamescope presents through a toplevel
 * and a plane per layer, attaches to all of them and commits the toplevel last. Following one
 * plane counts nothing once the game moves to another, so such a surface is counted as the window
 * it is part of, once per commit of that window. */
static struct surface *g_hud_surface;
static struct surface *hud_window(struct surface *s) {
    while (s->parent && s->sub_sync) s = s->parent;
    return s;
}
extern void wnc_on_game_surface(const char *window, const char *gpu); /* window NULL = gone */
extern void wnc_on_game_frame(void);
/* The program behind that window: its Linux pid (the Wayland client's credentials) and executable name
 * ("" when /proc gave none) - the app arms its CPU affinity on it (X11 does that from window events). */
extern void wnc_on_game_program(int pid, const char *program);

static void take_dmabuf(struct surface *s, struct dmabuf_buffer *b, struct wl_resource *buffer) {
    if (s->dmabuf != buffer || s->dmabuf_buf != b) {
        drop_dmabuf(s, 1);
        s->dmabuf = buffer;
        s->dmabuf_buf = b;
        b->refs++;
        s->dmabuf_destroy.notify = on_dmabuf_destroyed;
        wl_resource_add_destroy_listener(buffer, &s->dmabuf_destroy);
    }
    if (!b->img && !b->import_failed && b->n_planes >= 1) {
        b->img = vkp_image_from_dmabuf(b->fd[0], b->format, b->modifier, b->width, b->height,
                                       b->stride[0], b->offset[0]);
        if (b->img && g_zero_copy) sc_layer_probe_dmabuf_fd(b->fd[0]);
        if (!b->img) {
            b->import_failed = 1;
            WLOGE("dmabuf import failed (%dx%d fmt=0x%08x mod=0x%llx)", b->width, b->height,
                  b->format, (unsigned long long)b->modifier);
        }
    }
    if (s->shm_img) { vkp_image_destroy(s->shm_img); s->shm_img = NULL; }
    s->buf_w = b->width;
    s->buf_h = b->height;
    s->buf_alpha = (b->format & 0xff) == 'A';
    s->has_content = b->img != NULL;
    g_stat_dmabuf++;
    if (!s->announced_vulkan) {
        char name[160];
        s->announced_vulkan = 1;
        describe(s, name, sizeof(name));
        if (b->img)
            /* Imported for the copy path. Whether its frames ALSO go on the display layer without a copy
             * is decided per frame (fullscreen, zero-copy on) and counted in the 10 s lines. */
            wnc_log("vulkan", "%s is presenting GPU frames through Wayland: %dx%d, format %c%c%c%c, %s (%s)",
                       name, b->width, b->height, b->format & 0xff, (b->format >> 8) & 0xff,
                       (b->format >> 16) & 0xff, (b->format >> 24) & 0xff,
                       vkp_modifier_name(b->modifier),
                       ahb_swapchain_has_ahb(b) ? "gralloc buffers: can go on the display layer without a copy"
                                                : "dma-buf: copied into the screen swapchain");
        else if (ahb_swapchain_has_ahb(b))
            wnc_log("vulkan", "%s is presenting GPU frames through Wayland: %dx%d on its own display layer only "
                       "(gralloc buffers the compositor cannot import for the copy path)",
                       name, b->width, b->height);
        else
            wnc_log("error", "could not import GPU frames from %s (%dx%d, modifier %#llx)",
                       name, b->width, b->height, (unsigned long long)b->modifier);
        /* The HUD follows the window whether its frames are copied or go straight to the layer:
         * a zero-copy frame the compositor never imported is still a presented game frame. */
        if ((b->img || ahb_swapchain_has_ahb(b)) && g_hud_surface != hud_window(s)) {
            g_hud_surface = hud_window(s);
            g_hud_surface->hud_frame = 0;
            wnc_on_game_surface(name, vkp_gpu_name());
            struct client_info *ci = client_info_of(wl_resource_get_client(s->resource));
            if (ci) wnc_on_game_program((int)ci->pid, strncmp(ci->name, "pid ", 4) ? ci->name : "");
        }
    }
    /* HDR session (gate open): a DXVK game switches to HDR by REBUILDING its swapchain on the same
     * surface, so the one-shot line above never sees the 10-bit buffers - name every format change. */
    if (b->format != s->hdr_fmt_logged && wnc_color_hdr_open()) {
        char name[160];
        uint32_t af = ahb_swapchain_ahb_format(b);
        int ten = b->format == FOURCC('A', 'B', '3', '0') || b->format == FOURCC('X', 'B', '3', '0');
        describe(s, name, sizeof(name));
        wnc_log("color", "%s presents %dx%d buffers in %c%c%c%c (%s), %s%s%s; %s", name, b->width, b->height,
                   b->format & 0xff, (b->format >> 8) & 0xff, (b->format >> 16) & 0xff, (b->format >> 24) & 0xff,
                   ten ? "10-bit A2B10G10R10" : "8-bit", vkp_modifier_name(b->modifier),
                   af ? ", gralloc " : ", no gralloc buffer (copy path only)", af ? wnc_ahb_format_name(af) : "",
                   b->img ? "the compositor imported it"
                          : af ? "the compositor's driver could NOT import it (display layer only, fullscreen)"
                               : "the compositor's driver could NOT import it (nothing can show it)");
        s->hdr_fmt_logged = b->format;
    }
    if (hud_window(s) == g_hud_surface && (b->img || ahb_swapchain_has_ahb(b))) g_hud_surface->hud_frame = 1;
}

/* ---- hooks for ahb_swapchain.c (zero-copy layers) */
struct dmabuf_buffer *wnc_dmabuf_from_resource(struct wl_resource *buffer) { return get_dmabuf(buffer); }
int wnc_dmabuf_fd(const struct dmabuf_buffer *b) { return b && b->n_planes > 0 ? b->fd[0] : -1; }
void wnc_dmabuf_size(const struct dmabuf_buffer *b, int *w, int *h) { *w = b->width; *h = b->height; }
void **wnc_dmabuf_ahb_slot(struct dmabuf_buffer *b) { return &b->ahb_state; }
void wnc_dmabuf_ref(struct dmabuf_buffer *b) { b->refs++; }
void wnc_dmabuf_unref(struct dmabuf_buffer *b) { dmabuf_buffer_unref(b); }
void wnc_release_buffer(struct surface *s, struct wl_resource *buffer, int paced, int64_t since_ns) {
    if (paced && s) release_buffer(s, buffer, since_ns);
    else { wl_buffer_send_release(buffer); perf_note_release(since_ns); }
}
void wnc_surface_describe(const struct surface *s, char *out, size_t size) { describe(s, out, size); }
const struct wnc_color *wnc_surface_color(const struct surface *s) {
    return s ? wnc_color_of(s->resource) : NULL;
}

/* ------------------------------------------------------------------ wl_surface */

static void surface_destroy(struct wl_client *c, struct wl_resource *r) {
    wl_resource_destroy(r);
}
static void surface_attach(struct wl_client *c, struct wl_resource *r,
                           struct wl_resource *buffer, int32_t x, int32_t y) {
    struct surface *s = wl_resource_get_user_data(r);
    if (s->pending_buffer) wl_list_remove(&s->pending_buffer_destroy.link);
    wl_list_init(&s->pending_buffer_destroy.link);
    s->pending_buffer = buffer;
    s->pending_attach = 1;
    if (buffer) {
        s->pending_buffer_destroy.notify = on_pending_buffer_destroyed;
        wl_resource_add_destroy_listener(buffer, &s->pending_buffer_destroy);
    }
}
static void surface_damage(struct wl_client *c, struct wl_resource *r,
                           int32_t x, int32_t y, int32_t w, int32_t h) {}
static void frame_callback_destroy(struct wl_resource *r) {
    wl_list_remove(wl_resource_get_link(r));
}

/* wp_presentation: tells Mesa's Vulkan driver when a frame reached the screen or was replaced
 * before it did. Without it the driver waits for a frame callback (one per refresh) for every
 * frame, which paced games to the screen rate even in mailbox mode. */
static void feedback_resource_destroy(struct wl_resource *r) {
    wl_list_remove(wl_resource_get_link(r));
}

static void feedback_discard_all(struct wl_list *list) {
    struct wl_resource *fb, *tmp;
    wl_resource_for_each_safe(fb, tmp, list) {
        wp_presentation_feedback_send_discarded(fb);
        wl_resource_destroy(fb);
    }
}

/* The refresh a presentation timestamp carries is the output's nominal interval - the same value
 * every frame, matching the mode wl_output advertises. The measured one drives this compositor's
 * own pacing and drifts by a few hundred microseconds a frame; reported here it reads as a mode
 * change that often, and a client tracking the output's mode acts on every one of them. */
static uint32_t output_refresh_ns(void) {
    int mhz = g_output_refresh_mhz > 0 ? g_output_refresh_mhz : 60000;
    return (uint32_t)(1000000000000LL / mhz);
}

static void feedback_present_all(struct wl_list *list, int64_t t) {
    struct wl_resource *fb, *tmp;
    uint64_t sec = (uint64_t)(t / 1000000000LL);
    uint32_t nsec = (uint32_t)(t % 1000000000LL);
    uint32_t refresh = output_refresh_ns();
    wl_resource_for_each_safe(fb, tmp, list) {
        wp_presentation_feedback_send_presented(fb, (uint32_t)(sec >> 32), (uint32_t)sec, nsec,
                                                refresh, 0, 0,
                                                WP_PRESENTATION_FEEDBACK_KIND_VSYNC);
        wl_resource_destroy(fb);
    }
}

static void presentation_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void presentation_feedback(struct wl_client *c, struct wl_resource *r,
                                  struct wl_resource *surface, uint32_t id) {
    struct surface *s = wl_resource_get_user_data(surface);
    struct wl_resource *fb = wl_resource_create(c, &wp_presentation_feedback_interface,
                                                wl_resource_get_version(r), id);
    if (!fb) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(fb, NULL, NULL, feedback_resource_destroy);
    if (!s) { wp_presentation_feedback_send_discarded(fb); wl_resource_destroy(fb); return; }
    wl_list_insert(s->pending_feedback.prev, wl_resource_get_link(fb));
}
static const struct wp_presentation_interface presentation_impl = {
    .destroy = presentation_destroy,
    .feedback = presentation_feedback,
};
static void bind_presentation(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &wp_presentation_interface, ver, id);
    if (!r) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(r, &presentation_impl, NULL, NULL);
    wp_presentation_send_clock_id(r, CLOCK_MONOTONIC);
}
static void surface_frame(struct wl_client *c, struct wl_resource *r, uint32_t cb) {
    struct surface *s = wl_resource_get_user_data(r);
    struct wl_resource *callback = wl_resource_create(c, &wl_callback_interface, 1, cb);
    if (!callback) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(callback, NULL, NULL, frame_callback_destroy);
    wl_list_insert(s->pending_frames.prev, wl_resource_get_link(callback));
}
/* A region is kept as its bounding box; `exact` while that box is the region itself. */
struct region { int set, exact; int x, y, w, h; };

static void surface_set_opaque(struct wl_client *c, struct wl_resource *r,
                               struct wl_resource *region) {
    struct surface *s = wl_resource_get_user_data(r);
    struct region *rg = region ? wl_resource_get_user_data(region) : NULL;
    memset(s->pending_opaque, 0, sizeof(s->pending_opaque));
    s->pending_opaque_set = 1;
    if (!rg || !rg->set) return;
    if (rg->exact) {
        s->pending_opaque[0] = rg->x; s->pending_opaque[1] = rg->y;
        s->pending_opaque[2] = rg->w; s->pending_opaque[3] = rg->h;
    }
    struct client_info *ci = client_info_of(c);
    if (ci) ci->declares_opaque = 1;
}
static void surface_set_input(struct wl_client *c, struct wl_resource *r,
                              struct wl_resource *region) {}

static void surface_commit(struct wl_client *c, struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    struct surface *child;

    if (s->pending_src_set) {
        s->src_set = s->pending_src[2] > 0;
        memcpy(s->src, s->pending_src, sizeof(s->src));
        s->pending_src_set = 0;
    }
    if (s->pending_dst_set) {
        s->dst_set = s->pending_dst[0] > 0;
        memcpy(s->dst, s->pending_dst, sizeof(s->dst));
        s->pending_dst_set = 0;
    }
    if (s->pending_opaque_set) {
        memcpy(s->opaque, s->pending_opaque, sizeof(s->opaque));
        s->pending_opaque_set = 0;
    }

    if (s->pending_attach) {
        struct wl_resource *buffer = s->pending_buffer;
        struct dmabuf_buffer *db = get_dmabuf(buffer);
        struct wl_shm_buffer *shm = buffer && !db ? wl_shm_buffer_get(buffer) : NULL;

        /* The previous content is replaced before reaching the screen. */
        feedback_discard_all(&s->feedback);

        if (buffer) {
            wl_list_remove(&s->pending_buffer_destroy.link);
            wl_list_init(&s->pending_buffer_destroy.link);
        }
        s->pending_buffer = NULL;
        s->pending_attach = 0;

        if (s->role == ROLE_NONE) {
            /* Cursor or role-less surface: never drawn (the app draws its own pointer). */
            if (buffer) wl_buffer_send_release(buffer);
        } else if (db) {
            take_dmabuf(s, db, buffer);
        } else if (shm) {
            drop_dmabuf(s, 1);
            take_shm(s, shm, buffer);
        } else {
            drop_dmabuf(s, 1);
            s->has_content = 0;
            if (buffer) wl_buffer_send_release(buffer);
        }
    }

    /* Subsurface positions are applied when the parent commits. */
    wl_list_for_each(child, &s->children, child_link) {
        if (child->sub_pending) {
            child->sub_x = child->sub_pending_x;
            child->sub_y = child->sub_pending_y;
            child->sub_pending = 0;
        }
    }

    wl_list_insert_list(s->frames.prev, &s->pending_frames);
    wl_list_init(&s->pending_frames);
    wl_list_insert_list(s->feedback.prev, &s->pending_feedback);
    wl_list_init(&s->pending_feedback);
    if (s == g_hud_surface && s->hud_frame) {
        s->hud_frame = 0;
        wnc_on_game_frame();
    }
    constraints_surface_commit(s);
    wnc_color_commit(s->resource); /* wp_color_management_surface_v1 state (returns at once when HDR is off) */

    if (s->role == ROLE_TOPLEVEL && s->xdg_toplevel) {
        if (s->has_content) map_toplevel(s);
        else unmap_toplevel(s);
    }
    schedule_render();
}
static void surface_set_buffer_transform(struct wl_client *c, struct wl_resource *r, int32_t t) {}
static void surface_set_buffer_scale(struct wl_client *c, struct wl_resource *r, int32_t s) {}
static void surface_damage_buffer(struct wl_client *c, struct wl_resource *r,
                                  int32_t x, int32_t y, int32_t w, int32_t h) {}
static void surface_offset(struct wl_client *c, struct wl_resource *r, int32_t x, int32_t y) {}

static const struct wl_surface_interface surface_impl = {
    .destroy = surface_destroy,
    .attach = surface_attach,
    .damage = surface_damage,
    .frame = surface_frame,
    .set_opaque_region = surface_set_opaque,
    .set_input_region = surface_set_input,
    .commit = surface_commit,
    .set_buffer_transform = surface_set_buffer_transform,
    .set_buffer_scale = surface_set_buffer_scale,
    .damage_buffer = surface_damage_buffer,
    .offset = surface_offset,
};

static void detach_from_parent(struct surface *s) {
    if (!s->parent) return;
    wl_list_remove(&s->child_link);
    wl_list_init(&s->child_link);
    s->parent = NULL;
}

static void surface_resource_destroy(struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    struct surface *child, *tmp;
    struct wl_resource *cb, *cbtmp;

    if (!s) return;
    for (int i = 0; i < g_nptrs; i++) if (g_ptrs[i].focus == r) g_ptrs[i].focus = NULL;
    for (int i = 0; i < g_nkbs; i++) if (g_kbs[i].focus == r) g_kbs[i].focus = NULL;
    if (g_grab == s) g_grab = NULL;
    if (g_key_target == s) g_key_target = NULL;
    if (g_ime_click == s) g_ime_click = NULL;
    wnc_text_input_surface_gone(r);
    if (g_desktop == s) { g_desktop = NULL; wnc_log("desktop", "the desktop closed"); }
    if (g_hud_surface == s) { g_hud_surface = NULL; wnc_on_game_surface(NULL, NULL); }
    constraints_surface_gone(s);

    unmap_toplevel(s);
    detach_from_parent(s);
    wl_list_for_each_safe(child, tmp, &s->children, child_link) {
        wl_list_remove(&child->child_link);
        wl_list_init(&child->child_link);
        child->parent = NULL;
    }
    if (s->pending_buffer) wl_list_remove(&s->pending_buffer_destroy.link);
    pending_releases_forget_surface(s);
    drop_dmabuf(s, 0);
    ahb_swapchain_surface_gone(s);
    vkp_image_destroy(s->shm_img);
    wl_resource_for_each_safe(cb, cbtmp, &s->pending_frames) wl_resource_destroy(cb);
    wl_resource_for_each_safe(cb, cbtmp, &s->frames) wl_resource_destroy(cb);
    feedback_discard_all(&s->pending_feedback);
    feedback_discard_all(&s->feedback);
    if (s->viewport) wl_resource_set_user_data(s->viewport, NULL);
    if (s->subsurface) wl_resource_set_user_data(s->subsurface, NULL);
    if (s->xdg_surface) wl_resource_set_user_data(s->xdg_surface, NULL);
    if (s->xdg_toplevel) wl_resource_set_user_data(s->xdg_toplevel, NULL);
    wl_list_remove(&s->link);
    free(s->title);
    free(s);
    schedule_render();
}

/* ------------------------------------------------------------------ wl_region */

/* A region is kept as the bounding box of its rectangles: enough for pointer confinement,
 * where winewayland sends one rectangle (the ClipCursor area). Subtractions are ignored. */
static void region_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void region_add(struct wl_client *c, struct wl_resource *r,
                       int32_t x, int32_t y, int32_t w, int32_t h) {
    struct region *rg = wl_resource_get_user_data(r);
    if (!rg || w <= 0 || h <= 0) return;
    if (!rg->set) { rg->x = x; rg->y = y; rg->w = w; rg->h = h; rg->set = rg->exact = 1; return; }
    long long x2 = (long long)rg->x + rg->w, y2 = (long long)rg->y + rg->h;
    long long nx2 = (long long)x + w, ny2 = (long long)y + h;
    int inside = x >= rg->x && y >= rg->y && nx2 <= x2 && ny2 <= y2;
    int around = x <= rg->x && y <= rg->y && nx2 >= x2 && ny2 >= y2;
    if (!inside && !around) rg->exact = 0;
    if (nx2 > x2) x2 = nx2;
    if (ny2 > y2) y2 = ny2;
    if (x < rg->x) rg->x = x;
    if (y < rg->y) rg->y = y;
    rg->w = (int)(x2 - rg->x > INT32_MAX ? INT32_MAX : x2 - rg->x);
    rg->h = (int)(y2 - rg->y > INT32_MAX ? INT32_MAX : y2 - rg->y);
}
static void region_subtract(struct wl_client *c, struct wl_resource *r,
                            int32_t x, int32_t y, int32_t w, int32_t h) {
    struct region *rg = wl_resource_get_user_data(r);
    if (rg && w > 0 && h > 0) rg->exact = 0;
}
static void region_resource_destroy(struct wl_resource *r) { free(wl_resource_get_user_data(r)); }
static const struct wl_region_interface region_impl = {
    .destroy = region_destroy,
    .add = region_add,
    .subtract = region_subtract,
};

/* ------------------------------------------------------------------ wl_compositor */

static void compositor_create_surface(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct surface *s = calloc(1, sizeof(*s));
    if (!s) { wl_client_post_no_memory(c); return; }
    s->resource = wl_resource_create(c, &wl_surface_interface, wl_resource_get_version(r), id);
    if (!s->resource) { free(s); wl_client_post_no_memory(c); return; }
    wl_list_init(&s->pending_frames);
    wl_list_init(&s->frames);
    wl_list_init(&s->pending_feedback);
    wl_list_init(&s->feedback);
    wl_list_init(&s->children);
    wl_list_init(&s->child_link);
    wl_list_init(&s->toplevel_link);
    wl_list_init(&s->pending_buffer_destroy.link);
    wl_list_insert(&g_surfaces, &s->link);
    wl_resource_set_implementation(s->resource, &surface_impl, s, surface_resource_destroy);
}
static void compositor_create_region(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct wl_resource *reg = wl_resource_create(c, &wl_region_interface, 1, id);
    if (!reg) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(reg, &region_impl, calloc(1, sizeof(struct region)),
                                   region_resource_destroy);
}
static const struct wl_compositor_interface compositor_impl = {
    .create_surface = compositor_create_surface,
    .create_region = compositor_create_region,
};
static void bind_compositor(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &wl_compositor_interface, ver, id);
    wl_resource_set_implementation(r, &compositor_impl, NULL, NULL);
}

/* --------------------------------------------------------------- wl_subcompositor
 * winewayland puts a window's Vulkan/GL swapchain in a subsurface over the window. */

static void subsurface_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void subsurface_set_position(struct wl_client *c, struct wl_resource *r, int32_t x, int32_t y) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s) return;
    s->sub_pending_x = x;
    s->sub_pending_y = y;
    s->sub_pending = 1;
}
static void restack_child(struct surface *s, struct surface *sibling, int above) {
    if (!s || !s->parent || !sibling) return;
    if (sibling == s->parent) {
        /* Relative to the parent: directly above it or directly below it. */
        s->below_parent = !above;
        wl_list_remove(&s->child_link);
        if (above) wl_list_insert(&s->parent->children, &s->child_link);
        else wl_list_insert(s->parent->children.prev, &s->child_link);
    } else if (sibling->parent == s->parent) {
        s->below_parent = sibling->below_parent;
        wl_list_remove(&s->child_link);
        if (above) wl_list_insert(&sibling->child_link, &s->child_link);
        else wl_list_insert(sibling->child_link.prev, &s->child_link);
    }
    schedule_render();
}
static void subsurface_place_above(struct wl_client *c, struct wl_resource *r,
                                   struct wl_resource *sibling) {
    restack_child(wl_resource_get_user_data(r), wl_resource_get_user_data(sibling), 1);
}
static void subsurface_place_below(struct wl_client *c, struct wl_resource *r,
                                   struct wl_resource *sibling) {
    restack_child(wl_resource_get_user_data(r), wl_resource_get_user_data(sibling), 0);
}
static void subsurface_set_sync(struct wl_client *c, struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    if (s) s->sub_sync = 1;
}
static void subsurface_set_desync(struct wl_client *c, struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    if (s) s->sub_sync = 0;
}
static const struct wl_subsurface_interface subsurface_impl = {
    .destroy = subsurface_destroy,
    .set_position = subsurface_set_position,
    .place_above = subsurface_place_above,
    .place_below = subsurface_place_below,
    .set_sync = subsurface_set_sync,
    .set_desync = subsurface_set_desync,
};
static void subsurface_resource_destroy(struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s) return;
    detach_from_parent(s);
    s->subsurface = NULL;
    s->role = ROLE_NONE;
    s->has_content = 0;
    drop_dmabuf(s, 1);
    schedule_render();
}

static void subcompositor_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void subcompositor_get_subsurface(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                         struct wl_resource *surface, struct wl_resource *parent) {
    struct surface *s = wl_resource_get_user_data(surface);
    struct surface *p = wl_resource_get_user_data(parent);
    struct wl_resource *sub = wl_resource_create(c, &wl_subsurface_interface, wl_resource_get_version(r), id);
    if (!sub) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(sub, &subsurface_impl, s, subsurface_resource_destroy);
    if (!s || !p || s == p) return;
    detach_from_parent(s);
    s->role = ROLE_SUBSURFACE;
    s->subsurface = sub;
    s->parent = p;
    s->below_parent = 0;
    s->sub_sync = 1;
    s->sub_x = s->sub_y = 0;
    wl_list_insert(p->children.prev, &s->child_link); /* new subsurfaces go on top */
}
static const struct wl_subcompositor_interface subcompositor_impl = {
    .destroy = subcompositor_destroy,
    .get_subsurface = subcompositor_get_subsurface,
};
static void bind_subcompositor(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &wl_subcompositor_interface, ver, id);
    wl_resource_set_implementation(r, &subcompositor_impl, NULL, NULL);
}

/* ---------------------------------------------------------------- wp_viewporter
 * winewayland crops window buffers to the window size and scales swapchains to the
 * client area with viewports, so they must be honoured for things to line up. */

static void viewport_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void viewport_set_source(struct wl_client *c, struct wl_resource *r,
                                wl_fixed_t x, wl_fixed_t y, wl_fixed_t w, wl_fixed_t h) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s) return;
    s->pending_src[0] = (float)wl_fixed_to_double(x);
    s->pending_src[1] = (float)wl_fixed_to_double(y);
    s->pending_src[2] = w == wl_fixed_from_int(-1) ? -1.0f : (float)wl_fixed_to_double(w);
    s->pending_src[3] = h == wl_fixed_from_int(-1) ? -1.0f : (float)wl_fixed_to_double(h);
    s->pending_src_set = 1;
}
static void viewport_set_destination(struct wl_client *c, struct wl_resource *r, int32_t w, int32_t h) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s) return;
    s->pending_dst[0] = w;
    s->pending_dst[1] = h;
    s->pending_dst_set = 1;
}
static const struct wp_viewport_interface viewport_impl = {
    .destroy = viewport_destroy,
    .set_source = viewport_set_source,
    .set_destination = viewport_set_destination,
};
static void viewport_resource_destroy(struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s) return;
    s->viewport = NULL;
    s->pending_src[2] = -1.0f; s->pending_src_set = 1;
    s->pending_dst[0] = -1; s->pending_dst_set = 1;
}

static void viewporter_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void viewporter_get_viewport(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                    struct wl_resource *surface) {
    struct surface *s = wl_resource_get_user_data(surface);
    struct wl_resource *vp = wl_resource_create(c, &wp_viewport_interface, wl_resource_get_version(r), id);
    if (!vp) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(vp, &viewport_impl, s, viewport_resource_destroy);
    if (s) s->viewport = vp;
}
static const struct wp_viewporter_interface viewporter_impl = {
    .destroy = viewporter_destroy,
    .get_viewport = viewporter_get_viewport,
};
static void bind_viewporter(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &wp_viewporter_interface, ver, id);
    wl_resource_set_implementation(r, &viewporter_impl, NULL, NULL);
}

/* ------------------------------------------------------------------ xdg_shell */

static void xdg_toplevel_destroy_req(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void xdg_toplevel_noop_parent(struct wl_client *c, struct wl_resource *r, struct wl_resource *p) {}
static void xdg_toplevel_set_title(struct wl_client *c, struct wl_resource *r, const char *title) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s) return;
    free(s->title);
    s->title = title ? strdup(title) : NULL;
}
static void xdg_toplevel_set_app_id(struct wl_client *c, struct wl_resource *r, const char *id) {}
static void xdg_toplevel_show_menu(struct wl_client *c, struct wl_resource *r, struct wl_resource *seat,
                                   uint32_t serial, int32_t x, int32_t y) {}
static void xdg_toplevel_move(struct wl_client *c, struct wl_resource *r, struct wl_resource *seat,
                              uint32_t serial) {}
static void xdg_toplevel_resize(struct wl_client *c, struct wl_resource *r, struct wl_resource *seat,
                                uint32_t serial, uint32_t edges) {}
static void xdg_toplevel_set_i32(struct wl_client *c, struct wl_resource *r, int32_t w, int32_t h) {}
static void xdg_toplevel_noop(struct wl_client *c, struct wl_resource *r) {}
/* Size 0x0 = the client picks; active, and fullscreen when asked (libdecor draws no frame then). */
static void send_toplevel_configure(struct surface *s) {
    struct wl_array states;
    wl_array_init(&states);
    uint32_t *st = wl_array_add(&states, sizeof(uint32_t));
    *st = XDG_TOPLEVEL_STATE_ACTIVATED;
    if (s->fullscreen) {
        st = wl_array_add(&states, sizeof(uint32_t));
        *st = XDG_TOPLEVEL_STATE_FULLSCREEN;
    }
    xdg_toplevel_send_configure(s->xdg_toplevel, 0, 0, &states);
    wl_array_release(&states);
    xdg_surface_send_configure(s->xdg_surface, wl_display_next_serial(g_display));
}
static void xdg_toplevel_set_fullscreen(struct wl_client *c, struct wl_resource *r, struct wl_resource *output) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s || !s->xdg_surface || s->fullscreen) return;
    s->fullscreen = 1;
    send_toplevel_configure(s);
}
static void xdg_toplevel_unset_fullscreen(struct wl_client *c, struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s || !s->xdg_surface || !s->fullscreen) return;
    s->fullscreen = 0;
    send_toplevel_configure(s);
}
static const struct xdg_toplevel_interface xdg_toplevel_impl = {
    .destroy = xdg_toplevel_destroy_req,
    .set_parent = xdg_toplevel_noop_parent,
    .set_title = xdg_toplevel_set_title,
    .set_app_id = xdg_toplevel_set_app_id,
    .show_window_menu = xdg_toplevel_show_menu,
    .move = xdg_toplevel_move,
    .resize = xdg_toplevel_resize,
    .set_max_size = xdg_toplevel_set_i32,
    .set_min_size = xdg_toplevel_set_i32,
    .set_maximized = xdg_toplevel_noop,
    .unset_maximized = xdg_toplevel_noop,
    .set_fullscreen = xdg_toplevel_set_fullscreen,
    .unset_fullscreen = xdg_toplevel_unset_fullscreen,
    .set_minimized = xdg_toplevel_noop,
};
static void xdg_toplevel_resource_destroy(struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    if (!s) return;
    s->xdg_toplevel = NULL;
    unmap_toplevel(s);
    s->placed = 0;
    s->hwnd = 0;
    schedule_render();
}

static void xdg_surface_destroy_req(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void xdg_surface_get_toplevel(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct surface *s = wl_resource_get_user_data(r);
    struct wl_resource *tl = wl_resource_create(c, &xdg_toplevel_interface, wl_resource_get_version(r), id);
    if (!tl) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(tl, &xdg_toplevel_impl, s, xdg_toplevel_resource_destroy);
    if (!s) return;
    s->role = ROLE_TOPLEVEL;
    s->xdg_toplevel = tl;
    s->fullscreen = 0;
    send_toplevel_configure(s);
}
static void xdg_surface_get_popup(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                  struct wl_resource *parent, struct wl_resource *positioner) {}
static void xdg_surface_set_geometry(struct wl_client *c, struct wl_resource *r,
                                     int32_t x, int32_t y, int32_t w, int32_t h) {}
static void xdg_surface_ack_configure(struct wl_client *c, struct wl_resource *r, uint32_t serial) {}
static const struct xdg_surface_interface xdg_surface_impl = {
    .destroy = xdg_surface_destroy_req,
    .get_toplevel = xdg_surface_get_toplevel,
    .get_popup = xdg_surface_get_popup,
    .set_window_geometry = xdg_surface_set_geometry,
    .ack_configure = xdg_surface_ack_configure,
};
static void xdg_surface_resource_destroy(struct wl_resource *r) {
    struct surface *s = wl_resource_get_user_data(r);
    if (s) s->xdg_surface = NULL;
}

static void xdg_wm_base_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void xdg_wm_base_create_positioner(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct wl_resource *p = wl_resource_create(c, &xdg_positioner_interface, wl_resource_get_version(r), id);
    if (p) wl_resource_set_implementation(p, NULL, NULL, NULL);
}
static void xdg_wm_base_get_xdg_surface(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                        struct wl_resource *surf) {
    struct surface *s = wl_resource_get_user_data(surf);
    struct wl_resource *xs = wl_resource_create(c, &xdg_surface_interface, wl_resource_get_version(r), id);
    if (!xs) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(xs, &xdg_surface_impl, s, xdg_surface_resource_destroy);
    if (s) s->xdg_surface = xs;
}
static void xdg_wm_base_pong(struct wl_client *c, struct wl_resource *r, uint32_t serial) {}
static const struct xdg_wm_base_interface xdg_wm_base_impl = {
    .destroy = xdg_wm_base_destroy,
    .create_positioner = xdg_wm_base_create_positioner,
    .get_xdg_surface = xdg_wm_base_get_xdg_surface,
    .pong = xdg_wm_base_pong,
};
static void bind_xdg_wm_base(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &xdg_wm_base_interface, ver, id);
    wl_resource_set_implementation(r, &xdg_wm_base_impl, NULL, NULL);
}

/* ------------------------------------------------------------------ banner_desktop_v1 */

static void desktop_destroy_req(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void desktop_set_desktop(struct wl_client *c, struct wl_resource *r, struct wl_resource *surface) {
    struct surface *s = wl_resource_get_user_data(surface);
    if (!s || (s->role != ROLE_NONE && s->role != ROLE_DESKTOP)) return;
    s->role = ROLE_DESKTOP;
    g_desktop = s;
    wnc_log("desktop", "Windows virtual desktop created by %s", client_name(c));
    schedule_render();
}
static void desktop_set_window(struct wl_client *c, struct wl_resource *r, struct wl_resource *surface,
                               uint32_t hwnd, int32_t x, int32_t y) {
    struct surface *s = wl_resource_get_user_data(surface);
    if (!s) return;
    if (s->placed && (s->x != x || s->y != y) && s->mapped && log_budget()) {
        char name[160];
        describe(s, name, sizeof(name));
        wnc_log("window", "moved %s to %d,%d", name, x, y);
    }
    s->hwnd = hwnd;
    s->x = x;
    s->y = y;
    s->placed = 1;
    if (s->mapped) apply_zorder();
    schedule_render();
}
static void desktop_set_zorder(struct wl_client *c, struct wl_resource *r, struct wl_array *hwnds) {
    size_t count = hwnds->size / sizeof(uint32_t);
    uint32_t *copy = count ? malloc(count * sizeof(uint32_t)) : NULL;
    if (count && !copy) return;
    if (count) memcpy(copy, hwnds->data, count * sizeof(uint32_t));
    free(g_zorder);
    g_zorder = copy;
    g_zorder_count = count;
    apply_zorder();
    schedule_render();
}
static const struct banner_desktop_v1_interface desktop_impl = {
    .destroy = desktop_destroy_req,
    .set_desktop = desktop_set_desktop,
    .set_window = desktop_set_window,
    .set_zorder = desktop_set_zorder,
};
static void bind_desktop(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &banner_desktop_v1_interface, ver, id);
    wl_resource_set_implementation(r, &desktop_impl, NULL, NULL);
}

/* ------------------------------------------------------------ zwp_linux_dmabuf_v1 */

static void dbuf_buffer_resource_destroy(struct wl_resource *r) {
    dmabuf_buffer_unref(wl_resource_get_user_data(r));
}

static void params_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void params_add(struct wl_client *c, struct wl_resource *r, int32_t fd, uint32_t plane,
                       uint32_t offset, uint32_t stride, uint32_t mod_hi, uint32_t mod_lo) {
    struct dmabuf_params *p = wl_resource_get_user_data(r);
    if (plane >= MAX_PLANES) { close(fd); return; }
    if (p->fd[plane] >= 0) close(p->fd[plane]);
    p->fd[plane] = fd;
    p->offset[plane] = offset;
    p->stride[plane] = stride;
    p->modifier[plane] = ((uint64_t)mod_hi << 32) | mod_lo;
    if ((int)plane + 1 > p->n_planes) p->n_planes = plane + 1;
}
static struct wl_resource *params_do_create(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                            int32_t w, int32_t h, uint32_t format, uint32_t flags) {
    struct dmabuf_params *p = wl_resource_get_user_data(r);
    struct dmabuf_buffer *b = calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->n_planes = p->n_planes;
    b->refs = 1; /* the wl_buffer resource's */
    b->width = w; b->height = h; b->format = format;
    b->modifier = p->modifier[0];
    for (int i = 0; i < MAX_PLANES; i++) b->fd[i] = -1;
    for (int i = 0; i < p->n_planes; i++) {
        b->fd[i] = p->fd[i];
        b->offset[i] = p->offset[i];
        b->stride[i] = p->stride[i];
        p->fd[i] = -1; /* ownership moves to the buffer */
    }
    struct wl_resource *buf = wl_resource_create(c, &wl_buffer_interface, 1, id);
    if (!buf) {
        for (int i = 0; i < b->n_planes; i++) if (b->fd[i] >= 0) close(b->fd[i]);
        free(b);
        wl_client_post_no_memory(c);
        return NULL;
    }
    wl_resource_set_implementation(buf, &dbuf_buffer_impl, b, dbuf_buffer_resource_destroy);
    struct client_info *ci = client_info_of(c);
    if (ci) ci->dmabuf_buffers++;
    return buf;
}
static void params_create(struct wl_client *c, struct wl_resource *r, int32_t w, int32_t h,
                          uint32_t format, uint32_t flags) {
    struct wl_resource *buf = params_do_create(c, r, 0, w, h, format, flags);
    if (buf) zwp_linux_buffer_params_v1_send_created(r, buf);
    else zwp_linux_buffer_params_v1_send_failed(r);
}
static void params_create_immed(struct wl_client *c, struct wl_resource *r, uint32_t buffer_id,
                                int32_t w, int32_t h, uint32_t format, uint32_t flags) {
    params_do_create(c, r, buffer_id, w, h, format, flags);
}
static const struct zwp_linux_buffer_params_v1_interface params_impl = {
    .destroy = params_destroy,
    .add = params_add,
    .create = params_create,
    .create_immed = params_create_immed,
};
static void params_resource_destroy(struct wl_resource *r) {
    struct dmabuf_params *p = wl_resource_get_user_data(r);
    if (!p) return;
    for (int i = 0; i < MAX_PLANES; i++)
        if (p->fd[i] >= 0) close(p->fd[i]);
    free(p);
}

static void dmabuf_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void dmabuf_create_params(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct dmabuf_params *p = calloc(1, sizeof(*p));
    if (!p) { wl_client_post_no_memory(c); return; }
    for (int i = 0; i < MAX_PLANES; i++) p->fd[i] = -1;
    struct wl_resource *pr = wl_resource_create(c, &zwp_linux_buffer_params_v1_interface,
                                                wl_resource_get_version(r), id);
    if (!pr) { free(p); wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(pr, &params_impl, p, params_resource_destroy);
}
static void dmabuf_get_default_feedback(struct wl_client *c, struct wl_resource *r, uint32_t id);
static void dmabuf_get_surface_feedback(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                        struct wl_resource *surface);
static const struct zwp_linux_dmabuf_v1_interface dmabuf_impl = {
    .destroy = dmabuf_destroy,
    .create_params = dmabuf_create_params,
    .get_default_feedback = dmabuf_get_default_feedback,
    .get_surface_feedback = dmabuf_get_surface_feedback,
};
/* The advertised format/modifier table, built at the first bind from what the renderer's driver
 * can import (vkp_dmabuf_modifiers). INVALID is always offered too (Mesa drops it; other clients
 * may use it for the implicit path). Without a renderer the list is LINEAR + INVALID, as before. */
/* The last two rows are HDR10's (color_mgmt.h): A2B10G10R10 as AB30 (alpha) + XB30 (opaque) - Mesa
 * lists a VkFormat only when both are advertised, and it is the one 10-bit layout our zero-copy WSI
 * can put in a gralloc buffer (AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM). They are advertised ONLY
 * while the HDR gate is open, and after the 8-bit rows, so every other session - and every client that
 * takes the first format it sees - gets exactly the table it always had. */
#define DMABUF_NFMT_MAX 6
#define DMABUF_NFMT_SDR 4
#define DMABUF_NMOD 4
#define DRM_ABGR2101010 FOURCC('A', 'B', '3', '0')
#define DRM_XBGR2101010 FOURCC('X', 'B', '3', '0')
static struct { uint32_t fmt; uint64_t mods[DMABUF_NMOD]; int n; } g_dmabuf_fmts[DMABUF_NFMT_MAX] = {
    {DRM_ARGB8888}, {DRM_XRGB8888}, {DRM_ABGR8888}, {DRM_XBGR8888}, {DRM_ABGR2101010}, {DRM_XBGR2101010}};
static int g_dmabuf_nfmt = DMABUF_NFMT_SDR;   /* rows advertised: the SDR four, + 2 with the HDR gate open */
static int g_dmabuf_fmts_ready;
static void dmabuf_build_feedback(void);

static void dmabuf_build_formats(void) {
    char line[320];
    int pos = 0, compressed = 0;
    g_dmabuf_fmts_ready = 1;
    g_dmabuf_nfmt = wnc_color_hdr_open() ? DMABUF_NFMT_MAX : DMABUF_NFMT_SDR;
    for (int f = 0; f < g_dmabuf_nfmt; f++) {
        uint64_t got[DMABUF_NMOD];
        int n = vkp_dmabuf_modifiers(g_dmabuf_fmts[f].fmt, got, DMABUF_NMOD), k = 0;
        /* LINEAR first: it is the layout every client and the shm fallback agree on, and the one
         * we advertised before this table existed. */
        g_dmabuf_fmts[f].mods[k++] = MOD_LINEAR;
        for (int i = 0; i < n && k < DMABUF_NMOD - 1; i++) {
            if (got[i] == MOD_LINEAR) continue;
            if (got[i] == VKP_MOD_QCOM_COMPRESSED && !g_ubwc) continue;
            g_dmabuf_fmts[f].mods[k++] = got[i];
            if (got[i] == VKP_MOD_QCOM_COMPRESSED) compressed++;
        }
        g_dmabuf_fmts[f].mods[k++] = MOD_INVALID;
        g_dmabuf_fmts[f].n = k;
        uint32_t fmt = g_dmabuf_fmts[f].fmt;
        pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "%s%c%c%c%c", f ? ", " : "",
                        fmt & 0xff, (fmt >> 8) & 0xff, (fmt >> 16) & 0xff, (fmt >> 24) & 0xff);
        for (int i = 0; i < k - 1 && pos < (int)sizeof(line); i++)
            pos += snprintf(line + pos, sizeof(line) - (size_t)pos, "%s%s", i ? "+" : " ",
                            vkp_modifier_name(g_dmabuf_fmts[f].mods[i]));
        if (pos >= (int)sizeof(line)) pos = (int)sizeof(line) - 1;
    }
    wnc_log("dmabuf", "formats: %s%s", line,
               !g_ubwc ? " (BANNER_WAYLAND_UBWC=0: qcom_compressed not advertised)" : "");
    if (g_ubwc && !compressed)
        wnc_log("dmabuf", "the compositor's driver (%s) reports no importable qcom_compressed layout: "
                   "game swapchains stay linear", vkp_gpu_name());
    if (g_dmabuf_nfmt > DMABUF_NFMT_SDR) {
        /* The rows above always carry LINEAR; say what the compositor's own driver can really import
         * for 10-bit, since a gralloc buffer it cannot import is shown on the display layer only. */
        uint64_t got[DMABUF_NMOD];
        int n = vkp_dmabuf_modifiers(DRM_XBGR2101010, got, DMABUF_NMOD), ubwc10 = 0;
        for (int i = 0; i < n; i++) if (got[i] == VKP_MOD_QCOM_COMPRESSED) ubwc10 = 1;
        wnc_log("color", "10-bit dma-buf formats AB30/XB30 advertised for HDR10; the compositor's driver (%s) "
                   "imports XB30 %s", vkp_gpu_name(),
                   n == 0 ? "with no layout it reports (copy path unlikely; display layer only)"
                          : ubwc10 ? "linear and UBWC" : "linear only (UBWC 10-bit frames: display layer only)");
    }
    dmabuf_build_feedback();
}

/* ---- dmabuf feedback (zwp_linux_dmabuf_v1 version 4)
 *
 * Turnip's Vulkan WSI is happy with the version 3 format/modifier events, so every Vulkan game
 * worked while we only advertised 3. Mesa's EGL is not: its Wayland platform only takes the GPU
 * (kopper/Zink) path when it can bind this interface with FEEDBACK, and with 3 it silently drops
 * to its wl_shm software path. On this Proton layer that path has no software rasteriser to fall
 * back to (the gallium build is zink+kopper+swrast, no llvmpipe), so the shm buffer it commits is
 * never written: a native OpenGL window came out solid black, ~30 shm commits/s and no GPU frame
 * at all (Wizardry: The Labyrinth of Lost Souls). Feedback is what makes that path work.
 *
 * What a client needs from us is one tranche describing "everything the compositor can import":
 * a format table it mmaps read-only, the device to allocate on, and the indices it may use.
 * Clients binding versions 1-3 keep getting the old format/modifier events instead. */

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
#ifndef MFD_ALLOW_SEALING
#define MFD_ALLOW_SEALING 0x0002U
#endif
#ifndef F_ADD_SEALS
#define F_ADD_SEALS 1033
#define F_SEAL_SEAL 0x0001
#define F_SEAL_SHRINK 0x0002
#define F_SEAL_GROW 0x0004
#define F_SEAL_WRITE 0x0008
#endif

struct dmabuf_fmt_entry { uint32_t format; uint32_t pad; uint64_t modifier; }; /* the wire layout */

static int g_fmt_table_fd = -1;         /* sealed read-only memfd of dmabuf_fmt_entry[] */
static size_t g_fmt_table_size;
static uint16_t g_fmt_table_n;          /* entries, == the indices a tranche may name */
static dev_t g_main_device;             /* the render node clients should allocate on */

/* The GPU we import through is reached with KGSL, not DRM, so there is no render node of our own
 * to name. Clients only use main_device to match "the same device as the compositor" and to pick
 * a driver; the one DRM render node this platform has is the right answer, and Zink ignores it
 * anyway (it renders on the Vulkan device it already has). 0 if the platform has none. */
static dev_t dmabuf_render_node(void) {
    static const char *nodes[] = {"/dev/dri/renderD128", "/dev/dri/renderD129", "/dev/dri/card0"};
    struct stat st;
    if (g_no_render_node) return 0;
    for (size_t i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
        if (!stat(nodes[i], &st) && S_ISCHR(st.st_mode)) return st.st_rdev;
    return 0;
}

static void dmabuf_build_feedback(void) {
    struct dmabuf_fmt_entry entries[DMABUF_NFMT_MAX * DMABUF_NMOD];
    int n = 0;

    g_main_device = dmabuf_render_node();
    for (int f = 0; f < g_dmabuf_nfmt; f++)
        for (int m = 0; m < g_dmabuf_fmts[f].n; m++) {
            if (g_dmabuf_fmts[f].mods[m] == MOD_INVALID) continue; /* never offer INVALID here */
            entries[n].format = g_dmabuf_fmts[f].fmt;
            entries[n].pad = 0;
            entries[n].modifier = g_dmabuf_fmts[f].mods[m];
            n++;
        }
    if (!n) return;

    int fd = (int)syscall(__NR_memfd_create, "wn-dmabuf-formats",
                          MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { WLOGE("dmabuf feedback: memfd_create failed (%s)", strerror(errno)); return; }
    size_t size = (size_t)n * sizeof(entries[0]);
    if (write(fd, entries, size) != (ssize_t)size) {
        WLOGE("dmabuf feedback: could not write the format table (%s)", strerror(errno));
        close(fd);
        return;
    }
    /* The client mmaps this read-only and trusts it not to change under it. */
    fcntl(fd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE);
    g_fmt_table_fd = fd;
    g_fmt_table_size = size;
    g_fmt_table_n = (uint16_t)n;
    wnc_log("dmabuf", "feedback ready: %d format/modifier pairs, main device %u:%u",
               n, (unsigned)major(g_main_device), (unsigned)minor(g_main_device));
    /* Vulkan games never need the node. Mesa's EGL did until Wayland layer versionCode 9: without
     * one it fell back to a software path that draws nothing here (black window, sound plays). */
    if (!g_main_device)
        wnc_log("dmabuf", "%s: OpenGL games need Wayland layer versionCode 9 or newer, "
                   "which runs OpenGL on the GPU without a DRM node; older layers show a black window",
                   g_no_render_node ? "no DRM device named (forced by BANNER_WAYLAND_NO_RENDER_NODE=1)"
                                    : "this device gives apps no display (DRM) device (/dev/dri)");
}

/* One tranche: our device, every pair in the table, no scanout flag. */
static void dmabuf_feedback_send(struct wl_resource *fb) {
    struct wl_array dev, idx;
    uint16_t *ind;

    if (g_fmt_table_fd < 0) { zwp_linux_dmabuf_feedback_v1_send_done(fb); return; }

    zwp_linux_dmabuf_feedback_v1_send_format_table(fb, g_fmt_table_fd, (uint32_t)g_fmt_table_size);

    wl_array_init(&dev);
    memcpy(wl_array_add(&dev, sizeof(dev_t)), &g_main_device, sizeof(dev_t));
    zwp_linux_dmabuf_feedback_v1_send_main_device(fb, &dev);
    zwp_linux_dmabuf_feedback_v1_send_tranche_target_device(fb, &dev);
    wl_array_release(&dev);

    wl_array_init(&idx);
    ind = wl_array_add(&idx, (size_t)g_fmt_table_n * sizeof(uint16_t));
    if (ind) for (uint16_t i = 0; i < g_fmt_table_n; i++) ind[i] = i;
    zwp_linux_dmabuf_feedback_v1_send_tranche_formats(fb, &idx);
    wl_array_release(&idx);

    zwp_linux_dmabuf_feedback_v1_send_tranche_flags(fb, 0);
    zwp_linux_dmabuf_feedback_v1_send_tranche_done(fb);
    zwp_linux_dmabuf_feedback_v1_send_done(fb);
}

static void dmabuf_feedback_destroy(struct wl_client *c, struct wl_resource *r) {
    wl_resource_destroy(r);
}
static const struct zwp_linux_dmabuf_feedback_v1_interface dmabuf_feedback_impl = {
    .destroy = dmabuf_feedback_destroy,
};

static void dmabuf_new_feedback(struct wl_client *c, struct wl_resource *parent, uint32_t id) {
    struct wl_resource *fb = wl_resource_create(c, &zwp_linux_dmabuf_feedback_v1_interface,
                                                wl_resource_get_version(parent), id);
    if (!fb) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(fb, &dmabuf_feedback_impl, NULL, NULL);
    if (!g_dmabuf_fmts_ready) dmabuf_build_formats();
    dmabuf_feedback_send(fb);
}

static void dmabuf_get_default_feedback(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct client_info *ci = client_info_of(c);
    if (ci) ci->asked_feedback = 1;
    dmabuf_new_feedback(c, r, id);
}
/* Per-surface feedback would let us hint a different tranche for a window on its own display
 * layer; we have nothing better to say per surface, so it is the default one. */
static void dmabuf_get_surface_feedback(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                        struct wl_resource *surface) {
    dmabuf_new_feedback(c, r, id);
}

static void bind_dmabuf(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &zwp_linux_dmabuf_v1_interface, ver, id);
    wl_resource_set_implementation(r, &dmabuf_impl, NULL, NULL);
    if (!g_dmabuf_fmts_ready) dmabuf_build_formats();
    /* From version 4 the format and modifier events are gone: the client asks for feedback
     * instead, and sending both would only confuse it about which list is authoritative. */
    if (ver >= 4) return;
    for (int f = 0; f < g_dmabuf_nfmt; f++) {
        zwp_linux_dmabuf_v1_send_format(r, g_dmabuf_fmts[f].fmt);
        if (ver >= 3)
            for (int m = 0; m < g_dmabuf_fmts[f].n; m++)
                zwp_linux_dmabuf_v1_send_modifier(r, g_dmabuf_fmts[f].fmt,
                                                  (uint32_t)(g_dmabuf_fmts[f].mods[m] >> 32),
                                                  (uint32_t)(g_dmabuf_fmts[f].mods[m] & 0xffffffff));
    }
}

/* ------------------------------------------------------------------ wl_output */

static void bind_output(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &wl_output_interface, ver, id);
    wl_resource_set_implementation(r, NULL, NULL, NULL);
    wl_output_send_geometry(r, 0, 0, 340, 190, WL_OUTPUT_SUBPIXEL_UNKNOWN,
                            "WinNative", "Wayland", WL_OUTPUT_TRANSFORM_NORMAL);
    wl_output_send_mode(r, WL_OUTPUT_MODE_CURRENT | WL_OUTPUT_MODE_PREFERRED,
                        g_output_w > 0 ? g_output_w : 1920, g_output_h > 0 ? g_output_h : 1080,
                        g_output_refresh_mhz > 0 ? g_output_refresh_mhz : 60000);
    if (ver >= WL_OUTPUT_NAME_SINCE_VERSION) {
        wl_output_send_name(r, "WinNative-1");
        wl_output_send_description(r, "WinNative display");
    }
    if (ver >= 2) {
        wl_output_send_scale(r, 1);
        wl_output_send_done(r);
    }
}

/* ------------------------------------------------------------------ rendering */

struct draw_list { struct vkp_draw *d; int n, cap; };

static struct vkp_image *surface_image(struct surface *s) {
    if (!s->has_content) return NULL;
    if (s->shm_img) return s->shm_img;
    return s->dmabuf_buf ? s->dmabuf_buf->img : NULL;
}

/* HDR (gate open) only: the scene's HDR surface whose current frame is one of the game's gralloc buffers
 * that the compositor's own driver could NOT import (a 10-bit UBWC layout it does not take, say). Such a
 * frame has no draw, so neither layer_candidate() nor ahb_layer_only_candidate() (toplevels only) can
 * find it - and a Vulkan game's swapchain is a SUBSURFACE of its window. Remembered here while the scene
 * is built so it can still go on the display layer, which never needed the import. Reset per scene;
 * never set while the HDR gate is closed. */
static struct surface *g_hdr_unimported;
static int g_hdr_unimported_rect[4], g_hdr_unimported_below; /* scene x,y,w,h; draws under it */

static void note_hdr_unimported(const struct draw_list *dl, struct surface *s, int ox, int oy) {
    if (!s->dmabuf_buf || s->dmabuf_buf->img || !ahb_swapchain_has_ahb(s->dmabuf_buf)) return;
    const struct wnc_color *c = wnc_color_of(s->resource);
    if (!c || !c->dataspace) return;
    if (s->src_set && (s->src[0] != 0 || s->src[1] != 0 || (int)(s->src[2] + 0.5f) != s->buf_w ||
                       (int)(s->src[3] + 0.5f) != s->buf_h)) return; /* cropped: the layer shows whole buffers */
    int dw, dh;
    surface_size(s, &dw, &dh);
    g_hdr_unimported = s;
    g_hdr_unimported_rect[0] = ox; g_hdr_unimported_rect[1] = oy;
    g_hdr_unimported_rect[2] = dw; g_hdr_unimported_rect[3] = dh;
    g_hdr_unimported_below = dl->n;
}

/* Whether what is under the surface shows through it: its buffer has alpha, and the program - one
 * that says where its surfaces are opaque - has not said so for all of this one. */
static int surface_translucent(const struct surface *s, int w, int h) {
    if (!s->buf_alpha) return 0;
    const struct client_info *ci = client_info_of(wl_resource_get_client(s->resource));
    if (!ci || !ci->declares_opaque) return 0;
    const int *o = s->opaque;
    return !(o[2] > 0 && o[0] <= 0 && o[1] <= 0 && (long long)o[0] + o[2] >= w && (long long)o[1] + o[3] >= h);
}

static void add_surface(struct draw_list *dl, struct surface *s, int ox, int oy) {
    struct vkp_image *img = surface_image(s);
    float sx = 0, sy = 0, sw = (float)s->buf_w, sh = (float)s->buf_h;
    int dw, dh;

    if (!img) {
        if (g_zero_copy && wnc_color_hdr_open()) note_hdr_unimported(dl, s, ox, oy);
        return;
    }
    if (s->src_set) { sx = s->src[0]; sy = s->src[1]; sw = s->src[2]; sh = s->src[3]; }
    surface_size(s, &dw, &dh);
    if (dw <= 0 || dh <= 0 || sw <= 0 || sh <= 0) return;
    if (dl->n == dl->cap) {
        int cap = dl->cap ? dl->cap * 2 : 32;
        struct vkp_draw *d = realloc(dl->d, (size_t)cap * sizeof(*d));
        if (!d) return;
        dl->d = d;
        dl->cap = cap;
    }
    dl->d[dl->n++] = (struct vkp_draw){img, sx, sy, sw, sh, ox, oy, dw, dh, surface_translucent(s, dw, dh)};
    s->drawn = 1;
}

static void add_tree(struct draw_list *dl, struct surface *s, int ox, int oy, int depth) {
    struct surface *c;
    if (depth > 8) return;
    wl_list_for_each(c, &s->children, child_link)
        if (c->below_parent) add_tree(dl, c, ox + c->sub_x, oy + c->sub_y, depth + 1);
    add_surface(dl, s, ox, oy);
    wl_list_for_each(c, &s->children, child_link)
        if (!c->below_parent) add_tree(dl, c, ox + c->sub_x, oy + c->sub_y, depth + 1);
}

/* Scene size: the desktop's, or without one the largest window's (a single
 * fullscreen game then fills the screen exactly as before). */
static void scene_size(int *w, int *h) {
    struct surface *s;
    long long best = 0;

    if (g_desktop && g_desktop->has_content) {
        surface_size(g_desktop, w, h);
        g_desktop_w = *w;
        g_desktop_h = *h;
        return;
    }
    if (g_desktop && g_desktop_w > 0) { *w = g_desktop_w; *h = g_desktop_h; return; }
    *w = *h = 0;
    wl_list_for_each(s, &g_toplevels, toplevel_link) {
        int sw, sh;
        surface_size(s, &sw, &sh);
        if ((long long)sw * sh > best) { best = (long long)sw * sh; *w = sw; *h = sh; }
    }
    if (*w <= 0 || *h <= 0) { *w = INPUT_SPACE_W; *h = INPUT_SPACE_H; }
}

static void fire_all_frames(void) {
    struct surface *s;
    wl_list_for_each(s, &g_surfaces, link) fire_frames(&s->frames);
}

/* With no output window (the app is in the background, or the SurfaceView is being rebuilt)
 * nothing gets drawn, but the clients must not be left waiting on us. A FIFO swapchain blocks
 * in its present until its wp_presentation feedback and frame callback arrive, and a client
 * blocked there stops pumping messages: winhandler.exe then wedges behind it and the app
 * concludes the program has exited and closes the whole session. (DXVK's mailbox presents
 * never wait, which is why only OpenGL/Zink clients died on a background/resume.) Answer every
 * pending frame and feedback as discarded, and keep doing so until a window is back. */
static void pace_without_output(void) {
    struct surface *s;
    wl_list_for_each(s, &g_surfaces, link) {
        fire_frames(&s->frames);
        feedback_discard_all(&s->feedback);
    }
    wl_display_flush_clients(g_display);
}

static int on_frame_timer(void *data) {
    pace_without_output();
    if (vkp_has_window()) schedule_render(); /* it is back: draw the current scene */
    else if (g_frame_timer) wl_event_source_timer_update(g_frame_timer, 16);
    return 0;
}

/* Layer mode: the index of a draw that shows a whole client GPU frame over the whole scene (one
 * fullscreen game), or -1. Only the top two positions are looked at: the game must be the topmost
 * draw, or have exactly ONE draw above it, which then goes on the overlay layer (sc_layer.h).
 * Anything deeper would need more display layers than HWC will compose. */
static int layer_candidate(const struct draw_list *dl, int scene_w, int scene_h) {
    for (int i = dl->n - 1; i >= 0 && i >= dl->n - 2; i--) {
        const struct vkp_draw *d = &dl->d[i];
        if (!vkp_image_is_dmabuf(d->img)) continue;
        if (d->dx != 0 || d->dy != 0 || d->dw != scene_w || d->dh != scene_h) continue;
        if (d->sx != 0 || d->sy != 0 || (int)d->sw != vkp_image_width(d->img) ||
            (int)d->sh != vkp_image_height(d->img)) continue;
        return i;
    }
    return -1;
}

/* Zero-copy: the surface whose current GPU frame `img` is (layer_candidate found the draw). */
static struct surface *surface_for_image(const struct vkp_image *img) {
    struct surface *s;
    wl_list_for_each(s, &g_surfaces, link)
        if (s->dmabuf_buf && s->dmabuf_buf->img == img) return s;
    return NULL;
}

/* Zero-copy: the topmost window whose frame is one of the game's AHardwareBuffers that this
 * renderer could NOT import (so it has no draw), covering the whole scene at 0,0 with nothing of
 * its own above it: it can only be shown on the layer. NULL otherwise. */
static struct surface *ahb_layer_only_candidate(int scene_w, int scene_h) {
    struct surface *s;
    int w, h;
    if (wl_list_empty(&g_toplevels)) return NULL;
    s = wl_container_of(g_toplevels.prev, s, toplevel_link);
    if (!s->dmabuf_buf || s->dmabuf_buf->img || !ahb_swapchain_has_ahb(s->dmabuf_buf)) return NULL;
    if ((g_desktop && !s->placed) || s->x != 0 || s->y != 0 || s->src_set || s->dst_set) return NULL;
    if (!wl_list_empty(&s->children)) return NULL;
    if (g_hide_shell && !strcmp(client_name(wl_resource_get_client(s->resource)), "explorer.exe")) return NULL;
    surface_size(s, &w, &h);
    if (w != scene_w || h != scene_h || s->buf_w != scene_w || s->buf_h != scene_h) return NULL;
    return s;
}

/* ---- HDR (color_mgmt.h) in the scene
 * An HDR frame is only right on the game's own display layer: everything the compositor draws itself -
 * the scene blit, the effects chain, frame generation, its swapchain - is 8-bit sRGB, and a PQ-encoded
 * frame pushed through it as it stands comes out washed out. So a scene with HDR in it takes one of
 * three routes (hdr_plan):
 *   0  the HDR game ALONE on its display layer (zero-copy, or one copy) - with at most one SDR window
 *      above it on the overlay layer where this display can compose that. Best: nothing is converted.
 *   1  the HDR PICTURE: everything else without frame generation - screen effects on, a window above
 *      the game on a display that cannot take a second layer, a windowed game, zero-copy off. The whole
 *      scene is composed into ONE 10-bit PQ BT.2020 picture (SDR content placed at the SDR white; the
 *      effects run on it in 10-bit) and put on the game layer tagged BT2020_PQ (hdr_compose.h).
 *   2  frame generation: its extra frames need presents of their own, so the scene is composed into PQ
 *      and presented through an HDR10 swapchain - or, where the screen surface offers none,
 *      tone-mapped to SDR (correct colours instead of washed out).
 *   3  a frame the compositor could not import: only its display layer can show it, so effects and
 *      frame generation are skipped for it (said in the log).
 * The drawer's HDR output switch (wnc_color_output) OFF: the same routes, tone-mapped - route 0
 * becomes route 1 (the whole scene composed, tone-mapped to sRGB, onto the game layer UNtagged), route 2
 * tone-maps into an SDR swapchain, route 3 cannot (nothing can read those frames) and stays HDR, said
 * in the log. The game is told nothing; it keeps rendering HDR.
 * Nothing here runs while the HDR gate is closed. */

struct hdr_scene {
    struct vkp_hdr_frame hf;
    unsigned char *is_hdr;              /* per draw (calloc'd; free after the frame) */
    int n_hdr;
    const struct wnc_color *color;   /* the topmost HDR draw's description: what the picture is tagged */
    struct surface *who;                /* its surface, for the log */
    const char *why;                    /* route 1: why the game cannot be alone on its layer */
};

/* The HDR surface note_hdr_unimported() saw in THIS scene, when it covers the whole scene at 0,0 and
 * nothing is drawn above it: the display layer is the only place its frame can be shown. */
static struct surface *hdr_unimported_fullscreen(const struct draw_list *dl, int w, int h) {
    if (!g_hdr_unimported || g_hdr_unimported_below != dl->n) return NULL;
    if (g_hdr_unimported_rect[0] != 0 || g_hdr_unimported_rect[1] != 0 ||
        g_hdr_unimported_rect[2] != w || g_hdr_unimported_rect[3] != h) return NULL;
    return g_hdr_unimported;
}

static int is_hdr_surface(struct surface *s) {
    const struct wnc_color *c = s ? wnc_color_of(s->resource) : NULL;
    return c && c->dataspace;
}

/* Which route this scene takes (see above); fills hs. 0 whenever the gate is closed or no HDR is drawn. */
static int hdr_plan(const struct draw_list *dl, int w, int h, int fx_on, int framegen, struct hdr_scene *hs) {
    memset(hs, 0, sizeof(*hs));
    if (!wnc_color_hdr_open()) return 0;
    /* A fullscreen HDR frame this compositor could not import has no draw: layer only (route 3). */
    struct surface *un = hdr_unimported_fullscreen(dl, w, h);
    if (!un) {
        struct surface *lo = ahb_layer_only_candidate(w, h);
        if (lo && is_hdr_surface(lo)) un = lo;
    }
    if (un && g_zero_copy) { hs->color = wnc_color_of(un->resource); hs->who = un; return 3; }
    hs->hf.tonemap = !wnc_color_output();
    if (dl->n <= 0 || !(hs->is_hdr = calloc((size_t)dl->n, 1))) return 0;
    for (int i = 0; i < dl->n; i++) {
        struct surface *s = surface_for_image(dl->d[i].img);
        if (!is_hdr_surface(s)) continue;
        hs->is_hdr[i] = 1;
        hs->n_hdr++;
        hs->color = wnc_color_of(s->resource); /* later draws are above: the topmost wins */
        hs->who = s;
    }
    if (!hs->n_hdr) { free(hs->is_hdr); hs->is_hdr = NULL; return 0; }
    hs->hf.is_hdr = hs->is_hdr;
    hs->hf.color = hs->color;
    hs->hf.peak_nits = hs->color->max_cll > 0.0f ? hs->color->max_cll
                     : hs->color->has_st2086 ? hs->color->max_lum : 1000.0f;
    if (framegen) return 2;
    /* Route 0 when the game can be alone on its layer: no effects, zero-copy on, the fullscreen draw
     * is the HDR game, and above it nothing - or one window the overlay layer can carry. Not while the
     * HDR output switch is off: its frame as it stands is PQ, and only the composition can tone-map it. */
    int li = g_zero_copy ? layer_candidate(dl, w, h) : -1;
    int over = li >= 0 ? dl->n - 1 - li : 0;
    if (!hs->hf.tonemap && !fx_on && li >= 0 && hs->is_hdr[li] &&
        (over == 0 || (over == 1 && sc_layer_overlay_affordable())))
        return 0;
    hs->why = hs->hf.tonemap ? "HDR output is switched off in the drawer"
            : fx_on ? "screen effects are on"
            : !g_zero_copy ? "zero-copy presentation is off"
            : (li >= 0 && hs->is_hdr[li]) ? "a window is above the game and this display cannot compose a second layer"
            : "the HDR game is not one fullscreen window";
    return 1;
}

/* One line per change of route (and of its reason, and of the HDR output switch). */
static void hdr_note_route(int route, const struct hdr_scene *hs) {
    static int said = -1, said_tm = -1;
    static const char *said_why;
    const int tm = route != 0 && !wnc_color_output(); /* route 0 never runs with the switch off */
    if (route == said && tm == said_tm && (route != 1 || hs->why == said_why)) return;
    int was = said;
    said = route; said_why = hs->why; said_tm = tm;
    char name[160] = "the HDR game";
    if (hs->who) describe(hs->who, name, sizeof(name));
    switch (route) {
    case 1:
        if (tm)
            wnc_log("color", "tone-mapped picture for %s: %s - the whole scene is composed and tone-mapped to SDR "
                       "(HDR peak %.0f nits rolled off to SDR white, screen effects applied after it) on the game's "
                       "display layer, untagged", name, hs->why ? hs->why : "?", hs->hf.peak_nits);
        else
            wnc_log("color", "HDR picture for %s: %s - the whole scene is composed into one 10-bit PQ BT.2020 picture "
                       "(SDR content at %.0f nits, screen effects applied in 10-bit) on the game's display layer, tagged "
                       "BT2020_PQ", name, hs->why ? hs->why : "?", wnc_color_sdr_white());
        break;
    case 2:
        if (tm)
            wnc_log("color", "HDR with frame generation for %s, HDR output switched off: the scene is tone-mapped to "
                       "SDR and presented through an SDR screen swapchain", name);
        else
            wnc_log("color", "HDR with frame generation for %s: the scene is composed into PQ and presented through "
                       "the screen swapchain (HDR10 where the surface offers it, else tone-mapped to SDR)", name);
        break;
    case 3:
        if (tm)
            wnc_log("color", "HDR output is switched off, but %s's HDR frames cannot be imported by the compositor, "
                       "so nothing can tone-map them: they stay HDR on its own display layer", name);
        break; /* otherwise hdr_note_precedence says it */
    default:
        if (was == 1 || was == 2)
            wnc_log("color", "%s is back on its own display layer (no composition needed)", name);
        break;
    }
}

/* Say when effects / frame generation start or stop being skipped for an HDR game (once per change). */
static void hdr_note_precedence(struct surface *hs, int fx_on, int framegen) {
    static int fx_said = -1, fg_said = -1;
    int fx_skip = hs && fx_on, fg_skip = hs && framegen;
    char name[160] = "the game";
    if (hs) describe(hs, name, sizeof(name));
    if (fx_skip != fx_said) {
        if (fx_skip)
            wnc_log("color", "screen effects are NOT applied to %s: its HDR frames cannot be imported by the "
                       "compositor, so only its own display layer can show them (as they are)", name);
        else if (fx_said == 1)
            wnc_log("color", "screen effects apply to the scene again (no HDR game on its layer, or effects off)");
        fx_said = fx_skip;
    }
    if (fg_skip != fg_said) {
        if (fg_skip)
            wnc_log("color", "frame generation is NOT applied to %s: its HDR frames cannot be imported by the "
                       "compositor, so only its own display layer can show them (as they are)", name);
        else if (fg_said == 1)
            wnc_log("color", "frame generation applies again (no HDR game on its layer, or frame generation off)");
        fg_said = fg_skip;
    }
}

/* This scene went through the copy path: count every HDR-described surface in it, with the reason. */
static void hdr_count_copied(const struct draw_list *dl, const char *reason) {
    for (int i = 0; i < dl->n; i++) {
        struct surface *s = surface_for_image(dl->d[i].img);
        const struct wnc_color *c = s ? wnc_color_of(s->resource) : NULL;
        if (!c || !c->dataspace) continue;
        char name[160];
        describe(s, name, sizeof(name));
        wnc_color_frame_copied(name, reason);
    }
}

static void render_scene(void) {
    struct draw_list dl = {0};
    struct surface *s;
    int w, h;
    const int64_t t_scene = now_ns();
    int copy = 0; /* this scene went through the screen swapchain (the perf line's "copy") */

    g_dirty = 0;
    g_hdr_unimported = NULL; /* found again while the scene is built (HDR gate open only) */
    wl_list_for_each(s, &g_surfaces, link) s->drawn = 0;
    scene_size(&w, &h);
    if (w != g_scene_w || h != g_scene_h) {
        if (g_desktop) wnc_log("desktop", "size %dx%d", w, h);
        else wnc_log("desktop", "no desktop: showing %dx%d (largest window)", w, h);
        g_scene_w = w;
        g_scene_h = h;
    }
    if (g_desktop && !g_hide_shell) add_tree(&dl, g_desktop, 0, 0, 0);
    wl_list_for_each(s, &g_toplevels, toplevel_link) {
        if (g_desktop && !s->placed) continue; /* wait for its position */
        if (g_hide_shell && !strcmp(client_name(wl_resource_get_client(s->resource)), "explorer.exe")) continue;
        add_tree(&dl, s, s->placed ? s->x : 0, s->placed ? s->y : 0, 0);
    }

    int rendered = 0;
    int fx_on = vkp_effects_active();
    int framegen = vkp_framegen_active();
    /* HDR (see hdr_plan): route 0 = the old paths below; 1 / 2 = composed; 3 = layer only, effects and
     * frame generation skipped. Always 0 while the HDR gate is closed: nothing changes for that session. */
    struct hdr_scene hs;
    int hdr_route = hdr_plan(&dl, w, h, fx_on, framegen, &hs);
    const char *hdr_copy = NULL;           /* why this scene's HDR frames (if any) took the copy path */
    if (wnc_color_hdr_open()) {
        hdr_note_route(hdr_route, &hs);
        hdr_note_precedence(hdr_route == 3 ? hs.who : NULL, fx_on, framegen);
    }
    if (hdr_route == 3) { fx_on = 0; framegen = 0; }
    if (hdr_route == 1) {
        static int fell_back;
        vkp_apply_window_request();
        sc_layer_hide_overlay();
        if (sc_layer_present_hdr_scene(dl.d, dl.n, &hs.hf, w, h, hs.color) == 0) {
            rendered = vkp_base_black(w, h) == 0; /* black under the composed picture, presented only when it is not already */
            if (fell_back) { fell_back = 0; wnc_log("color", "the composed picture is on the game's display layer again"); }
        } else {
            hdr_route = 2; /* no layer this frame: the picture goes through the swapchain instead */
            if (!fell_back) {
                fell_back = 1;
                wnc_log("color", "the composed picture could not go on the game's display layer (no layer, or the "
                           "10-bit pass failed): it goes through the screen swapchain instead until that changes");
            }
        }
    }
    if (hdr_route == 2) {
        int how = 0;
        sc_layer_hide();
        rendered = vkp_render_hdr(w, h, dl.d, dl.n, &hs.hf, &how) == 0;
        copy = 1;
        if (rendered && how == 1) wnc_color_frame_shown(hs.color, WNC_HDR_SWAPCHAIN, 0);
        else if (rendered && how == 2) wnc_color_frame_shown(hs.color, WNC_HDR_TONEMAPPED, 0);
        else if (rendered) hdr_copy = "the HDR composition pass is unavailable on this driver";
    }
    if (hdr_route == 0 || hdr_route == 3) {
    /* What still forces the whole scene back through the compositor pass (and so through the app's
     * own swapchain), and what no longer does:
     *   - frame generation ALWAYS does: its extra frames each need a present of their own on
     *     consecutive vblanks, and a display layer latches one buffer per refresh;
     *   - screen effects do NOT any more when the game is alone on screen - the chain runs and its
     *     result is copied into the game layer's own gralloc buffer (sc_layer_present_pass), so the
     *     game keeps its layer and the display does the scene -> output mapping;
     *   - screen effects DO when a window is drawn above the game: the chain has to see the whole
     *     scene to look the way it does on the copy path, and here the game is only part of it. */
    const int pass_on = fx_on || framegen;
    int li = (g_zero_copy && !framegen) ? layer_candidate(&dl, w, h) : -1;
    int over = li >= 0 ? dl.n - 1 - li : 0; /* draws above the fullscreen game (0 or 1) */
    const int fx_blocks = (li >= 0 && fx_on && over > 0);
    if (fx_blocks) { li = -1; over = 0; }
    /* A window above the game needs a SECOND display layer, and on some displays that costs the
     * hardware composition the first layer was worth having (sc_layer_overlay_affordable). Where it
     * does, the whole scene goes down the copy path exactly as it did before the layer set existed.
     * Decided HERE, before the game is committed to a layer: deciding it after the present would
     * show the game layer and hide it again on every frame. */
    const int ov_blocks = (li >= 0 && over > 0 && !fx_blocks && !sc_layer_overlay_affordable());
    if (ov_blocks) { li = -1; over = 0; }
    const int blocked = framegen ? 1 : (fx_blocks ? 2 : (ov_blocks ? 3 : 0));
    if (g_zero_copy && blocked != g_zero_copy_paused) {
        g_zero_copy_paused = blocked;
        if (blocked == 1)
            wnc_log("framegen", "zero-copy paused: frame generation needs the compositor pass");
        else if (blocked == 2)
            wnc_log("effects", "zero-copy paused: a window above the game needs the compositor pass for the whole scene");
        else if (blocked == 3)
            wnc_log("layer", "zero-copy paused: a window above the game would need a second display layer, "
                                "which this display cannot compose in hardware - whole scene on the copy path");
        else
            wnc_log("effects", "zero-copy resumed: the game is back on its own display layer");
    }
    struct surface *ls = li >= 0 ? surface_for_image(dl.d[li].img) : NULL;
    /* A frame this renderer could not import can only be shown on the layer, effects or not. */
    if (li < 0) ls = ahb_layer_only_candidate(w, h);
    /* HDR: the same, for a game's SUBSURFACE swapchain whose 10-bit frame the compositor could not
     * import (NULL whenever the HDR gate is closed - nothing is ever noted then). */
    if (li < 0 && !ls) ls = hdr_unimported_fullscreen(&dl, w, h);
    if (ls && pass_on && li < 0 && !g_zero_copy_fx_skip_said) {
        g_zero_copy_fx_skip_said = 1;
        wnc_log(framegen ? "framegen" : "effects",
                   "the game's frames cannot be imported by the compositor: shown zero-copy, %s skipped",
                   framegen ? "frame generation" : "effects");
    }
    if (li >= 0 || ls) {
        /* Layer mode: the fullscreen window goes on the game layer and (at most) one window above
         * it on the overlay layer; whatever is under the game is hidden by it, so the screen
         * surface only needs to be black. The game's frame goes on its layer as is when it is one
         * of the game's own AHardwareBuffers (zero-copy, ahb_swapchain.c), through the effects
         * chain's result when a Look is on, else through one blit into the pool. If a layer can't
         * take its frame, the whole scene is drawn the usual way. */
        /* The layers go up FIRST and the base surface's black frame is presented after them. The
         * order matters: a present holds an acquired swapchain image, and the acquire semaphore is
         * a vblank gate - putting the effects chain or the layer transaction behind it costs a
         * whole refresh (measured on the Pocket FIT: 72 fps that way, 111 fps this way, with Retro
         * CRT at 1920x1080). The window request the app may have left is applied here instead,
         * since vkp_render no longer runs before the layer path. */
        vkp_apply_window_request();
        int r = -1;
        /* The game's own buffer can only go on the layer AS IS - with a Look on, the frame has to
         * go through the chain first, so the raw buffer is skipped in favour of the pass (a frame
         * this renderer could not import has no draw, li < 0, and stays zero-copy with the effects
         * skipped, said once above). */
        if (!(fx_on && li >= 0) && ls && ls->dmabuf_buf && ahb_swapchain_has_ahb(ls->dmabuf_buf))
            r = ahb_swapchain_present(ls->dmabuf_buf, ls, w, h);
        if (r != 0 && li >= 0)
            r = fx_on ? sc_layer_present_pass(&dl.d[li], 1, w, h)
                      : sc_layer_present(dl.d[li].img, w, h, ls ? wnc_surface_color(ls) : NULL);
        if (r == 0) {
            if (ls) ls->drawn = 1;
            /* The one window above the game keeps the game off the copy path entirely: it goes on
             * its own layer, cropped and placed by the display. */
            int go[8], ov = 0;
            if (over == 1 && li >= 0 && vkp_map_draw(&dl.d[li + 1], go))
                ov = sc_layer_present_overlay(dl.d[li + 1].img, go) == 0 ? 1 : -1;
            if (ov <= 0) sc_layer_hide_overlay();
            if (ov < 0) { /* the overlay layer refused it: draw the whole scene the old way */
                sc_layer_hide();
                rendered = vkp_render(w, h, dl.d, dl.n) == 0;
                copy = 1;
                hdr_copy = "the overlay layer refused the window above it";
            } else {
                /* Black under the opaque layers. It is a plain black frame (no effects, no frame
                 * generation on it: they would only run on black), and it is presented only when the
                 * base surface does not already show one - entering layer mode, after a copy-path or
                 * frame-generation frame, after a swapchain rebuild. Every other layer frame draws and
                 * presents nothing here: the display layers are all that changes on screen. This loop
                 * is paced by the Choreographer's ticks (on_vsync), not by this present. */
                rendered = vkp_base_black(w, h) == 0;
            }
        } else {
            rendered = vkp_render(w, h, dl.d, dl.n) == 0;
            copy = 1;
            hdr_copy = "its display layer was unavailable this frame";
        }
    } else {
        sc_layer_hide();
        rendered = vkp_render(w, h, dl.d, dl.n) == 0;
        copy = 1;
        hdr_copy = ov_blocks ? "a window is above it and this display cannot compose a second layer in hardware"
                 : fx_blocks ? "a window is above it and screen effects need the whole scene"
                 : !g_zero_copy ? "zero-copy presentation is off"
                 : framegen ? "frame generation needs the compositor pass"
                 : "it is not the one fullscreen window (only that one gets its own display layer)";
    }
    } /* routes 0 and 3 */
    free(hs.is_hdr);
    if (hdr_copy && rendered && wnc_color_hdr_open()) hdr_count_copied(&dl, hdr_copy);
    if (rendered) {
        int64_t t = now_ns();
        g_stat_frames++;
        /* A surface outside the scene (role-less, not placed yet, a hidden helper window such
         * as wined3d's device window) was not shown, so its feedback is discarded rather than
         * left pending: a FIFO present waits on it, and a client blocked there never commits
         * again. */
        wl_list_for_each(s, &g_surfaces, link) {
            if (s->drawn) feedback_present_all(&s->feedback, t);
            else feedback_discard_all(&s->feedback);
        }
        fire_all_frames();
    } else {
        /* No output surface yet (or it went away): keep clients paced without it, now and on a
         * timer that re-arms itself until a window is back (see pace_without_output). */
        pace_without_output();
        if (!g_frame_timer)
            g_frame_timer = wl_event_loop_add_timer(wl_display_get_event_loop(g_display),
                                                    on_frame_timer, NULL);
        if (g_frame_timer) wl_event_source_timer_update(g_frame_timer, 16);
    }
    free(dl.d);
    wl_display_flush_clients(g_display);
    if (rendered && copy) g_perf.copy_scenes++;
    g_perf.scenes++;
    const int64_t scene_ns = now_ns() - t_scene;
    g_perf.scene_ns += scene_ns;
    if (scene_ns > g_perf.scene_max_ns) g_perf.scene_max_ns = scene_ns;
}

static int on_fallback_timer(void *data) {
    g_fallback_armed = 0;
    if (g_dirty) render_scene();
    return 0;
}

static void schedule_render(void) {
    g_dirty = 1;
    if (!g_display) return;
    /* With vsync ticks flowing the next tick renders; otherwise render on a timer. */
    if (g_last_vsync_ns && now_ns() - g_last_vsync_ns < 100000000LL) return;
    if (g_fallback_armed) return;
    if (!g_fallback_timer)
        g_fallback_timer = wl_event_loop_add_timer(wl_display_get_event_loop(g_display),
                                                   on_fallback_timer, NULL);
    if (!g_fallback_timer) return;
    g_fallback_armed = 1;
    wl_event_source_timer_update(g_fallback_timer, 8);
}

/* A screen refresh (Choreographer tick): draw the newest state once. */
static void on_vsync(int64_t frame_time_ns) {
    int64_t now = now_ns();
    g_perf.ticks++;
    if (g_last_vsync_ns) {
        int64_t d = now - g_last_vsync_ns;
        if (d > 3000000LL && d < 40000000LL) g_refresh_ns = (g_refresh_ns * 7 + d) / 8;
    }
    g_last_vsync_ns = now;
    (void)frame_time_ns;
    /* The app's surface may have been replaced or taken away since the last frame: apply that now
     * (the app never waits for us), and redraw the scene onto a new one. */
    if (vkp_apply_window_request()) g_dirty = 1;
    /* A screen-effect setting changed (JNI, any thread): redraw so it shows on a static scene too. */
    if (vkp_effects_sync()) g_dirty = 1;
    if (g_dirty) render_scene();
}

/* ------------------------------------------------------------------ wl_seat */

static uint32_t now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

static void pointer_set_cursor(struct wl_client *c, struct wl_resource *r, uint32_t serial,
                               struct wl_resource *surface, int32_t hx, int32_t hy) {
    /* The app draws its own pointer; cursor surfaces stay role-less and aren't drawn. */
}
static void pointer_release(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static const struct wl_pointer_interface pointer_impl = {
    .set_cursor = pointer_set_cursor, .release = pointer_release,
};
static void pointer_res_destroy(struct wl_resource *r) {
    constraints_pointer_gone(r);
    relative_pointers_pointer_gone(r);
    for (int i = 0; i < g_nptrs; i++)
        if (g_ptrs[i].ptr == r) { g_ptrs[i] = g_ptrs[--g_nptrs]; break; }
}

static void keyboard_release(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static const struct wl_keyboard_interface keyboard_impl = { .release = keyboard_release };
static void keyboard_res_destroy(struct wl_resource *r) {
    for (int i = 0; i < g_nkbs; i++)
        if (g_kbs[i].kb == r) { g_kbs[i] = g_kbs[--g_nkbs]; break; }
}
static void touch_release(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static const struct wl_touch_interface touch_impl = { .release = touch_release };

static void seat_get_pointer(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct wl_resource *p = wl_resource_create(c, &wl_pointer_interface, wl_resource_get_version(r), id);
    if (!p) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(p, &pointer_impl, NULL, pointer_res_destroy);
    if (g_nptrs < MAX_PTRS) { g_ptrs[g_nptrs].ptr = p; g_ptrs[g_nptrs].focus = NULL; g_nptrs++; }
}
static void seat_get_keyboard(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct wl_resource *k = wl_resource_create(c, &wl_keyboard_interface, wl_resource_get_version(r), id);
    if (!k) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(k, &keyboard_impl, NULL, keyboard_res_destroy);
    if (g_nkbs < MAX_PTRS) { g_kbs[g_nkbs].kb = k; g_kbs[g_nkbs].focus = NULL; g_nkbs++; }
    /* Send the xkb keymap the app extracted to $XDG_RUNTIME_DIR/keymap.xkb. Without a keymap the
     * client can't interpret our evdev key codes. */
    const char *rt = getenv("XDG_RUNTIME_DIR");
    char path[512];
    snprintf(path, sizeof(path), "%s/keymap.xkb", rt ? rt : ".");
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0)
            wl_keyboard_send_keymap(k, WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1, fd, (uint32_t)st.st_size);
        close(fd);
    } else {
        WLOGE("keymap open failed: %s", path);
    }
    if (wl_resource_get_version(k) >= WL_KEYBOARD_REPEAT_INFO_SINCE_VERSION)
        wl_keyboard_send_repeat_info(k, 25, 500);
}
static void seat_get_touch(struct wl_client *c, struct wl_resource *r, uint32_t id) {
    struct wl_resource *t = wl_resource_create(c, &wl_touch_interface, wl_resource_get_version(r), id);
    if (t) wl_resource_set_implementation(t, &touch_impl, NULL, NULL);
}
static void seat_release(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static const struct wl_seat_interface seat_impl = {
    .get_pointer = seat_get_pointer, .get_keyboard = seat_get_keyboard,
    .get_touch = seat_get_touch, .release = seat_release,
};
static void bind_seat(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &wl_seat_interface, ver, id);
    wl_resource_set_implementation(r, &seat_impl, NULL, NULL);
    wl_seat_send_capabilities(r, WL_SEAT_CAPABILITY_POINTER | WL_SEAT_CAPABILITY_KEYBOARD);
    if (ver >= 2) wl_seat_send_name(r, "winnative-seat");
}

/* A client may hold several wl_pointer/wl_keyboard objects (gamescope reads input on its own
 * thread through a second pair): events go to every one of them. */
static int client_has_pointer(struct wl_client *client) {
    for (int i = 0; i < g_nptrs; i++)
        if (wl_resource_get_client(g_ptrs[i].ptr) == client) return 1;
    return 0;
}
static int client_has_keyboard(struct wl_client *client) {
    for (int i = 0; i < g_nkbs; i++)
        if (wl_resource_get_client(g_kbs[i].kb) == client) return 1;
    return 0;
}
static void pointer_frame(struct wl_resource *ptr) {
    if (wl_resource_get_version(ptr) >= WL_POINTER_FRAME_SINCE_VERSION) wl_pointer_send_frame(ptr);
}

/* Topmost window whose scene rectangle contains the point (no-desktop fallback). */
static struct surface *toplevel_at(double x, double y) {
    struct surface *s;
    wl_list_for_each_reverse(s, &g_toplevels, toplevel_link) {
        int w, h, sx = s->placed ? s->x : 0, sy = s->placed ? s->y : 0;
        surface_size(s, &w, &h);
        if (x >= sx && y >= sy && x < sx + w && y < sy + h) return s;
    }
    return NULL;
}

static void pointer_focus(struct wl_resource *target, wl_fixed_t fx, wl_fixed_t fy) {
    struct wl_client *client = wl_resource_get_client(target);
    int entered = 0;
    for (int i = 0; i < g_nptrs; i++) {
        if (!g_ptrs[i].focus || g_ptrs[i].focus == target) continue;
        wl_pointer_send_leave(g_ptrs[i].ptr, wl_display_next_serial(g_display), g_ptrs[i].focus);
        pointer_frame(g_ptrs[i].ptr);
        g_ptrs[i].focus = NULL;
    }
    for (int i = 0; i < g_nptrs; i++) {
        struct seat_pointer *sp = &g_ptrs[i];
        if (wl_resource_get_client(sp->ptr) != client || sp->focus == target) continue;
        sp->focus = target;
        wl_pointer_send_enter(sp->ptr, wl_display_next_serial(g_display), target, fx, fy);
        entered = 1;
    }
    if (entered) constraints_focus_entered(target, client);
}

/* The seat's modifiers in the keymap's real-modifier bits. Wine and gamescope work them out from
 * the keys they are sent; a nested wlroots compositor (the desktop's labwc) takes them from
 * wl_keyboard.modifiers alone, and without it Shift never reaches its programs. */
static uint32_t g_mod_keys_held, g_mods_locked;

static uint32_t mods_depressed(void) {
    static const uint32_t bits[] = { 1u << 0, 1u << 0, 1u << 2, 1u << 2, 1u << 3, 1u << 3, 1u << 6, 1u << 6 };
    uint32_t mods = 0;
    for (unsigned i = 0; i < sizeof(bits) / sizeof(bits[0]); i++)
        if (g_mod_keys_held & (1u << i)) mods |= bits[i];
    return mods;
}

/* Returns whether the key changed the seat's modifiers. */
static int mods_key(uint32_t evdev, int pressed) {
    static const uint32_t keys[] = { 42, 54, 29, 97, 56, 100, 125, 126 }; /* L/R Shift, Ctrl, Alt, Meta */
    if (evdev == 58) { /* Caps Lock toggles on its press */
        if (pressed) g_mods_locked ^= 1u << 1;
        return pressed;
    }
    for (unsigned i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        if (keys[i] != evdev) continue;
        uint32_t held = pressed ? g_mod_keys_held | (1u << i) : g_mod_keys_held & ~(1u << i);
        if (held == g_mod_keys_held) return 0;
        g_mod_keys_held = held;
        return 1;
    }
    return 0;
}

static void keyboard_focus(struct wl_resource *target) {
    struct wl_client *client = wl_resource_get_client(target);
    int entered = 0;
    struct wl_array keys;
    wl_array_init(&keys);
    for (int i = 0; i < g_nkbs; i++) {
        struct seat_keyboard *sk = &g_kbs[i];
        if (sk->focus == target) continue;
        if (sk->focus) {
            wl_keyboard_send_leave(sk->kb, wl_display_next_serial(g_display), sk->focus);
            sk->focus = NULL;
        }
        if (wl_resource_get_client(sk->kb) != client) continue;
        sk->focus = target;
        wl_keyboard_send_enter(sk->kb, wl_display_next_serial(g_display), target, &keys);
        wl_keyboard_send_modifiers(sk->kb, wl_display_next_serial(g_display), mods_depressed(), 0, g_mods_locked, 0);
        entered = 1;
    }
    wl_array_release(&keys);
    if (entered) wnc_clipboard_keyboard_focus(client); /* wl_data_device selection follows focus */
}

/* ------------------------------------------------------------------ pointer constraints
 * zwp_pointer_constraints_v1 and zwp_relative_pointer_manager_v1: what winewayland uses for
 * SetCursorPos, ClipCursor and hidden-cursor (mouse-look) games.
 *
 * A lock freezes the pointer where it is: no wl_pointer.motion is sent while it holds, and
 * every input delta (a relative event from the app, or the difference between absolute
 * positions) reaches the locking program as zwp_relative_pointer_v1.relative_motion. A confine
 * clamps the pointer to the surface (and the program's region inside it). Relative motion is
 * also sent while unlocked, to any program holding a relative pointer.
 *
 * Input normally goes to the desktop surface; while a constraint holds, pointer focus moves
 * to the constrained surface, so the constraining program receives the events itself
 * (winewayland only enables relative motion for a window its pointer has entered). One
 * constraint holds at a time; a newer request ends the older one. A lock's cursor position
 * hint is applied on the surface's commit and is where the pointer ends up when the lock
 * ends (winewayland's SetCursorPos: lock, hint, commit, unlock). */

struct constraint {
    struct wl_list link;                    /* g_constraints */
    struct wl_resource *resource;           /* zwp_locked_pointer_v1 or zwp_confined_pointer_v1 */
    struct wl_resource *pointer;            /* the program's wl_pointer; NULL once released */
    struct surface *surface;                /* NULL once the surface is gone */
    int is_lock;
    uint32_t lifetime;
    int active;
    int defunct;                            /* a oneshot that ended, or surface/pointer gone */
    struct region region, pending_region;   /* confine area, surface-local; unset = whole surface */
    int pending_region_set;
    int hint_set, pending_hint_set;         /* lock: cursor position hint, surface-local */
    double hint_x, hint_y, pending_hint_x, pending_hint_y;
};
static struct wl_list g_constraints;
static struct constraint *g_active_constraint;

struct relative_pointer {
    struct wl_list link;                    /* g_relative_pointers */
    struct wl_resource *resource;
    struct wl_resource *pointer;            /* the program's wl_pointer; NULL once released */
};
static struct wl_list g_relative_pointers;

/* Last pointer position in scene coordinates (buttons and scrolls from the app's X-server
 * input path arrive without one). */
static double g_ptr_x, g_ptr_y;
/* Last absolute input position: absolute input becomes deltas while the pointer is locked. */
static double g_raw_x, g_raw_y;
static int g_raw_valid;

/* JNI: the app switches its touch/mouse path to deltas while a lock holds and re-syncs its
 * pointer to x,y when it ends. */
extern void wnc_on_pointer_lock(int locked, int x, int y);

static void sync_lock_notify(void) {
    static int last;
    int locked = g_active_constraint && g_active_constraint->is_lock;
    if (locked == last) return;
    last = locked;
    wnc_on_pointer_lock(locked, (int)g_ptr_x, (int)g_ptr_y);
}

static void constraint_describe(const struct constraint *k, char *out, size_t size) {
    if (k->surface) describe(k->surface, out, size);
    else snprintf(out, size, "a closed window");
}

/* Scene position of the constrained surface. */
static void constraint_origin(const struct constraint *k, int *x, int *y) {
    *x = *y = 0;
    if (k->surface && k->surface != g_desktop && k->surface->placed) { *x = k->surface->x; *y = k->surface->y; }
}

/* Keep x,y (scene coordinates) inside the confine area. */
static void confine_clamp(const struct constraint *k, double *x, double *y) {
    int ox, oy, w, h, rx = 0, ry = 0, rw, rh;
    if (!k->surface) return;
    constraint_origin(k, &ox, &oy);
    surface_size(k->surface, &w, &h);
    rw = w; rh = h;
    if (k->region.set) {
        int x2 = rx + rw < k->region.x + k->region.w ? rx + rw : k->region.x + k->region.w;
        int y2 = ry + rh < k->region.y + k->region.h ? ry + rh : k->region.y + k->region.h;
        if (k->region.x > rx) rx = k->region.x;
        if (k->region.y > ry) ry = k->region.y;
        rw = x2 - rx; rh = y2 - ry;
    }
    if (rw <= 0 || rh <= 0) return;
    if (*x < ox + rx) *x = ox + rx;
    if (*x > ox + rx + rw - 1) *x = ox + rx + rw - 1;
    if (*y < oy + ry) *y = oy + ry;
    if (*y > oy + ry + rh - 1) *y = oy + ry + rh - 1;
}

static void constraint_send_state(struct constraint *k, int on) {
    if (k->is_lock) {
        if (on) zwp_locked_pointer_v1_send_locked(k->resource);
        else zwp_locked_pointer_v1_send_unlocked(k->resource);
    } else {
        if (on) zwp_confined_pointer_v1_send_confined(k->resource);
        else zwp_confined_pointer_v1_send_unconfined(k->resource);
    }
}

/* A constraint stops holding. tell_client = 0 when its resource is being destroyed. */
static void constraint_end(struct constraint *k, int tell_client, const char *why) {
    char name[160];
    if (!k->active) return;
    k->active = 0;
    if (g_active_constraint == k) g_active_constraint = NULL;
    if (k->lifetime == ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_ONESHOT) k->defunct = 1;
    if (k->is_lock && k->hint_set && k->surface) {
        int ox, oy;
        constraint_origin(k, &ox, &oy);
        g_ptr_x = ox + k->hint_x;
        g_ptr_y = oy + k->hint_y;
    }
    g_raw_valid = 0;
    if (tell_client) constraint_send_state(k, 0);
    constraint_describe(k, name, sizeof(name));
    wnc_log("pointer", "%s: %s (%s), pointer at %d,%d", k->is_lock ? "unlocked" : "unconfined",
               name, why, (int)g_ptr_x, (int)g_ptr_y);
    sync_lock_notify();
}

static void constraint_activate(struct constraint *k) {
    char name[160];
    int ox, oy;
    if (k->active || k->defunct || !k->surface || !k->pointer) return;
    if (g_active_constraint && g_active_constraint != k)
        constraint_end(g_active_constraint, 1, "replaced by a newer request");
    k->active = 1;
    g_active_constraint = k;
    if (!k->is_lock) confine_clamp(k, &g_ptr_x, &g_ptr_y);
    /* Pointer focus follows the constraint: the program gets the events itself. */
    constraint_origin(k, &ox, &oy);
    pointer_focus(k->surface->resource, wl_fixed_from_double(g_ptr_x - ox), wl_fixed_from_double(g_ptr_y - oy));
    constraint_send_state(k, 1);
    constraint_describe(k, name, sizeof(name));
    if (k->is_lock)
        wnc_log("pointer", "locked: %s (%s), pointer frozen at %d,%d", name,
                   k->lifetime == ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_ONESHOT ? "oneshot" : "persistent",
                   (int)g_ptr_x, (int)g_ptr_y);
    else if (k->region.set)
        wnc_log("pointer", "confined: %s (%s) to %dx%d at %d,%d of the window", name,
                   k->lifetime == ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_ONESHOT ? "oneshot" : "persistent",
                   k->region.w, k->region.h, k->region.x, k->region.y);
    else
        wnc_log("pointer", "confined: %s (%s) to the whole window", name,
                   k->lifetime == ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_ONESHOT ? "oneshot" : "persistent");
    sync_lock_notify();
    wl_display_flush_clients(g_display);
}

/* Pointer focus entered target: a persistent constraint waiting on it takes hold again. */
static void constraints_focus_entered(struct wl_resource *target, struct wl_client *client) {
    struct constraint *k;
    if (g_active_constraint) return;
    wl_list_for_each(k, &g_constraints, link) {
        if (k->active || k->defunct || !k->surface || !k->pointer) continue;
        if (k->surface->resource != target || wl_resource_get_client(k->pointer) != client) continue;
        constraint_activate(k);
        return;
    }
}

static void constraints_surface_gone(struct surface *s) {
    struct constraint *k;
    wl_list_for_each(k, &g_constraints, link) {
        if (k->surface != s) continue;
        constraint_end(k, 1, "window closed");
        k->surface = NULL;
        k->defunct = 1;
    }
}

static void constraints_pointer_gone(struct wl_resource *pointer) {
    struct constraint *k;
    wl_list_for_each(k, &g_constraints, link) {
        if (k->pointer != pointer) continue;
        constraint_end(k, 1, "pointer released");
        k->pointer = NULL;
        k->defunct = 1;
    }
}

/* Double-buffered state (position hint, confine region) applies on the surface's commit. */
static void constraints_surface_commit(struct surface *s) {
    struct constraint *k;
    wl_list_for_each(k, &g_constraints, link) {
        if (k->surface != s) continue;
        if (k->pending_hint_set) {
            k->hint_set = 1;
            k->hint_x = k->pending_hint_x;
            k->hint_y = k->pending_hint_y;
            k->pending_hint_set = 0;
            if (k->active && k->is_lock) {
                int ox, oy;
                constraint_origin(k, &ox, &oy);
                g_ptr_x = ox + k->hint_x;
                g_ptr_y = oy + k->hint_y;
                if (log_budget())
                    wnc_log("pointer", "position hint: pointer moved to %d,%d", (int)g_ptr_x, (int)g_ptr_y);
            }
        }
        if (k->pending_region_set) {
            k->region = k->pending_region;
            k->pending_region_set = 0;
            if (k->active && !k->is_lock) confine_clamp(k, &g_ptr_x, &g_ptr_y);
        }
    }
}

static void constraint_resource_destroy(struct wl_resource *r) {
    struct constraint *k = wl_resource_get_user_data(r);
    if (!k) return;
    constraint_end(k, 0, "released by the program");
    wl_list_remove(&k->link);
    free(k);
}

static void constraint_destroy_req(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void locked_pointer_set_cursor_position_hint(struct wl_client *c, struct wl_resource *r,
                                                    wl_fixed_t x, wl_fixed_t y) {
    struct constraint *k = wl_resource_get_user_data(r);
    if (!k) return;
    k->pending_hint_set = 1;
    k->pending_hint_x = wl_fixed_to_double(x);
    k->pending_hint_y = wl_fixed_to_double(y);
}
static void locked_pointer_set_region(struct wl_client *c, struct wl_resource *r, struct wl_resource *region) {
    /* A lock's region only says where it may take hold; a lock here takes hold anywhere. */
}
static const struct zwp_locked_pointer_v1_interface locked_pointer_impl = {
    .destroy = constraint_destroy_req,
    .set_cursor_position_hint = locked_pointer_set_cursor_position_hint,
    .set_region = locked_pointer_set_region,
};
static void confined_pointer_set_region(struct wl_client *c, struct wl_resource *r, struct wl_resource *region) {
    struct constraint *k = wl_resource_get_user_data(r);
    struct region *rg = region ? wl_resource_get_user_data(region) : NULL;
    if (!k) return;
    k->pending_region_set = 1;
    if (rg) k->pending_region = *rg;
    else memset(&k->pending_region, 0, sizeof(k->pending_region));
}
static const struct zwp_confined_pointer_v1_interface confined_pointer_impl = {
    .destroy = constraint_destroy_req,
    .set_region = confined_pointer_set_region,
};

static void constraint_create(struct wl_client *c, struct wl_resource *r, uint32_t id,
                              struct wl_resource *surface_res, struct wl_resource *pointer_res,
                              struct wl_resource *region_res, uint32_t lifetime, int is_lock) {
    struct surface *s = wl_resource_get_user_data(surface_res);
    struct constraint *k;
    struct wl_resource *res;
    char name[160];

    wl_list_for_each(k, &g_constraints, link) {
        if (k->defunct || k->surface != s || k->pointer != pointer_res) continue;
        wl_resource_post_error(r, ZWP_POINTER_CONSTRAINTS_V1_ERROR_ALREADY_CONSTRAINED,
                               "the surface already has a pointer constraint for this pointer");
        return;
    }
    res = wl_resource_create(c, is_lock ? &zwp_locked_pointer_v1_interface : &zwp_confined_pointer_v1_interface,
                             wl_resource_get_version(r), id);
    if (!res || !(k = calloc(1, sizeof(*k)))) { wl_client_post_no_memory(c); return; }
    k->resource = res;
    k->pointer = pointer_res;
    k->surface = s;
    k->is_lock = is_lock;
    k->lifetime = lifetime == ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT
                  ? ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_PERSISTENT : ZWP_POINTER_CONSTRAINTS_V1_LIFETIME_ONESHOT;
    if (!is_lock && region_res) {
        struct region *rg = wl_resource_get_user_data(region_res);
        if (rg) k->region = *rg;
    }
    wl_resource_set_implementation(res, is_lock ? (const void *)&locked_pointer_impl : (const void *)&confined_pointer_impl,
                                   k, constraint_resource_destroy);
    wl_list_insert(g_constraints.prev, &k->link);
    if (!s) { k->defunct = 1; return; }
    constraint_describe(k, name, sizeof(name));
    wnc_log("pointer", "%s requested by %s for %s", is_lock ? "lock" : "confine", client_name(c), name);
    constraint_activate(k);
}

static void pointer_constraints_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void pointer_constraints_lock_pointer(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                             struct wl_resource *surface, struct wl_resource *pointer,
                                             struct wl_resource *region, uint32_t lifetime) {
    constraint_create(c, r, id, surface, pointer, region, lifetime, 1);
}
static void pointer_constraints_confine_pointer(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                                struct wl_resource *surface, struct wl_resource *pointer,
                                                struct wl_resource *region, uint32_t lifetime) {
    constraint_create(c, r, id, surface, pointer, region, lifetime, 0);
}
static const struct zwp_pointer_constraints_v1_interface pointer_constraints_impl = {
    .destroy = pointer_constraints_destroy,
    .lock_pointer = pointer_constraints_lock_pointer,
    .confine_pointer = pointer_constraints_confine_pointer,
};
static void bind_pointer_constraints(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &zwp_pointer_constraints_v1_interface, ver, id);
    if (!r) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(r, &pointer_constraints_impl, NULL, NULL);
}

/* --- relative pointers --- */

static void relative_pointer_resource_destroy(struct wl_resource *r) {
    struct relative_pointer *rp = wl_resource_get_user_data(r);
    if (!rp) return;
    wnc_log("pointer", "relative pointer released by %s", client_name(wl_resource_get_client(r)));
    wl_list_remove(&rp->link);
    free(rp);
}
static void relative_pointer_destroy_req(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static const struct zwp_relative_pointer_v1_interface relative_pointer_impl = {
    .destroy = relative_pointer_destroy_req,
};

static void relative_pointer_manager_destroy(struct wl_client *c, struct wl_resource *r) { wl_resource_destroy(r); }
static void relative_pointer_manager_get(struct wl_client *c, struct wl_resource *r, uint32_t id,
                                         struct wl_resource *pointer) {
    struct wl_resource *res = wl_resource_create(c, &zwp_relative_pointer_v1_interface, wl_resource_get_version(r), id);
    struct relative_pointer *rp;
    if (!res || !(rp = calloc(1, sizeof(*rp)))) { wl_client_post_no_memory(c); return; }
    rp->resource = res;
    rp->pointer = pointer;
    wl_resource_set_implementation(res, &relative_pointer_impl, rp, relative_pointer_resource_destroy);
    wl_list_insert(g_relative_pointers.prev, &rp->link);
    wnc_log("pointer", "relative pointer created for %s (motion arrives as deltas)", client_name(c));
}
static const struct zwp_relative_pointer_manager_v1_interface relative_pointer_manager_impl = {
    .destroy = relative_pointer_manager_destroy,
    .get_relative_pointer = relative_pointer_manager_get,
};
static void bind_relative_pointer_manager(struct wl_client *c, void *data, uint32_t ver, uint32_t id) {
    struct wl_resource *r = wl_resource_create(c, &zwp_relative_pointer_manager_v1_interface, ver, id);
    if (!r) { wl_client_post_no_memory(c); return; }
    wl_resource_set_implementation(r, &relative_pointer_manager_impl, NULL, NULL);
}

static void relative_pointers_pointer_gone(struct wl_resource *pointer) {
    struct relative_pointer *rp;
    wl_list_for_each(rp, &g_relative_pointers, link)
        if (rp->pointer == pointer) rp->pointer = NULL;
}

/* relative_motion to every relative pointer of the program owning the focus (winewayland wants
 * it whenever it holds one, locked or not). Returns whether any was sent. */
static int send_relative_motion(struct wl_client *client, double dx, double dy) {
    struct relative_pointer *rp;
    struct timespec ts;
    uint64_t ut;
    int sent = 0;
    if (dx == 0 && dy == 0) return 0;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ut = (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000;
    wl_list_for_each(rp, &g_relative_pointers, link) {
        if (!rp->pointer || wl_resource_get_client(rp->pointer) != client) continue;
        zwp_relative_pointer_v1_send_relative_motion(rp->resource, (uint32_t)(ut >> 32), (uint32_t)ut,
                                                     wl_fixed_from_double(dx), wl_fixed_from_double(dy),
                                                     wl_fixed_from_double(dx), wl_fixed_from_double(dy));
        sent = 1;
    }
    return sent;
}

/* ------------------------------------------------------------------ pointer events */

/* One pointer event: a motion, then an optional button change (button 0 = none). Absolute
 * x,y are scene coordinates; with `relative` they are a delta. While a lock holds the pointer
 * stays put and the delta (or the change between absolute positions) becomes relative motion
 * for the locking program; a confine keeps the pointer inside its area. */
static void pointer_input(double x, double y, int relative, uint32_t button, int pressed) {
    struct surface *target;
    struct constraint *k = g_active_constraint;
    double dx, dy;

    if (relative) {
        dx = x; dy = y;
    } else {
        dx = g_raw_valid ? x - g_raw_x : 0;
        dy = g_raw_valid ? y - g_raw_y : 0;
        g_raw_x = x; g_raw_y = y; g_raw_valid = 1;
    }

    if (k && k->is_lock) {
        target = k->surface;
    } else {
        if (relative) {
            int w, h;
            scene_size(&w, &h);
            x = g_ptr_x + dx; y = g_ptr_y + dy;
            if (x < 0) x = 0;
            if (y < 0) y = 0;
            if (x > w - 1) x = w - 1;
            if (y > h - 1) y = h - 1;
        }
        if (k) confine_clamp(k, &x, &y);
        g_ptr_x = x; g_ptr_y = y;
        if (k) {
            target = k->surface;
        } else if (g_desktop) {
            target = g_desktop;
        } else {
            target = g_grab ? g_grab : toplevel_at(x, y);
            if (button && pressed) { g_grab = target; g_key_target = target; }
            if (button && !pressed) g_grab = NULL;
        }
    }
    if (!target) return;

    struct wl_client *client = wl_resource_get_client(target->resource);
    if (!client_has_pointer(client)) return;
    int tx = 0, ty = 0;
    if (target != g_desktop && target->placed) { tx = target->x; ty = target->y; }
    wl_fixed_t fx = wl_fixed_from_double(g_ptr_x - tx), fy = wl_fixed_from_double(g_ptr_y - ty);
    uint32_t t = now_ms();

    pointer_focus(target->resource, fx, fy);
    for (int i = 0; i < g_nptrs; i++) {
        struct wl_resource *ptr = g_ptrs[i].ptr;
        if (wl_resource_get_client(ptr) != client) continue;
        if (!(k && k->is_lock)) wl_pointer_send_motion(ptr, t, fx, fy);
        if (button)
            wl_pointer_send_button(ptr, wl_display_next_serial(g_display), t, button,
                                   pressed ? WL_POINTER_BUTTON_STATE_PRESSED : WL_POINTER_BUTTON_STATE_RELEASED);
        pointer_frame(ptr);
    }
    send_relative_motion(client, dx, dy);
    /* A click lands text input (the soft keyboard's IME text) on the window under the pointer:
     * that's where Wine's focus goes, and winewayland routes IME updates per process. */
    if (button && pressed && !(k && k->is_lock)) {
        struct surface *clicked = toplevel_at(g_ptr_x, g_ptr_y);
        if (clicked) g_ime_click = clicked;
        wnc_text_input_refocus();
    }
    wl_display_flush_clients(g_display);
}

/* Absolute motion to scene x,y, then an optional button change. */
static void pointer_event(double x, double y, uint32_t button, int pressed) {
    pointer_input(x, y, 0, button, pressed);
}

/* A delta from the app's relative input path (Relative Mouse, captured mouse, stick-as-mouse). */
static void pointer_delta(double dx, double dy) {
    pointer_input(dx, dy, 1, 0, 0);
}

/* A button change at the current pointer position. */
static void pointer_button(uint32_t button, int pressed) {
    pointer_input(0, 0, 1, button, pressed);
}

/* Java touch events arrive in INPUT_SPACE over the whole output: action 0=press, 1=move, 2=release.
 * Output pixels go through the inverse of the scale mode's mapping (letterbox bars, FILL crop, a
 * TOP/BOTTOM half), so the touch lands on the scene pixel that is drawn under the finger. */
static void deliver_pointer(const struct input_msg *m) {
    int w, h, ow, oh;
    double x, y;
    scene_size(&w, &h);
    vkp_output_size(&ow, &oh);
    if (ow <= 0 || oh <= 0 ||
        !vkp_output_to_scene((double)m->p2 * ow / INPUT_SPACE_W, (double)m->p3 * oh / INPUT_SPACE_H, &x, &y)) {
        x = (double)m->p2 * w / INPUT_SPACE_W; /* nothing drawn yet: plain stretch */
        y = (double)m->p3 * h / INPUT_SPACE_H;
    }
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > w - 1) x = w - 1;
    if (y > h - 1) y = h - 1;
    pointer_event(x, y, m->p1 == 1 ? 0 : BTN_LEFT, m->p1 == 0);
}

static void key_event(uint32_t evdev, int pressed);
struct wl_resource *wnc_ime_target(void);
static void deliver_key(const struct input_msg *m) {
    key_event((uint32_t)m->p1, m->p2);
}

/* Vertical wheel steps (negative = up) at the current pointer position. */
static void scroll_event(int steps) {
    struct surface *target = g_active_constraint ? g_active_constraint->surface
                           : g_desktop ? g_desktop : (g_grab ? g_grab : toplevel_at(g_ptr_x, g_ptr_y));
    if (!target || !steps) return;
    struct wl_client *client = wl_resource_get_client(target->resource);
    if (!client_has_pointer(client)) return;
    int tx = 0, ty = 0;
    if (target != g_desktop && target->placed) { tx = target->x; ty = target->y; }
    uint32_t t = now_ms();
    pointer_focus(target->resource, wl_fixed_from_double(g_ptr_x - tx), wl_fixed_from_double(g_ptr_y - ty));
    for (int i = 0; i < g_nptrs; i++) {
        struct wl_resource *ptr = g_ptrs[i].ptr;
        if (wl_resource_get_client(ptr) != client) continue;
        /* From version 8 the discrete event is replaced by value120 and must not be sent. */
        if (wl_resource_get_version(ptr) >= WL_POINTER_AXIS_VALUE120_SINCE_VERSION)
            wl_pointer_send_axis_value120(ptr, WL_POINTER_AXIS_VERTICAL_SCROLL, steps * 120);
        else if (wl_resource_get_version(ptr) >= WL_POINTER_AXIS_DISCRETE_SINCE_VERSION)
            wl_pointer_send_axis_discrete(ptr, WL_POINTER_AXIS_VERTICAL_SCROLL, steps);
        wl_pointer_send_axis(ptr, t, WL_POINTER_AXIS_VERTICAL_SCROLL, wl_fixed_from_int(steps * 10));
        pointer_frame(ptr);
    }
    wl_display_flush_clients(g_display);
}

static void key_event(uint32_t evdev, int pressed) {
    /* Keys go to the program window the user last clicked (else the topmost non-shell window),
     * never to Wine's desktop surface: winewayland hands each key to the hwnd of the surface that
     * holds keyboard focus, and a key handed to explorer's desktop hwnd is queued to explorer's
     * thread, not to the foreground program (X11 delivers keys to the focused app window, so
     * this mirrors it). The desktop itself is the last resort. */
    struct surface *target = NULL;
    struct wl_resource *ime = wnc_ime_target();
    if (ime) target = wl_resource_get_user_data(ime);
    if (!target && g_key_target && g_key_target->mapped && g_key_target != g_desktop) target = g_key_target;
    if (!target) target = g_desktop;
    if (!target) {
        struct surface *s;
        wl_list_for_each_reverse(s, &g_toplevels, toplevel_link) { target = s; break; }
    }
    if (!target) return;
    struct wl_client *client = wl_resource_get_client(target->resource);
    if (!client_has_keyboard(client)) return;
    keyboard_focus(target->resource);
    int mods_changed = mods_key(evdev, pressed);
    for (int i = 0; i < g_nkbs; i++) {
        if (wl_resource_get_client(g_kbs[i].kb) != client) continue;
        wl_keyboard_send_key(g_kbs[i].kb, wl_display_next_serial(g_display), now_ms(), evdev,
                             pressed ? WL_KEYBOARD_KEY_STATE_PRESSED : WL_KEYBOARD_KEY_STATE_RELEASED);
        if (mods_changed)
            wl_keyboard_send_modifiers(g_kbs[i].kb, wl_display_next_serial(g_display), mods_depressed(), 0, g_mods_locked, 0);
    }
    wl_display_flush_clients(g_display);
}

/* The compositor lives as long as the process; a game session is what starts and ends.
 * WinNative returns to its launcher without killing the process, so the session that ends here
 * must leave nothing behind for the next one: every client is disconnected (their surfaces,
 * buffers, constraints and seat resources go through the ordinary destroy paths), the display
 * layer is hidden, the per-session counters and the dma-buf format table are reset (the next
 * session may run with another UBWC setting) and the session log is closed. The screen surface
 * is dropped by the app (nativeSetSurface(null) / nativeEndSession). Compositor thread. */
static void session_end(void) {
    while (g_clients) wl_client_destroy(g_clients->client);
    g_grab = NULL; g_key_target = NULL; g_ime_click = NULL;
    g_desktop = NULL; g_hud_surface = NULL; g_hdr_unimported = NULL;
    g_active_constraint = NULL;
    g_mod_keys_held = g_mods_locked = 0;
    sc_layer_hide();
    wnc_color_session_end();
    g_zero_copy_paused = 0;
    g_zero_copy_fx_skip_said = 0;
    g_stat_frames = g_stat_dmabuf = g_stat_shm = 0;
    g_scene_w = g_scene_h = 0;
    g_desktop_w = g_desktop_h = 0;
    g_dirty = 1;
    if (g_fmt_table_fd >= 0) { close(g_fmt_table_fd); g_fmt_table_fd = -1; }
    g_fmt_table_size = 0; g_fmt_table_n = 0;
    g_dmabuf_fmts_ready = 0;
    wnc_log("display", "session ended: clients disconnected, state reset");
    close_session_log();
}

/* A new session on the running compositor: a fresh session log. The app has already stored the
 * session's settings (hide shell, output mode, FPS limit, zero-copy...). Compositor thread. */
static void session_begin(void) {
    open_session_log();
    wnc_log("display", "session started on the running compositor (%s)", vkp_gpu_name());
}

/* wl event-loop callback: drain queued input events written by the Android UI thread. */
static int on_input_readable(int fd, uint32_t mask, void *data) {
    struct input_msg m;
    while (read(fd, &m, sizeof(m)) == (ssize_t)sizeof(m)) {
        switch (m.type) {
        case 1: deliver_key(&m); break;
        case 2: pointer_event(m.p1, m.p2, 0, 0); break;          /* scene motion */
        case 3: pointer_button(m.p1, m.p2); break;               /* button at pointer */
        case 4: scroll_event(m.p1); break;
        case 5: on_vsync(((int64_t)m.p1 << 32) | (uint32_t)m.p2); break;
        case 6: pointer_delta(m.p1 / 256.0, m.p2 / 256.0); break; /* relative motion, 1/256 px */
        case 7: session_end(); break;
        case 8: session_begin(); break;
        default: deliver_pointer(&m); break;
        }
    }
    return 0;
}

/* Called from JNI (Android UI thread). Queues a pointer event; the compositor thread
 * dispatches it. x/y are in INPUT_SPACE (0..1919, 0..1079) over the whole output. */
void wnc_wayland_send_pointer(int action, int x, int y) {
    if (g_input_pipe[1] < 0) return;
    struct input_msg m = { 0, action, x, y };
    ssize_t n = write(g_input_pipe[1], &m, sizeof(m));
    (void)n;
}

/* Called from JNI. Queues a key event. evdev = Linux input keycode (e.g. KEY_A=30); state 1=down 0=up. */
void wnc_wayland_send_key(int evdev, int state) {
    if (g_input_pipe[1] < 0) return;
    struct input_msg m = { 1, evdev, state, 0 };
    ssize_t n = write(g_input_pipe[1], &m, sizeof(m));
    (void)n;
}

/* Called from JNI when a game session ends / a new one starts on the running compositor. */
void wnc_wayland_end_session(void) {
    if (g_input_pipe[1] < 0) return;
    struct input_msg m = { 7, 0, 0, 0 };
    ssize_t n = write(g_input_pipe[1], &m, sizeof(m));
    (void)n;
}

void wnc_wayland_begin_session(void) {
    if (g_input_pipe[1] < 0) return;
    struct input_msg m = { 8, 0, 0, 0 };
    ssize_t n = write(g_input_pipe[1], &m, sizeof(m));
    (void)n;
}

/* Called from JNI on every screen refresh (Choreographer). */
void wnc_wayland_vsync(int64_t frame_time_ns) {
    if (g_input_pipe[1] < 0) return;
    struct input_msg m = { 5, (int)(frame_time_ns >> 32), (int)(uint32_t)frame_time_ns, 0 };
    ssize_t n = write(g_input_pipe[1], &m, sizeof(m));
    (void)n;
}

/* Called from JNI with the app's X-server input (on-screen controls, mouse): type 2 = motion
 * to scene x,y; 3 = evdev button a pressed/released (b); 4 = a wheel steps (negative = up);
 * 6 = relative motion by a,b in 1/256 pixel (the app's Relative Mouse / captured-mouse path). */
void wnc_wayland_send_scene_input(int type, int a, int b) {
    if (g_input_pipe[1] < 0) return;
    struct input_msg m = { type, a, b, 0 };
    ssize_t n = write(g_input_pipe[1], &m, sizeof(m));
    (void)n;
}

/* ------------------------------------------------------------------ extension module hooks
 * What wl_clipboard.c / wl_text_input.c need from the scene (see desktop_ext.h). */

const char *wnc_client_name(struct wl_client *client) { return client_name(client); }
struct wl_display *wnc_get_display(void) { return g_display; }
void wnc_request_redraw(void) { schedule_render(); }

/* The surface text input follows: the last clicked mapped program window, else the topmost
 * mapped window not owned by the desktop's process (explorer). NULL when there is none. */
struct wl_resource *wnc_ime_target(void) {
    struct surface *s;
    if (g_ime_click && g_ime_click->mapped) return g_ime_click->resource;
    struct wl_client *shell = g_desktop ? wl_resource_get_client(g_desktop->resource) : NULL;
    wl_list_for_each_reverse(s, &g_toplevels, toplevel_link)
        if (!shell || wl_resource_get_client(s->resource) != shell) return s->resource;
    return NULL;
}

void wnc_surface_scene_origin(struct wl_resource *surface, int *x, int *y) {
    struct surface *s = surface ? wl_resource_get_user_data(surface) : NULL;
    *x = 0; *y = 0;
    if (s && s->placed) { *x = s->x; *y = s->y; }
}

void wnc_inject_key(uint32_t evdev, int pressed) { key_event(evdev, pressed); }

/* ------------------------------------------------------------------ 10 s summary */

static struct wl_event_source *g_stats_timer;
/* The last window's zero-copy count, kept for the app (WaylandCompositor.nativeZeroCopyFrames). */
volatile unsigned g_zero_copy_last;

static int on_stats_timer(void *data) {
    int windows = 0;
    struct surface *s;
    wl_list_for_each(s, &g_toplevels, toplevel_link) windows++;
    unsigned zero_copy = ahb_swapchain_stats_take();
    unsigned layer_frames = sc_layer_frames_take();
    g_zero_copy_last = zero_copy;
    /* Interpolated frames the compositor added (framegen_bridge.c). They are on screen, so they
     * count there; they are NOT GPU frames from games and never inflate that number. */
    unsigned generated = vkp_framegen_stats_take();
    if (g_stat_frames || g_stat_dmabuf || g_stat_shm || generated) {
        char extra[160] = "";
        int off = 0;
        if (g_zero_copy || zero_copy) off += snprintf(extra + off, sizeof(extra) - (size_t)off, " | %u zero-copy frames", zero_copy);
        /* Frames the compositor put on a display layer through one of its OWN gralloc buffers (the
         * plain layer blit, or the effects chain's result): still hardware-composed, but not
         * zero-copy, so they are counted apart from the line above. */
        if (layer_frames) off += snprintf(extra + off, sizeof(extra) - (size_t)off, " | %u layer frames", layer_frames);
        if (generated) snprintf(extra + off, sizeof(extra) - (size_t)off, " | %u generated frames", generated);
        wnc_log("stats", "last 10 s: %u frames on screen (%.1f fps) | %u GPU frames from games | %u window redraws | %d windows open%s",
                   g_stat_frames + generated, (g_stat_frames + generated) / 10.0, g_stat_dmabuf, g_stat_shm, windows, extra);
    }
    /* The perf line, next to the stats line and only when something happened: the compositor thread's
     * time per scene and inside the driver (avg/max), what the base surface did, and how long games
     * waited for their buffers back. See the audit memory for how to read it. */
    {
        struct vkp_perf vp;
        vkp_perf_take(&vp);
        const unsigned drops = sc_layer_drops_take();
#define PERF_MS(sum, n) ((n) ? (double)(sum) / 1e6 / (double)(n) : 0.0)
        if (g_perf.scenes || vp.presents || g_perf.releases || drops)
            wnc_log("perf", "last 10 s: %u ticks, %u scenes, %u on screen (copy %u, zero-copy %u, layer copy %u) | "
                       "render_scene %.2f/%.2f ms | base %u black kept, %u presented | acquire %.2f/%.2f ms | "
                       "present %.2f/%.2f ms (%u) | fence wait %.2f/%.2f ms (%u, %u GPU release waits) | "
                       "release %.2f/%.2f ms (%u, %u held) | %u pool drops",
                       g_perf.ticks, g_perf.scenes, g_stat_frames, g_perf.copy_scenes, zero_copy, layer_frames,
                       PERF_MS(g_perf.scene_ns, g_perf.scenes), (double)g_perf.scene_max_ns / 1e6,
                       vp.base_kept, vp.base_presents,
                       PERF_MS(vp.acquire_ns, vp.acquires), (double)vp.acquire_max_ns / 1e6,
                       PERF_MS(vp.present_ns, vp.presents), (double)vp.present_max_ns / 1e6, vp.presents,
                       PERF_MS(vp.wait_ns, vp.waits), (double)vp.wait_max_ns / 1e6, vp.waits, vp.gpu_release_waits,
                       PERF_MS(g_perf.release_ns, g_perf.releases), (double)g_perf.release_max_ns / 1e6,
                       g_perf.releases, g_perf.releases_held, drops);
#undef PERF_MS
        memset(&g_perf, 0, sizeof(g_perf));
    }
    g_stat_frames = g_stat_dmabuf = g_stat_shm = 0;
    wnc_color_stats_tick(); /* HDR evidence for the same window (nothing when the HDR gate is closed) */
    /* A program that asked for dma-buf feedback (EGL's GPU path does, first thing) but has drawn
     * only wl_shm frames since is an OpenGL program whose EGL gave up on the GPU. The software
     * path it fell to has no rasteriser in the Wayland layers, so what it commits is black. */
    for (struct client_info *ci = g_clients; ci; ci = ci->next) {
        if (!ci->asked_feedback || ci->dmabuf_buffers || ci->shm_gl_said || ci->shm_frames < 150)
            continue;
        ci->shm_gl_said = 1;
        wnc_log("opengl", "%s asked for GPU buffers but has drawn only software (shared-memory) "
                   "frames: its OpenGL fell back to software rendering, which the Wayland layer cannot "
                   "draw - expect a black picture (main device %u:%u)", ci->name,
                   (unsigned)major(g_main_device), (unsigned)minor(g_main_device));
    }
    wl_event_source_timer_update(g_stats_timer, 10000);
    return 0;
}

/* ------------------------------------------------------------------ test input
 * $XDG_RUNTIME_DIR/test-input is a FIFO for scripted testing from a root shell, in scene
 * (desktop) coordinates: "move X Y", "click X Y", "rclick X Y", "dclick X Y", "down X Y",
 * "up X Y", "rdown X Y", "rup X Y", "rel DX DY" (relative motion, like the app's Relative
 * Mouse path), "key CODE" (evdev), "keydown CODE", "keyup CODE".
 * It lives in the app's private directory, so only the app and root can reach it. */

static char g_test_buf[512];
static size_t g_test_len;

static void test_input_line(const char *line) {
    double x, y;
    unsigned code;
    char cmd[16];
    int n = sscanf(line, "%15s %lf %lf", cmd, &x, &y);

    if (n < 2) return;
    wnc_log("test", "input: %s", line);
    if (!strcmp(cmd, "key") || !strcmp(cmd, "keydown") || !strcmp(cmd, "keyup")) {
        code = (unsigned)x;
        if (strcmp(cmd, "keyup")) key_event(code, 1);
        if (strcmp(cmd, "keydown")) key_event(code, 0);
        return;
    }
    if (n < 3) return;
    if (!strcmp(cmd, "move")) pointer_event(x, y, 0, 0);
    else if (!strcmp(cmd, "rel")) pointer_delta(x, y);
    else if (!strcmp(cmd, "down")) pointer_event(x, y, BTN_LEFT, 1);
    else if (!strcmp(cmd, "up")) pointer_event(x, y, BTN_LEFT, 0);
    else if (!strcmp(cmd, "rdown")) pointer_event(x, y, BTN_RIGHT, 1);
    else if (!strcmp(cmd, "rup")) pointer_event(x, y, BTN_RIGHT, 0);
    else if (!strcmp(cmd, "click") || !strcmp(cmd, "rclick") || !strcmp(cmd, "dclick")) {
        uint32_t button = cmd[0] == 'r' ? BTN_RIGHT : BTN_LEFT;
        int times = cmd[0] == 'd' ? 2 : 1;
        pointer_event(x, y, 0, 0);
        for (int i = 0; i < times; i++) {
            pointer_event(x, y, button, 1);
            pointer_event(x, y, button, 0);
        }
    }
}

static int on_test_input(int fd, uint32_t mask, void *data) {
    ssize_t r;
    while ((r = read(fd, g_test_buf + g_test_len, sizeof(g_test_buf) - 1 - g_test_len)) > 0) {
        char *start = g_test_buf, *nl;
        g_test_len += (size_t)r;
        g_test_buf[g_test_len] = 0;
        while ((nl = strchr(start, '\n'))) {
            *nl = 0;
            test_input_line(start);
            start = nl + 1;
        }
        g_test_len = strlen(start);
        memmove(g_test_buf, start, g_test_len);
        if (g_test_len >= sizeof(g_test_buf) - 1) g_test_len = 0; /* overlong line: drop */
    }
    return 0;
}

static void start_test_input(struct wl_event_loop *loop) {
    const char *rt = getenv("XDG_RUNTIME_DIR");
    char path[512];
    int fd;

    if (!rt || !*rt) return;
    snprintf(path, sizeof(path), "%s/test-input", rt);
    unlink(path);
    if (mkfifo(path, 0600) != 0) return;
    /* O_RDWR keeps a writer open ourselves, so a closing test shell never leaves us at EOF. */
    if ((fd = open(path, O_RDWR | O_NONBLOCK | O_CLOEXEC)) < 0) return;
    wl_event_loop_add_fd(loop, fd, WL_EVENT_READABLE, on_test_input, NULL);
}

/* ------------------------------------------------------------------ entry
 * Blocks in the wl event loop, so the JNI wrapper runs it on a dedicated thread.
 * XDG_RUNTIME_DIR must be set by the caller before this runs. */

static struct wl_listener g_client_created = { .notify = on_client_created };

int wnc_wayland_run(void) {
    wl_list_init(&g_surfaces);
    wl_list_init(&g_toplevels);
    wl_list_init(&g_constraints);
    wl_list_init(&g_relative_pointers);
    open_session_log();

    /* Bring the GPU up before any client can connect: loading Turnip the first time
     * can take many seconds, and it must not stall a client mid-handshake. */
    if (vkp_ready() != 0) WLOGE("renderer unavailable; clients will connect but nothing is drawn");

    struct wl_display *display = wl_display_create();
    if (!display) {
        WLOGE("wl_display_create failed");
        return 1;
    }

    /* Use a FIXED socket name so it always matches the guest's WAYLAND_DISPLAY=wayland-0
     * (wl_display_add_socket_auto would drift to wayland-1/2/… if a stale socket exists,
     * and the guest only ever looks for wayland-0). Unlink any stale socket+lock first so
     * a previous run that didn't clean up can't block the bind. */
    const char *rt = getenv("XDG_RUNTIME_DIR");
    if (rt && *rt) {
        char p[512];
        snprintf(p, sizeof(p), "%s/wayland-0", rt);      unlink(p);
        snprintf(p, sizeof(p), "%s/wayland-0.lock", rt); unlink(p);
    }
    if (wl_display_add_socket(display, "wayland-0") != 0) {
        WLOGE("add_socket(wayland-0) failed in XDG_RUNTIME_DIR=%s (errno path/perms?)",
              rt ? rt : "(null)");
        return 1;
    }
    wnc_log("display", "Wayland compositor listening on %s/wayland-0", rt ? rt : "?");
    wl_display_add_client_created_listener(display, &g_client_created);

    wl_global_create(display, &wl_compositor_interface, 6, NULL, bind_compositor);
    wl_global_create(display, &wl_subcompositor_interface, 1, NULL, bind_subcompositor);
    wl_global_create(display, &wp_viewporter_interface, 1, NULL, bind_viewporter);
    wl_display_init_shm(display); /* wl_shm global + pool/buffer handling */
    /* libdecor binds wl_output at 4; a lower version is a protocol error for the client. */
    wl_global_create(display, &wl_output_interface, 4, NULL, bind_output);
    wl_global_create(display, &xdg_wm_base_interface, 1, NULL, bind_xdg_wm_base);
    wl_global_create(display, &zwp_linux_dmabuf_v1_interface, 4, NULL, bind_dmabuf);
    /* gamescope's Wayland backend refuses a seat older than 8. */
    wl_global_create(display, &wl_seat_interface, 9, NULL, bind_seat);
    wl_global_create(display, &banner_desktop_v1_interface, 1, NULL, bind_desktop);
    wl_global_create(display, &wp_presentation_interface, 2, NULL, bind_presentation);
    wl_global_create(display, &zwp_pointer_constraints_v1_interface, 1, NULL, bind_pointer_constraints);
    wl_global_create(display, &zwp_relative_pointer_manager_v1_interface, 1, NULL, bind_relative_pointer_manager);
    wnc_ext_init(display); /* clipboard, text input, toplevel icons (own files, see desktop_ext.h) */
    ahb_swapchain_init(display); /* zero-copy layers: banner_ahb_v1, advertised whenever a display
                                  * layer is possible; the mode event carries the live switch
                                  * (ahb_swapchain.h) */
    wnc_color_init(display);  /* HDR10 (opt-in): decides the gate from the app's request + the
                                  * display + the layer path above; creates wp_color_manager_v1
                                  * only when it is open (color_mgmt.h). Before any client can
                                  * bind zwp_linux_dmabuf_v1, whose table it widens. */
    wl_list_init(&g_pending_releases);
    g_release_timer_fd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
    if (g_release_timer_fd >= 0)
        g_release_source = wl_event_loop_add_fd(wl_display_get_event_loop(display), g_release_timer_fd,
                                                WL_EVENT_READABLE, on_release_timer, NULL);

    /* Input injection: the Android UI thread writes events to g_input_pipe[1];
     * the wl event loop drains them on this thread. */
    g_display = display;
    if (pipe2(g_input_pipe, O_CLOEXEC | O_NONBLOCK) == 0) {
        wl_event_loop_add_fd(wl_display_get_event_loop(display), g_input_pipe[0],
                             WL_EVENT_READABLE, on_input_readable, NULL);
    } else {
        WLOGE("input pipe creation failed");
    }
    start_test_input(wl_display_get_event_loop(display));
    if ((g_stats_timer = wl_event_loop_add_timer(wl_display_get_event_loop(display), on_stats_timer, NULL)))
        wl_event_source_timer_update(g_stats_timer, 10000);

    wl_display_run(display); /* blocks, dispatches the event loop */

    wl_display_destroy(display);
    return 0;
}
