package com.winlator.cmod.shared.framegen

import android.opengl.GLSurfaceView
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Choreographer
import android.view.SurfaceHolder
import java.lang.ref.WeakReference
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.egl.EGLDisplay
import javax.microedition.khronos.egl.EGLSurface

object FrameGenGlSurface {
    private const val TAG = "WnFrameGen"

    fun attach(view: GLSurfaceView, paceRate: Float): Boolean {
        if (!FrameGen.requested) return false

        val redirected =
            runCatching {
                val threadField = GLSurfaceView::class.java.getDeclaredField("mGLThread")
                threadField.isAccessible = true
                val running = threadField.get(view)
                threadField.set(view, null)
                try {
                    view.setEGLWindowSurfaceFactory(Factory(view))
                } finally {
                    threadField.set(view, running)
                }
                true
            }.getOrElse {
                Log.w(TAG, "GL surface could not be redirected: ${it.message}")
                false
            }
        if (!redirected) return false

        view.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        Pacer.start(view, paceRate)
        return true
    }

    fun detach() = Pacer.stop()

    private object Pacer {
        private var worker: HandlerThread? = null
        private var handler: Handler? = null
        private var view: WeakReference<GLSurfaceView>? = null
        private var callback: Choreographer.FrameCallback? = null

        @Synchronized
        fun start(target: GLSurfaceView, rate: Float) {
            stop()
            val interval = if (rate > 1f) (1000000000.0 / rate).toLong() else 0L
            val thread = HandlerThread("wn-framegen-pace").apply { start() }
            val post = Handler(thread.looper)
            worker = thread
            handler = post
            view = WeakReference(target)
            post.post {
                val tick =
                    object : Choreographer.FrameCallback {
                        private var next = 0L
                        private var previous = 0L
                        private var period = interval

                        override fun doFrame(frameTimeNanos: Long) {
                            val current = synchronized(Pacer) { view?.get() } ?: return
                            if (previous > 0L) {
                                val delta = frameTimeNanos - previous
                                if (delta in 1..40000000L) period = delta
                            }
                            previous = frameTimeNanos
                            if (interval <= 0L) {
                                current.requestRender()
                            } else {
                                if (next == 0L) next = frameTimeNanos
                                if (frameTimeNanos + period / 2 >= next) {
                                    current.requestRender()
                                    next += interval
                                    if (next < frameTimeNanos) next = frameTimeNanos + interval
                                }
                            }
                            Choreographer.getInstance().postFrameCallback(this)
                        }
                    }
                synchronized(Pacer) { callback = tick }
                Choreographer.getInstance().postFrameCallback(tick)
            }
            Log.i(TAG, "frame generation paces the GL thread at ${Math.round(rate)}Hz")
        }

        @Synchronized
        fun stop() {
            val thread = worker ?: return
            val post = handler
            val tick = callback
            view = null
            callback = null
            handler = null
            worker = null
            post?.post { if (tick != null) Choreographer.getInstance().removeFrameCallback(tick) }
            thread.quitSafely()
        }
    }

    private class Factory(private val view: GLSurfaceView) : GLSurfaceView.EGLWindowSurfaceFactory {
        override fun createWindowSurface(
            egl: EGL10,
            display: EGLDisplay,
            config: EGLConfig,
            nativeWindow: Any?,
        ): EGLSurface? {
            val holder = nativeWindow as? SurfaceHolder
            var target: Any? = nativeWindow
            if (holder != null) {
                val frame = holder.surfaceFrame
                val width = if (frame.width() > 0) frame.width() else view.width
                val height = if (frame.height() > 0) frame.height() else view.height
                val producer = FrameGen.wrap(holder.surface, width, height)
                if (producer != null) target = producer
            }
            return runCatching { egl.eglCreateWindowSurface(display, config, target, null) }
                .getOrElse {
                    Log.w(TAG, "eglCreateWindowSurface failed: ${it.message}")
                    null
                }
        }

        override fun destroySurface(egl: EGL10, display: EGLDisplay, surface: EGLSurface) {
            egl.eglDestroySurface(display, surface)
        }
    }
}
