package com.winlator.cmod.shared.framegen

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.SystemClock
import android.util.Log
import android.view.Display
import android.view.Surface
import android.view.WindowManager

object FrameGen {
    private const val TAG = "WnFrameGen"
    private const val CADENCE_EPSILON = 0.01f
    private const val SOURCE_WINDOW_NANOS = 500_000_000L
    private const val SOURCE_RATE_EPSILON = 1.5f

    private var appContext: Context? = null
    private var options = FrameGenOptions()
    private var handle = 0L
    private var producer: Surface? = null
    private var boundOutput: Surface? = null
    private var boundWidth = 0
    private var boundHeight = 0
    private var refreshRate = 0f
    private var generatingEnabled = true
    private var rebinder: Runnable? = null
    private var stateListener: Runnable? = null
    private var liveSourceRate = 0f
    private var sourceFrames = 0L
    private var sourceWindow = 0L

    val requested: Boolean
        @Synchronized get() = options.usable

    val active: Boolean
        @Synchronized get() = handle != 0L

    val generating: Boolean
        @Synchronized get() = generatingEnabled

    val multiplier: Int
        @Synchronized get() = options.multiplier

    val targetRate: Int
        @Synchronized get() = options.targetRate

    val flowScale: Int
        @Synchronized get() = options.flowScale

    @JvmStatic
    @Synchronized
    fun install(context: Context, source: FrameGenOptions) {
        appContext = context.applicationContext
        if (options != source) release()
        options = source
        generatingEnabled = true
        if (options.usable) FrameGenNative.ensureLoaded()
        notifyState()
    }

    @JvmStatic
    fun installFromIntent(context: Context, intent: Intent?) {
        install(context, FrameGenOptions.fromIntent(intent))
    }

    @JvmStatic
    @Synchronized
    fun wrap(output: Surface?, width: Int, height: Int): Surface? {
        if (output == null || !output.isValid) return output
        if (!options.usable || width <= 0 || height <= 0) return output
        if (handle != 0L && boundOutput === output && boundWidth == width && boundHeight == height) {
            return producer ?: output
        }

        releaseLocked()

        val context = appContext ?: return output
        if (!FrameGenNative.ensureLoaded()) {
            Log.w(TAG, "native library unavailable; frame generation stays off")
            return output
        }

        val created =
            runCatching {
                FrameGenNative.nativeCreate(
                    context,
                    output,
                    width,
                    height,
                    options.cachePath,
                    options.driverName,
                    options.multiplier,
                    options.targetRate,
                    options.flowScale,
                    refreshRate,
                    sourceRate(),
                )
            }.getOrElse {
                Log.w(TAG, "frame generation could not start: ${it.message}")
                0L
            }
        if (created == 0L) {
            Log.w(TAG, "frame generation presenter unavailable for ${width}x$height; frames pass through")
            return output
        }

        val surface = runCatching { FrameGenNative.nativeProducerSurface(created) }.getOrNull()
        if (surface == null || !surface.isValid) {
            Log.w(TAG, "frame generation producer surface invalid; frames pass through")
            FrameGenNative.nativeDestroy(created)
            return output
        }

        handle = created
        producer = surface
        boundOutput = output
        boundWidth = width
        boundHeight = height
        voteFrameRate()
        Log.i(TAG, "frame generation active ${width}x$height multiplier=${options.multiplier} " +
            "target=${options.targetRate} flow=${options.flowScale} refresh=$refreshRate")
        return surface
    }

    @JvmStatic
    @Synchronized
    fun noteSourceFrames(count: Int) {
        if (count <= 0 || handle == 0L) return
        val now = SystemClock.elapsedRealtimeNanos()
        if (sourceWindow == 0L) {
            sourceWindow = now
            sourceFrames = 0L
            return
        }
        sourceFrames += count.toLong()
        val elapsed = now - sourceWindow
        if (elapsed < SOURCE_WINDOW_NANOS) return
        val rate = sourceFrames * 1_000_000_000f / elapsed
        sourceWindow = now
        sourceFrames = 0L
        if (rate <= 1f || Math.abs(rate - liveSourceRate) < SOURCE_RATE_EPSILON) return
        liveSourceRate = rate
        pushConfig()
    }

    @JvmStatic
    @Synchronized
    fun setSurfaceRebinder(value: Runnable?) {
        rebinder = value
    }

    @JvmStatic
    @Synchronized
    fun setStateListener(value: Runnable?) {
        stateListener = value
    }

    private fun notifyState() {
        val target = synchronized(this) { stateListener } ?: return
        runCatching { target.run() }
    }

    @JvmStatic
    fun rebindSurface() {
        val target = synchronized(this) { rebinder } ?: return
        runCatching { target.run() }
            .onFailure { Log.w(TAG, "frame generation could not rebind the surface: ${it.message}") }
    }

    @JvmStatic
    @Synchronized
    fun setGenerating(on: Boolean) {
        generatingEnabled = on
        pushConfig()
        notifyState()
    }

    @JvmStatic
    @Synchronized
    fun reconfigure(multiplier: Int, targetRate: Int, flowScale: Int) {
        options = options.copy(
            multiplier = FrameGenOptions.clampMultiplier(multiplier),
            targetRate = targetRate.coerceAtLeast(0),
            flowScale = FrameGenOptions.clampFlowScale(flowScale),
        )
        pushConfig()
    }

    private fun wantedRate(): Int {
        val source = if (options.sourceRate > 0) options.sourceRate else 60
        return if (options.targetRate > 0) options.targetRate else options.multiplier * source
    }

    private fun voteFrameRate() {
        val output = boundOutput ?: return
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return
        val wanted = if (generatingEnabled) wantedRate().toFloat() else refreshRate
        runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                output.setFrameRate(
                    wanted,
                    Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE,
                    Surface.CHANGE_FRAME_RATE_ALWAYS,
                )
            } else {
                output.setFrameRate(wanted, Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE)
            }
        }
    }

    private fun sourceRate(): Float =
        if (liveSourceRate > 1f) liveSourceRate else options.sourceRate.toFloat()

    private fun pushConfig() {
        voteFrameRate()
        if (handle == 0L) return
        val multiplier = if (generatingEnabled) options.multiplier else 1
        val target = if (generatingEnabled) options.targetRate else 0
        val outcome =
            runCatching {
                FrameGenNative.nativeConfigure(
                    handle,
                    multiplier,
                    target,
                    options.flowScale,
                    refreshRate,
                    sourceRate(),
                )
            }
        Log.i(TAG, "frame generation reconfigured multiplier=$multiplier target=$target " +
            "flow=${options.flowScale} refresh=$refreshRate ok=${outcome.isSuccess}")
    }

    @JvmStatic
    @Synchronized
    fun realFrames(): Long =
        if (handle == 0L) 0L else runCatching { FrameGenNative.nativeRealFrames(handle) }.getOrDefault(0L)

    @JvmStatic
    @Synchronized
    fun generatedFrames(): Long =
        if (handle == 0L) 0L
        else runCatching { FrameGenNative.nativeGeneratedFrames(handle) }.getOrDefault(0L)

    @JvmStatic
    @Synchronized
    fun release() = releaseLocked()

    private fun releaseLocked() {
        if (handle != 0L) {
            runCatching { FrameGenNative.nativeDestroy(handle) }
            handle = 0L
        }
        producer?.let { runCatching { it.release() } }
        producer = null
        boundOutput = null
        liveSourceRate = 0f
        sourceFrames = 0L
        sourceWindow = 0L
        boundWidth = 0
        boundHeight = 0
    }

    @JvmStatic
    fun applyDisplayMode(activity: Activity?): Float {
        val window = activity?.window ?: return 0f
        val params = window.attributes
        if (!requested) {
            if (params.preferredDisplayModeId != 0) {
                params.preferredDisplayModeId = 0
                window.attributes = params
            }
            synchronized(this) { refreshRate = 0f }
            return 0f
        }
        if (!generating) return synchronized(this) { refreshRate }

        val display = displayOf(activity) ?: return 0f
        val active = display.mode
        val source = if (options.sourceRate > 0) options.sourceRate else 60
        val wanted = wantedRate()

        var best: Display.Mode? = null
        for (mode in display.supportedModes) {
            if (!sameSize(mode, active)) continue
            if (best == null || betterMode(mode, best!!, wanted, source)) best = mode
        }
        val selected = best ?: active
        if (selected.modeId != params.preferredDisplayModeId || params.preferredRefreshRate != 0f) {
            params.preferredDisplayModeId = selected.modeId
            params.preferredRefreshRate = 0f
            window.attributes = params
        }

        val rate = selected.refreshRate
        synchronized(this) {
            refreshRate = rate
            pushConfig()
        }
        Log.i(TAG, "frame generation display mode: wanted ${wanted}Hz, selected ${Math.round(rate)}Hz " +
            "from ${display.supportedModes.joinToString(" ") {
                "${it.physicalWidth}x${it.physicalHeight}@${Math.round(it.refreshRate)}"
            }} active ${active.physicalWidth}x${active.physicalHeight}@${Math.round(active.refreshRate)}")
        return rate
    }

    private fun sameSize(mode: Display.Mode, active: Display.Mode): Boolean =
        (mode.physicalWidth == active.physicalWidth && mode.physicalHeight == active.physicalHeight) ||
            (mode.physicalWidth == active.physicalHeight && mode.physicalHeight == active.physicalWidth)

    private fun betterMode(
        candidate: Display.Mode,
        current: Display.Mode,
        wanted: Int,
        sourceRate: Int,
    ): Boolean {
        val a = candidate.refreshRate
        val b = current.refreshRate
        val aMeets = a + 0.5f >= wanted
        val bMeets = b + 0.5f >= wanted
        if (aMeets != bMeets) return aMeets
        if (!aMeets) return a > b
        val aCadence = cadenceFits(a, sourceRate)
        val bCadence = cadenceFits(b, sourceRate)
        if (aCadence != bCadence) return aCadence
        return a < b
    }

    private fun cadenceFits(refreshRate: Float, sourceRate: Int): Boolean {
        if (refreshRate <= 0f || sourceRate <= 0 || refreshRate < sourceRate) return false
        val ratio = refreshRate / sourceRate
        val nearest = Math.round(ratio)
        return nearest >= 1 && Math.abs(ratio - nearest) <= CADENCE_EPSILON
    }

    private fun displayOf(activity: Activity): Display? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            activity.display?.let { return it }
        }
        val manager = activity.getSystemService(Context.WINDOW_SERVICE) as? WindowManager
        @Suppress("DEPRECATION")
        return manager?.defaultDisplay
    }
}
