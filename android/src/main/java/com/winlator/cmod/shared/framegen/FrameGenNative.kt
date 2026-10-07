package com.winlator.cmod.shared.framegen

import android.content.Context
import android.view.Surface

object FrameGenNative {
    @Volatile private var loaded = false

    fun ensureLoaded(): Boolean {
        if (loaded) return true
        return synchronized(this) {
            if (!loaded) {
                runCatching { System.loadLibrary("c++_shared") }
                loaded = runCatching { System.loadLibrary("winlator") }.isSuccess
            }
            loaded
        }
    }

    @JvmStatic
    external fun nativeCreate(
        context: Context,
        output: Surface,
        width: Int,
        height: Int,
        cachePath: String,
        driverName: String?,
        multiplier: Int,
        targetRate: Int,
        flowScale: Int,
        refreshRate: Float,
        sourceRate: Float,
    ): Long

    @JvmStatic
    external fun nativeProducerSurface(handle: Long): Surface?

    @JvmStatic
    external fun nativeConfigure(
        handle: Long,
        multiplier: Int,
        targetRate: Int,
        flowScale: Int,
        refreshRate: Float,
        sourceRate: Float,
    )

    @JvmStatic
    external fun nativeRealFrames(handle: Long): Long

    @JvmStatic
    external fun nativeGeneratedFrames(handle: Long): Long

    @JvmStatic
    external fun nativeDestroy(handle: Long)
}
