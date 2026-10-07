package com.winlator.cmod.shared.framegen

import android.content.Intent

data class FrameGenOptions(
    val enabled: Boolean = false,
    val multiplier: Int = DEFAULT_MULTIPLIER,
    val targetRate: Int = 0,
    val flowScale: Int = DEFAULT_FLOW_SCALE,
    val cachePath: String = "",
    val driverName: String? = null,
    val sourceRate: Int = 0,
) {
    val usable: Boolean get() = enabled && cachePath.isNotEmpty()

    fun writeTo(intent: Intent): Intent {
        intent.putExtra(EXTRA_ENABLED, enabled)
        intent.putExtra(EXTRA_MULTIPLIER, multiplier)
        intent.putExtra(EXTRA_TARGET_RATE, targetRate)
        intent.putExtra(EXTRA_FLOW_SCALE, flowScale)
        intent.putExtra(EXTRA_CACHE_PATH, cachePath)
        intent.putExtra(EXTRA_DRIVER, driverName)
        intent.putExtra(EXTRA_SOURCE_RATE, sourceRate)
        return intent
    }

    companion object {
        const val KEY_ENABLED = "frameGen"
        const val KEY_MULTIPLIER = "frameGenMultiplier"
        const val KEY_TARGET_RATE = "frameGenTargetRate"
        const val KEY_FLOW_SCALE = "frameGenFlowScale"

        const val EXTRA_ENABLED = "wn_framegen"
        const val EXTRA_MULTIPLIER = "wn_framegen_multiplier"
        const val EXTRA_TARGET_RATE = "wn_framegen_target_rate"
        const val EXTRA_FLOW_SCALE = "wn_framegen_flow_scale"
        const val EXTRA_CACHE_PATH = "wn_framegen_cache"
        const val EXTRA_DRIVER = "wn_framegen_driver"
        const val EXTRA_SOURCE_RATE = "wn_framegen_source_rate"

        const val DEFAULT_MULTIPLIER = 2
        const val DEFAULT_FLOW_SCALE = 70
        const val MIN_MULTIPLIER = 2
        const val MAX_MULTIPLIER = 4
        const val MIN_FLOW_SCALE = 25
        const val MAX_FLOW_SCALE = 100

        val MULTIPLIER_OPTIONS = listOf(2, 3, 4)
        val TARGET_OPTIONS = listOf(0, 60, 90, 120, 144, 165, 185)

        fun clampMultiplier(value: Int): Int = value.coerceIn(MIN_MULTIPLIER, MAX_MULTIPLIER)

        fun clampFlowScale(value: Int): Int = value.coerceIn(MIN_FLOW_SCALE, MAX_FLOW_SCALE)

        fun fromIntent(intent: Intent?): FrameGenOptions {
            if (intent == null || !intent.getBooleanExtra(EXTRA_ENABLED, false)) {
                return FrameGenOptions()
            }
            return FrameGenOptions(
                enabled = true,
                multiplier = clampMultiplier(intent.getIntExtra(EXTRA_MULTIPLIER, DEFAULT_MULTIPLIER)),
                targetRate = intent.getIntExtra(EXTRA_TARGET_RATE, 0).coerceAtLeast(0),
                flowScale = clampFlowScale(intent.getIntExtra(EXTRA_FLOW_SCALE, DEFAULT_FLOW_SCALE)),
                cachePath = intent.getStringExtra(EXTRA_CACHE_PATH).orEmpty(),
                driverName = intent.getStringExtra(EXTRA_DRIVER),
                sourceRate = intent.getIntExtra(EXTRA_SOURCE_RATE, 0).coerceAtLeast(0),
            )
        }
    }
}
