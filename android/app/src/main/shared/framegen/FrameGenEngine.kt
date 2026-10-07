package com.winlator.cmod.shared.framegen

import com.winlator.cmod.R

enum class FrameGenEngine(val labelRes: Int) {
    OFF(R.string.session_drawer_frame_generation_engine_off),
    LSFG(R.string.session_drawer_frame_generation_engine_lsfg),
    DIS(R.string.session_drawer_frame_generation_engine_dis),
    ;

    companion object {
        @JvmStatic
        fun of(
            lsfgEnabled: Boolean,
            disEnabled: Boolean,
        ): FrameGenEngine =
            when {
                disEnabled -> DIS
                lsfgEnabled -> LSFG
                else -> OFF
            }
    }
}
