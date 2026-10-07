package app.framegen

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent

/**
 * Exists only to host the overlay as TYPE_ACCESSIBILITY_OVERLAY. Android 12+ blocks touches that
 * pass through an untrusted opaque overlay; accessibility overlays are trusted, so touches reach the game.
 */
class OverlayAccessibilityService : AccessibilityService() {
    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() = Unit

    override fun onUnbind(intent: android.content.Intent?): Boolean {
        if (instance === this) instance = null
        return super.onUnbind(intent)
    }

    companion object {
        @Volatile
        var instance: OverlayAccessibilityService? = null
    }
}
