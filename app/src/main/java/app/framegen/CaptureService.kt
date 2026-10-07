package app.framegen

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Toast
import com.winlator.cmod.shared.framegen.FrameGen

/**
 * Screen (single app) -> MediaProjection -> FrameGen producer surface -> native DIS engine
 * -> output SurfaceView in a click-through overlay window.
 */
class CaptureService : Service() {
    private val handler = Handler(Looper.getMainLooper())
    private var projection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var windowManager: WindowManager? = null
    private var overlay: SurfaceView? = null
    private var pipelineUp = false
    private var stopping = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> shutdown()
            ACTION_START -> start(intent)
        }
        return START_NOT_STICKY
    }

    private fun start(intent: Intent) {
        if (projection != null) return
        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
        @Suppress("DEPRECATION")
        val data: Intent? =
            if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
            else intent.getParcelableExtra(EXTRA_RESULT_DATA)
        if (data == null) {
            stopSelf()
            return
        }

        startForegroundWithNotification()

        val manager = getSystemService(MediaProjectionManager::class.java)
        val mp = manager.getMediaProjection(resultCode, data)
        if (mp == null) {
            toast("เริ่มจับภาพหน้าจอไม่ได้")
            shutdown()
            return
        }
        // Android 14+: the callback must be registered before createVirtualDisplay.
        mp.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                handler.post { shutdown() }
            }
        }, handler)
        projection = mp

        addOverlay(intent.getIntExtra(EXTRA_DISPLAY_MODE_ID, 0))
    }

    private fun addOverlay(preferredModeId: Int) {
        val host = OverlayAccessibilityService.instance
        if (host == null) {
            toast("ยังไม่ได้เปิดบริการการช่วยเหลือพิเศษของ FrameGen")
            shutdown()
            return
        }
        val wm = host.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        windowManager = wm
        val bounds = wm.maximumWindowMetrics.bounds

        val view = SurfaceView(host)
        view.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) = Unit

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                if (width > 0 && height > 0) beginPipeline(holder, width, height)
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                endPipeline()
            }
        })

        val lp = WindowManager.LayoutParams(
            bounds.width(),
            bounds.height(),
            WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or // touches go to the app underneath
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
            PixelFormat.OPAQUE,
        )
        lp.gravity = Gravity.TOP or Gravity.START
        lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
        if (preferredModeId != 0) lp.preferredDisplayModeId = preferredModeId

        overlay = view
        wm.addView(view, lp)
    }

    private fun beginPipeline(holder: SurfaceHolder, width: Int, height: Int) {
        if (pipelineUp) return
        val mp = projection ?: return
        val output = holder.surface

        val producer = FrameGen.wrap(output, width, height)
        if (producer == null || producer === output) {
            // wrap() hands the output back untouched when the presenter could not start.
            Log.w(TAG, "frame generation presenter did not start")
            toast("เริ่ม Frame Gen ไม่ได้ (ดู logcat tag WnFrameGen / FrameGen)")
            handler.post { shutdown() }
            return
        }

        val dpi = resources.displayMetrics.densityDpi
        virtualDisplay = mp.createVirtualDisplay(
            "framegen-capture", width, height, dpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            producer, null, null,
        )
        pipelineUp = virtualDisplay != null
        if (!pipelineUp) {
            toast("สร้าง virtual display ไม่ได้")
            handler.post { shutdown() }
        }
    }

    private fun endPipeline() {
        virtualDisplay?.release()
        virtualDisplay = null
        FrameGen.release()
        pipelineUp = false
    }

    private fun shutdown() {
        if (stopping) return
        stopping = true
        endPipeline()
        overlay?.let { v -> runCatching { windowManager?.removeView(v) } }
        overlay = null
        projection?.stop()
        projection = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        if (!stopping) shutdown()
        super.onDestroy()
    }

    private fun startForegroundWithNotification() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, "FrameGen", NotificationManager.IMPORTANCE_LOW),
        )
        val stop = PendingIntent.getService(
            this, 0,
            Intent(this, CaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setContentTitle("FrameGen กำลังทำงาน")
            .setContentText("แตะเพื่อหยุด")
            .setContentIntent(stop)
            .addAction(Notification.Action.Builder(null, "หยุด", stop).build())
            .setOngoing(true)
            .build()
        startForeground(NOTIF_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
    }

    private fun toast(text: String) =
        handler.post { Toast.makeText(applicationContext, text, Toast.LENGTH_LONG).show() }

    companion object {
        private const val TAG = "FrameGenService"
        private const val CHANNEL = "framegen"
        private const val NOTIF_ID = 1
        const val ACTION_START = "app.framegen.START"
        const val ACTION_STOP = "app.framegen.STOP"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        const val EXTRA_DISPLAY_MODE_ID = "display_mode_id"
    }
}
