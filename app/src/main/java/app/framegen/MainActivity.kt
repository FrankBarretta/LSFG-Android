package app.framegen

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import com.winlator.cmod.shared.framegen.FrameGen
import app.framegen.LsfgCache.preferFp16
import com.winlator.cmod.shared.framegen.FrameGenOptions

class MainActivity : Activity() {
    private lateinit var multiplierGroup: RadioGroup
    private lateinit var qualityGroup: RadioGroup
    private lateinit var engineGroup: RadioGroup
    private lateinit var dllStatus: TextView
    private lateinit var precisionGroup: RadioGroup

    // DIS quality = flow resolution (short side, px). FrameGenOptions.flowScale >= 1000 selects DIS
    // and encodes (value - 1000) as that short side (see fg_jni.c).
    private val qualities = listOf("เร็ว (180)" to 180, "สมดุล (270)" to 270, "คมชัด (360)" to 360)
    private val lsfgFlow = listOf("50%" to 50, "70%" to 70, "100%" to 100)
    private val multipliers = listOf(2, 3, 4)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 2, pad, pad)
        }
        fun label(text: String, bold: Boolean = false) = TextView(this).apply {
            this.text = text
            textSize = if (bold) 20f else 15f
            if (bold) setTypeface(null, Typeface.BOLD)
            setPadding(0, pad / 2, 0, pad / 4)
        }

        root.addView(label("FrameGen", bold = true))
        root.addView(label("เพิ่มเฟรมเรตด้วย DIS optical flow (Vulkan compute)"))

        root.addView(label("Engine"))
        engineGroup = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        engineGroup.addView(RadioButton(this).apply { id = 300; text = "DIS" })
        engineGroup.addView(RadioButton(this).apply { id = 301; text = "LSFG" })
        engineGroup.check(300)
        engineGroup.setOnCheckedChangeListener { _, _ -> refreshQualityLabels() }
        root.addView(engineGroup)
        dllStatus = label("")
        root.addView(dllStatus)
        root.addView(Button(this).apply {
            text = "เลือก Lossless.dll (สำหรับ LSFG)"
            setOnClickListener {
                startActivityForResult(
                    Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"),
                    REQ_DLL,
                )
            }
        })

        root.addView(label("ความละเอียด shader ของ LSFG"))
        precisionGroup = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        precisionGroup.addView(RadioButton(this).apply { id = 400; text = "FP16" })
        precisionGroup.addView(RadioButton(this).apply { id = 401; text = "FP32" })
        precisionGroup.check(if (preferFp16) 400 else 401)
        precisionGroup.setOnCheckedChangeListener { _, id ->
            val fp16 = id == 400
            preferFp16 = fp16
            if (LsfgCache.hasDll(this)) {
                Thread {
                    val error = LsfgCache.rebuild(this, fp16)
                    runOnUiThread {
                        if (error != null) Toast.makeText(this, error, Toast.LENGTH_LONG).show()
                        refreshDllStatus()
                    }
                }.start()
            }
        }
        root.addView(precisionGroup)

        root.addView(label("ตัวคูณเฟรม (LSFG สูงสุด 4x)"))
        multiplierGroup = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        multipliers.forEachIndexed { i, m ->
            multiplierGroup.addView(RadioButton(this).apply { id = 100 + i; text = "${m}x" })
        }
        multiplierGroup.check(100)
        root.addView(multiplierGroup)

        root.addView(label("คุณภาพ / flow scale"))
        qualityGroup = RadioGroup(this).apply { orientation = RadioGroup.VERTICAL }
        for (i in 0..2) qualityGroup.addView(RadioButton(this).apply { id = 200 + i })
        qualityGroup.check(201)
        root.addView(qualityGroup)
        refreshQualityLabels()
        refreshDllStatus()

        root.addView(label("วิธีใช้: กดเริ่ม → ในหน้าต่างแชร์หน้าจอของระบบ เลือก \"แอพเดียว\" แล้วเลือกเกม " +
            "(ห้ามเลือกทั้งหน้าจอ ไม่งั้น overlay จะถูกจับซ้ำ) จากนั้นสลับไปเล่นเกมตามปกติ " +
            "ต้องใช้ Android 14 ขึ้นไป"))

        root.addView(Button(this).apply {
            text = "เริ่ม"
            setOnClickListener { requestStart() }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(Button(this).apply {
            text = "หยุด"
            setOnClickListener {
                startService(Intent(this@MainActivity, CaptureService::class.java).setAction(CaptureService.ACTION_STOP))
            }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        setContentView(ScrollView(this).apply { addView(root) })
    }

    private val isLsfg get() = engineGroup.checkedRadioButtonId == 301

    private fun refreshQualityLabels() {
        val names = if (isLsfg) lsfgFlow.map { it.first } else qualities.map { it.first }
        for (i in 0..2) (qualityGroup.findViewById<RadioButton>(200 + i)).text = names[i]
    }

    private fun refreshDllStatus() {
        val variant = LsfgCache.variantName(this)
        dllStatus.text = if (variant != null) "LSFG cache: พร้อมใช้งาน ($variant)" else "LSFG cache: ยังไม่มี (เลือก Lossless.dll)"
    }

    private fun requestStart() {
        if (isLsfg && !LsfgCache.isReady(this)) {
            Toast.makeText(this, "ต้องเลือก Lossless.dll เพื่อสร้าง cache ก่อนใช้ LSFG", Toast.LENGTH_LONG).show()
            return
        }
        if (OverlayAccessibilityService.instance == null) {
            Toast.makeText(this, "เปิด FrameGen overlay ในการช่วยเหลือพิเศษ (Accessibility) ก่อน", Toast.LENGTH_LONG).show()
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            return
        }
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), REQ_NOTIF)
        }
        val manager = getSystemService(MediaProjectionManager::class.java)
        @Suppress("DEPRECATION")
        startActivityForResult(manager.createScreenCaptureIntent(), REQ_CAPTURE)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_DLL) {
            val uri = data?.data
            if (resultCode == RESULT_OK && uri != null) {
                Thread {
                    val error = LsfgCache.importDll(this, uri, preferFp16)
                    runOnUiThread {
                        Toast.makeText(this, error ?: "สร้าง LSFG cache สำเร็จ", Toast.LENGTH_LONG).show()
                        refreshDllStatus()
                    }
                }.start()
            }
            return
        }
        if (requestCode != REQ_CAPTURE) return
        if (resultCode != RESULT_OK || data == null) {
            Toast.makeText(this, "ยกเลิกการแชร์หน้าจอ", Toast.LENGTH_SHORT).show()
            return
        }

        val multiplier = multipliers[multiplierGroup.checkedRadioButtonId - 100]
        val qIndex = qualityGroup.checkedRadioButtonId - 200
        val options = if (isLsfg) {
            FrameGenOptions(
                enabled = true,
                multiplier = multiplier,
                targetRate = 0,
                flowScale = lsfgFlow[qIndex].second,           // < 1000 selects LSFG, value/100 = flow scale
                cachePath = LsfgCache.cacheFile(this).absolutePath,
                driverName = null,
                sourceRate = 60,
            )
        } else {
            FrameGenOptions(
                enabled = true,
                multiplier = multiplier,
                targetRate = 0,
                flowScale = 1000 + qualities[qIndex].second,   // DIS, flow short side in px
                cachePath = filesDir.absolutePath,             // required non-empty; DIS does not use it
                driverName = null,
                sourceRate = 60,
            )
        }
        FrameGen.install(this, options)
        // Picks the display mode that fits multiplier x source rate, stores the refresh rate in FrameGen.
        FrameGen.applyDisplayMode(this)
        val modeId = window.attributes.preferredDisplayModeId

        val service = Intent(this, CaptureService::class.java)
            .setAction(CaptureService.ACTION_START)
            .putExtra(CaptureService.EXTRA_RESULT_CODE, resultCode)
            .putExtra(CaptureService.EXTRA_RESULT_DATA, data)
            .putExtra(CaptureService.EXTRA_DISPLAY_MODE_ID, modeId)
        startForegroundService(service)
        moveTaskToBack(true)
    }

    companion object {
        private const val REQ_CAPTURE = 1
        private const val REQ_NOTIF = 2
        private const val REQ_DLL = 3
    }
}
