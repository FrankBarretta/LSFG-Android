package app.framegen

import android.content.Context
import android.net.Uri
import com.winlator.cmod.shared.framegen.FrameGenNative
import java.io.File

/** Builds the LSFG shader cache from the user's own Lossless.dll (SPIR-V read straight from its resources). */
object LsfgCache {
    @JvmStatic external fun nativeValidateDll(dllPath: String): Int
    @JvmStatic external fun nativeBuildCache(dllPath: String, cachePath: String, preferFp16: Boolean): Int
    @JvmStatic external fun nativeInspectCache(cachePath: String): Int
    @JvmStatic external fun nativeCacheVariant(cachePath: String): Int

    fun cacheFile(context: Context) = File(context.filesDir, "lsfg.cache")
    private fun dllFile(context: Context) = File(context.filesDir, "Lossless.dll")
    private fun prefs(context: Context) = context.getSharedPreferences("framegen", Context.MODE_PRIVATE)

    fun hasDll(context: Context) = dllFile(context).isFile

    var Context.preferFp16: Boolean
        get() = prefs(this).getBoolean("fp16", true)
        set(value) = prefs(this).edit().putBoolean("fp16", value).apply()

    fun isReady(context: Context): Boolean {
        val f = cacheFile(context)
        return f.isFile && FrameGenNative.ensureLoaded() && nativeInspectCache(f.absolutePath) == 0
    }

    /** "FP16" / "FP32" / null: precision the current cache actually holds. */
    fun variantName(context: Context): String? {
        if (!isReady(context)) return null
        return when (nativeCacheVariant(cacheFile(context).absolutePath)) {
            1 -> "FP16"
            2 -> "FP32"
            else -> null
        }
    }

    /** Stores the picked DLL in app storage (kept so precision can be switched) and builds the cache. */
    fun importDll(context: Context, uri: Uri, fp16: Boolean): String? {
        if (!FrameGenNative.ensureLoaded()) return "โหลด native library ไม่ได้"
        val dll = dllFile(context)
        val tmp = File(context.filesDir, "Lossless.dll.part")
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                tmp.outputStream().use { input.copyTo(it) }
            } ?: return "อ่านไฟล์ไม่ได้"
            val valid = nativeValidateDll(tmp.absolutePath)
            if (valid != 0) return "DLL ไม่ถูกต้อง (status $valid)"
            if (!tmp.renameTo(dll)) return "บันทึก DLL ไม่สำเร็จ"
        } finally {
            tmp.delete()
        }
        return rebuild(context, fp16)
    }

    /** Rebuilds the cache from the stored DLL. Returns an error text or null. */
    fun rebuild(context: Context, fp16: Boolean): String? {
        if (!FrameGenNative.ensureLoaded()) return "โหลด native library ไม่ได้"
        val dll = dllFile(context)
        if (!dll.isFile) return "ยังไม่ได้เลือก Lossless.dll"
        val st = nativeBuildCache(dll.absolutePath, cacheFile(context).absolutePath, fp16)
        return if (st == 0) null else "สร้าง cache ไม่สำเร็จ (status $st)"
    }
}
