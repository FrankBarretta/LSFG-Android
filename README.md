# FrameGen (standalone)

แอพ Android ที่ห่อ engine frame generation จาก WinNative (DIS optical flow + Vulkan compute) ให้ใช้เดี่ยวๆ

Pipeline: MediaProjection (แอพเดียว) → producer Surface (AImageReader) → fg_present.c (Vulkan, DIS) → overlay SurfaceView

## Build
Android Studio (Ladybug+) → Open โปรเจกต์นี้ → ติดตั้ง NDK + CMake จาก SDK Manager → Run
ต้องมี `glslc` ใน NDK (`<ndk>/shader-tools/...`, NDK r26 ขึ้นไปมี) ใช้คอมไพล์ shader DIS ตอน build

## ที่เพิ่ม/เปลี่ยนจาก extraction
- `shared/vk_driver.h` — แทนของ WinNative (ใช้ libvulkan.so ของเครื่อง ไม่ใช้ Turnip)
- LSFG เปิดใช้แล้ว: ดึง SPIR-V ตรงจากรีซอร์สใน Lossless.dll (ตัวแปร FP16/FP32) ไม่ต้องใช้ DXBC translator (`lsfg_dxbc_stub.c`) เลือก DLL ในแอพ → สร้าง `lsfg.cache` ใน filesDir (เก็บสำเนา DLL ไว้ในแอพ เพื่อสลับ FP16/FP32 ได้ในหน้าตั้งค่า)
- `bin2c.cmake`, `CMakeLists.txt` — build ใหม่แบบสแตนด์อโลน
- Kotlin: `MainActivity`, `CaptureService` (ตัว FrameGen*.kt เดิมไม่แก้)

## Overlay / สัมผัสทะลุ
overlay ใช้ `TYPE_ACCESSIBILITY_OVERLAY` ผ่าน `OverlayAccessibilityService` (Android 12+ บล็อกสัมผัสที่ทะลุ overlay ปกติที่ทึบ) ต้องเปิดบริการนี้ใน Settings → Accessibility ก่อนใช้ (ไม่ต้องใช้สิทธิ์ "แสดงทับแอพอื่น" แล้ว)

## ข้อจำกัด
- ยังไม่ได้ compile/ทดสอบบนเครื่องจริง
- ต้องเลือก "แอพเดียว" ตอนแชร์หน้าจอ (Android 14+) ไม่งั้น overlay ถูกจับซ้ำ
- เปลี่ยนค่าระหว่างรันไม่ได้ ต้องกดหยุดแล้วเริ่มใหม่ (FrameGen.reconfigure() clamp flowScale จนหลุด DIS)

License: GPL (ตาม LICENSE ของ WinNative) — โค้ดทั้งโปรเจกต์ต้องเป็น GPL ด้วย
