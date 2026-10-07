/* Current Lossless.dll builds ship SPIR-V (FP16/FP32) directly, so no DXBC translator is built.
 * Only very old DLLs that ship DXBC only would reach this; they are reported as unsupported. */
#include "lsfg_dxbc.h"

bool lsfg_translate_dxbc(const uint8_t* bytecode, uint32_t size, uint32_t** out_words,
                         uint32_t* out_word_count) {
    (void)bytecode; (void)size; (void)out_words; (void)out_word_count;
    return false;
}
void lsfg_free_translated(uint32_t* words) { (void)words; }
