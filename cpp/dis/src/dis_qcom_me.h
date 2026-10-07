// SPDX-FileCopyrightText: Copyright 2026 qwertypower (DEVAR Entertainment LLC)
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Hardware motion estimation through GL_QCOM_motion_estimation (Adreno), used by DIS as a
// second starting point for its block search. Private to the DIS module.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct DisQcomMe DisQcomMe;

// Whether the platform's GLES driver exposes the extension, and its search block size.
// Probed once per process; cheap after the first call.
bool dis_qcom_me_supported(uint32_t* block_x, uint32_t* block_y);

// Input frames are width x height R8 luminance, both multiples of the block size; the field is
// (width / block_x) x (height / block_y) vectors. NULL when unsupported or on failure.
DisQcomMe* dis_qcom_me_create(uint32_t width, uint32_t height);
void dis_qcom_me_destroy(DisQcomMe* me);

// Hands the newest frame over. When an earlier frame is held, estimates the motion from it to
// this one into out_xy (field_w * field_h pairs, pixels of the input size, previous -> newest)
// and returns true. Either way the newest frame becomes the reference for the next call.
bool dis_qcom_me_push(DisQcomMe* me, const uint8_t* luma, float* out_xy);

// Drops the held frame, so the next push only primes the history.
void dis_qcom_me_invalidate(DisQcomMe* me);
