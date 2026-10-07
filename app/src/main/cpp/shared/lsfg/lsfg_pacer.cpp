// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "lsfg_pacer.hpp"

#include <algorithm>
#include <cmath>

namespace lsfg {

namespace {

using Clock = std::chrono::steady_clock;

constexpr float INTERVAL_SMOOTHING = 0.25f;
constexpr float SOURCE_STALE_SECONDS = 0.5f;
constexpr float SOURCE_WINDOW_SECONDS = 1.0f;
constexpr float SOURCE_PROBE_SECONDS = 0.25f;
constexpr float SOURCE_SETTLE_SECONDS = 0.4f;
constexpr uint64_t SOURCE_SETTLE_FRAMES = 4;
constexpr size_t SOURCE_PROBE_SAMPLES = 6;
constexpr float DISCONTINUITY_SECONDS = 0.25f;
constexpr float HEADROOM_EPSILON = 0.02f;
constexpr float CREDIT_EPSILON = 1.0e-4f;
constexpr uint32_t MIN_RATE_SAMPLES = 12;
constexpr float COST_RAISE_SECONDS = 0.25f;
constexpr float COST_PROBE_SECONDS = 0.5f;
constexpr float COST_BLAME_SECONDS = 1.25f;
constexpr float BASELINE_SMOOTHING = 0.25f;
constexpr float RATE_JUMP_HIGH = 1.33f;
constexpr float RATE_JUMP_LOW = 0.75f;
constexpr float COST_RECOVER_RATIO = 0.97f;
constexpr float COST_HOLD_SECONDS = 2.0f;
constexpr float COST_DROP_SECONDS = 0.5f;
constexpr float HITCH_CLIP_RATIO = 2.5f;
constexpr float OUTPUT_SNAP_EPSILON = 0.08f;
constexpr float SLOT_SNAP_EPSILON = 0.2f;
constexpr float COST_DROP_RATIO = 0.9f;

float Seconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<float>(to - from).count();
}

float SnapWhole(float value, float epsilon) {
    const float whole = std::round(value);
    return std::fabs(value - whole) <= epsilon ? whole : value;
}

}

size_t LsfgPacer::MaxGenerations() const {
    if (config.multiplier < 2) return 0;
    if (config.target_rate != 0) return LSFG_MAX_MULTIPLIER - 1;
    return std::min<size_t>(config.multiplier, LSFG_MAX_MULTIPLIER) - 1;
}

const LsfgPacer::SourceSample& LsfgPacer::SourceSampleAt(size_t index) const {
    const size_t oldest = (source_window_head + SOURCE_WINDOW_CAPACITY - source_window_count) %
                          SOURCE_WINDOW_CAPACITY;
    return source_window[(oldest + index) % SOURCE_WINDOW_CAPACITY];
}

void LsfgPacer::PushSourceSample(Clock::time_point now, uint64_t source_frames) {
    source_window[source_window_head] = SourceSample{now, source_frames};
    source_window_head = (source_window_head + 1) % SOURCE_WINDOW_CAPACITY;
    if (source_window_count < SOURCE_WINDOW_CAPACITY) ++source_window_count;
}

void LsfgPacer::ClearSourceWindow() {
    source_window_head = 0;
    source_window_count = 0;
    if (source_span >= SOURCE_SETTLE_SECONDS) prior_interval = source_interval;
    source_interval = 0.0f;
    source_span = 0.0f;
    source_samples = 0;
}

void LsfgPacer::TrimSourceWindow(Clock::time_point now, float seconds) {
    size_t keep = 0;
    for (size_t i = 0; i < source_window_count; ++i) {
        if (Seconds(SourceSampleAt(i).when, now) <= seconds) {
            keep = source_window_count - i;
            break;
        }
    }
    source_window_count = keep;
}

float LsfgPacer::WindowInterval(Clock::time_point now, uint64_t source_frames, float seconds,
                                float reference, size_t& samples, float& span) const {
    samples = 0;
    span = 0.0f;
    if (source_window_count == 0) return 0.0f;

    size_t anchor = 0;
    for (size_t i = 0; i < source_window_count; ++i) {
        if (Seconds(SourceSampleAt(i).when, now) <= seconds) break;
        anchor = i;
    }
    const SourceSample& start = SourceSampleAt(anchor);
    samples = source_window_count - anchor;
    const float clip = reference > 0.0f ? reference * HITCH_CLIP_RATIO : 0.0f;
    Clock::time_point previous = start.when;
    for (size_t i = anchor + 1; i < source_window_count; ++i) {
        const Clock::time_point when = SourceSampleAt(i).when;
        const float gap = Seconds(previous, when);
        span += clip > 0.0f ? std::min(gap, clip) : gap;
        previous = when;
    }
    const float tail = Seconds(previous, now);
    span += clip > 0.0f ? std::min(tail, clip) : tail;
    const uint64_t frames = source_frames > start.frames ? source_frames - start.frames : 0;
    if (frames == 0 || span <= 0.0f) return 0.0f;
    return span / static_cast<float>(frames);
}

void LsfgPacer::TrackSourceRate(Clock::time_point now, uint64_t source_frames) {
    if (!last_source_sample) {
        last_source_sample = now;
        last_source_frames = source_frames;
        PushSourceSample(now, source_frames);
        return;
    }

    const float elapsed = Seconds(*last_source_sample, now);
    if (elapsed <= 0.0f) {
        return;
    }

    last_source_sample = now;
    const uint64_t drawn =
        source_frames > last_source_frames ? source_frames - last_source_frames : 0;
    last_source_frames = source_frames;

    if (elapsed > SOURCE_STALE_SECONDS) {
        ClearSourceWindow();
        PushSourceSample(now, source_frames);
        return;
    }

    last_drawn = drawn;
    last_elapsed = elapsed;
    last_instant = drawn > 0 ? elapsed / static_cast<float>(drawn) : 0.0f;
    PushSourceSample(now, source_frames);

    size_t long_samples = 0;
    float long_span = 0.0f;
    float long_interval = WindowInterval(now, source_frames, SOURCE_WINDOW_SECONDS,
                                         source_interval, long_samples, long_span);

    size_t probe_samples = 0;
    float probe_span = 0.0f;
    const float probe_interval = WindowInterval(now, source_frames, SOURCE_PROBE_SECONDS,
                                                source_interval, probe_samples, probe_span);

    if (long_interval > 0.0f && probe_interval > 0.0f && probe_samples >= SOURCE_PROBE_SAMPLES &&
        long_samples > probe_samples &&
        (probe_interval > long_interval * RATE_JUMP_HIGH ||
         probe_interval < long_interval * RATE_JUMP_LOW)) {
        TrimSourceWindow(now, SOURCE_PROBE_SECONDS);
        ++rate_jumps;
        prior_interval = 0.0f;
        long_interval = probe_interval;
        long_samples = probe_samples;
        long_span = probe_span;
    }

    if (prior_interval > 0.0f) {
        if (long_span >= SOURCE_WINDOW_SECONDS) {
            prior_interval = 0.0f;
        } else if (long_interval > 0.0f) {
            const float weight = long_span / SOURCE_WINDOW_SECONDS;
            long_interval = prior_interval * (1.0f - weight) + long_interval * weight;
        }
    }

    source_interval = long_interval;
    source_span = long_span;
    source_samples = static_cast<uint32_t>(long_samples);
}

void LsfgPacer::TrackLoopRate(float interval_seconds) {
    loop_interval = loop_interval > 0.0f
                        ? loop_interval + (interval_seconds - loop_interval) * INTERVAL_SMOOTHING
                        : interval_seconds;
    if (loop_samples < MIN_RATE_SAMPLES) ++loop_samples;
}

bool LsfgPacer::RatesSettled() const {
    if (loop_samples < MIN_RATE_SAMPLES) return false;
    if (source_interval <= 0.0f || source_span < SOURCE_SETTLE_SECONDS) return false;
    return source_span / source_interval >= static_cast<float>(SOURCE_SETTLE_FRAMES);
}

float LsfgPacer::SourceInterval() const {
    if (RatesSettled()) return source_interval;
    return config.source_rate > 0.0f ? 1.0f / config.source_rate : 0.0f;
}

size_t LsfgPacer::HeadroomLimit() const {
    const float interval = SourceInterval();
    if (config.refresh_rate <= 0.0f || interval <= 0.0f) {
        return LSFG_MAX_MULTIPLIER - 1;
    }

    const float fit = SnapWhole(config.refresh_rate * interval, SLOT_SNAP_EPSILON);
    const float budget = std::ceil(fit - HEADROOM_EPSILON);
    return budget < 2.0f ? 0 : static_cast<size_t>(budget) - 1;
}

size_t LsfgPacer::SlotLimit() const {
    const float interval = SourceInterval();
    if (config.refresh_rate <= 0.0f || interval <= 0.0f) {
        return LSFG_MAX_MULTIPLIER - 1;
    }

    const float fit = SnapWhole(config.refresh_rate * interval, SLOT_SNAP_EPSILON);
    const float slots = std::floor(fit + HEADROOM_EPSILON);
    return slots < 2.0f ? 0 : static_cast<size_t>(slots) - 1;
}

void LsfgPacer::TrackCost(Clock::time_point now, size_t ceiling) {
    const float interval = SourceInterval();
    if (interval <= 0.0f || !RatesSettled()) return;
    const float rate = 1.0f / interval;

    if (!last_cost_change) {
        last_cost_change = now;
        rate_at_raise = rate;
        return;
    }

    const float since = Seconds(*last_cost_change, now);

    if (probing) {
        if (since < COST_PROBE_SECONDS) return;
        if (rate >= rate_before_probe * COST_RECOVER_RATIO) {
            raise_delay = COST_HOLD_SECONDS;
        } else {
            cost_limit = probe_from;
        }
        probing = false;
        drop_since.reset();
        rate_at_raise = rate;
        last_cost_change = now;
        return;
    }

    if (rate_at_raise > 0.0f && rate < rate_at_raise * COST_DROP_RATIO) {
        if (!drop_since) {
            drop_since = now;
            return;
        }
        if (Seconds(*drop_since, now) < COST_DROP_SECONDS) return;
        drop_since.reset();
        if (cost_limit > 0 && since <= COST_BLAME_SECONDS + COST_DROP_SECONDS) {
            probe_from = cost_limit;
            rate_before_probe = rate_at_raise;
            cost_limit--;
            probing = true;
        } else {
            raise_delay = COST_RAISE_SECONDS;
        }
        rate_at_raise = rate;
        last_cost_change = now;
        return;
    }

    drop_since.reset();
    rate_at_raise += (rate - rate_at_raise) * BASELINE_SMOOTHING;
    if (since < raise_delay || cost_limit >= ceiling || limit < cost_limit) return;

    float target = static_cast<float>(config.target_rate);
    if (target > 0.0f && config.refresh_rate > 0.0f) {
        target = std::min(target, config.refresh_rate);
    }
    const float wanted = target > 0.0f ? target * interval
                                       : static_cast<float>(MaxGenerations()) + 1.0f;
    if (wanted <= static_cast<float>(cost_limit) + 1.0f + HEADROOM_EPSILON) return;

    cost_limit++;
    raise_delay = COST_RAISE_SECONDS;
    rate_at_raise = rate;
    last_cost_change = now;
}

LsfgPlan LsfgPacer::Plan(size_t capacity, uint64_t source_frames) {
    const size_t ceiling = std::min(capacity, MaxGenerations());
    if (ceiling == 0) {
        Reset();
        return {};
    }

    const Clock::time_point now = Clock::now();
    last_now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    TrackSourceRate(now, source_frames);
    if (!last_frame) {
        last_frame = now;
        return {};
    }

    const float interval_seconds = Seconds(*last_frame, now);
    last_frame = now;

    if (interval_seconds <= 0.0f || interval_seconds > DISCONTINUITY_SECONDS) {
        output_credit = 0.0f;
        return LsfgPlan{0, true};
    }

    TrackLoopRate(interval_seconds);

    float target_rate = static_cast<float>(config.target_rate);
    if (target_rate > 0.0f && config.refresh_rate > 0.0f) {
        target_rate = std::min(target_rate, config.refresh_rate);
    }

    if (target_rate == 0.0f) {
        output_credit = 0.0f;
        TrackCost(now, ceiling);
        limit = std::min(std::min(ceiling, SlotLimit()), cost_limit);
        return LsfgPlan{limit, true};
    }

    TrackCost(now, ceiling);
    const size_t allowed = std::min(std::min(ceiling, HeadroomLimit()), cost_limit);
    const float pace_interval = SourceInterval() > 0.0f ? SourceInterval() : loop_interval;
    const float desired_outputs = SnapWhole(pace_interval * target_rate, OUTPUT_SNAP_EPSILON);
    if (allowed == 0 || desired_outputs <= 1.0f) {
        output_credit = 0.0f;
        limit = 0;
        return {};
    }

    output_credit += desired_outputs;
    const size_t outputs =
        std::max<size_t>(1, static_cast<size_t>(std::floor(output_credit + CREDIT_EPSILON)));
    const size_t generations = std::min(outputs - 1, allowed);

    output_credit -= static_cast<float>(generations + 1);
    if (output_credit < 0.0f) {
        output_credit = 0.0f;
    } else if (generations == allowed && output_credit >= 1.0f) {
        output_credit = std::fmod(output_credit, 1.0f);
    }

    limit = generations;
    return LsfgPlan{generations, true};
}

LsfgPacerStats LsfgPacer::Stats() const {
    LsfgPacerStats stats;
    stats.source_rate = source_interval > 0.0f ? 1.0f / source_interval : 0.0f;
    stats.loop_rate = loop_interval > 0.0f ? 1.0f / loop_interval : 0.0f;
    stats.refresh_rate = config.refresh_rate;
    stats.target_rate = static_cast<float>(config.target_rate);
    stats.slots = config.refresh_rate * source_interval;
    stats.limit = limit;
    stats.cost_limit = cost_limit;
    stats.rates_settled = RatesSettled();
    stats.last_drawn = last_drawn;
    stats.last_elapsed = last_elapsed;
    stats.source_frames = last_source_frames;
    stats.now_ns = last_now_ns;
    stats.instant_interval = last_instant;
    stats.rate_jumps = rate_jumps;
    stats.probing = probing;
    stats.rate_at_raise = rate_at_raise;
    stats.output_credit = output_credit;
    stats.raise_delay = raise_delay;
    return stats;
}

void LsfgPacer::Reset() {
    last_frame.reset();
    last_source_sample.reset();
    last_source_frames = 0;
    ClearSourceWindow();
    prior_interval = 0.0f;
    loop_interval = 0.0f;
    loop_samples = 0;
    last_drawn = 0;
    last_elapsed = 0.0f;
    rate_jumps = 0;
    output_credit = 0.0f;
    limit = 0;
    cost_limit = 0;
    probe_from = 0;
    probing = false;
    raise_delay = COST_RAISE_SECONDS;
    rate_at_raise = 0.0f;
    rate_before_probe = 0.0f;
    last_cost_change.reset();
    drop_since.reset();
    last_now_ns = 0;
    last_instant = 0.0f;
}

}
