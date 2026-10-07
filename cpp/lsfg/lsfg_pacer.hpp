// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace lsfg {

constexpr size_t LSFG_MAX_MULTIPLIER = 4;

struct LsfgPacerConfig {
    uint32_t multiplier{2};
    uint32_t target_rate{};
    float refresh_rate{};
    float source_rate{};
};

struct LsfgPlan {
    size_t generations{};
    bool warm{};
};

struct LsfgPacerStats {
    float source_rate{};
    size_t cost_limit{};
    float loop_rate{};
    float refresh_rate{};
    float target_rate{};
    float slots{};
    size_t limit{};
    bool rates_settled{};
    uint64_t last_drawn{};
    float last_elapsed{};
    uint64_t source_frames{};
    int64_t now_ns{};
    float instant_interval{};
    uint32_t rate_jumps{};
    bool probing{};
    float rate_at_raise{};
    float output_credit{};
    float raise_delay{};
};

class LsfgPacer {
public:
    void SetConfig(const LsfgPacerConfig& config_) {
        config = config_;
    }

    [[nodiscard]] const LsfgPacerConfig& Config() const {
        return config;
    }

    [[nodiscard]] size_t MaxGenerations() const;

    [[nodiscard]] LsfgPlan Plan(size_t capacity, uint64_t source_frames);

    [[nodiscard]] LsfgPacerStats Stats() const;

    void Reset();

private:
    using Clock = std::chrono::steady_clock;

    struct SourceSample {
        Clock::time_point when;
        uint64_t frames;
    };
    static constexpr size_t SOURCE_WINDOW_CAPACITY = 128;

    void TrackSourceRate(Clock::time_point now, uint64_t source_frames);
    void TrackLoopRate(float interval_seconds);
    void PushSourceSample(Clock::time_point now, uint64_t source_frames);
    void ClearSourceWindow();
    void TrimSourceWindow(Clock::time_point now, float seconds);
    [[nodiscard]] const SourceSample& SourceSampleAt(size_t index) const;
    [[nodiscard]] float WindowInterval(Clock::time_point now, uint64_t source_frames,
                                       float seconds, float reference, size_t& samples,
                                       float& span) const;
    [[nodiscard]] bool RatesSettled() const;
    [[nodiscard]] float SourceInterval() const;
    [[nodiscard]] size_t HeadroomLimit() const;
    [[nodiscard]] size_t SlotLimit() const;
    void TrackCost(Clock::time_point now, size_t ceiling);

    LsfgPacerConfig config;

    std::optional<Clock::time_point> last_frame;
    std::optional<Clock::time_point> last_source_sample;
    uint64_t last_source_frames{};
    std::array<SourceSample, SOURCE_WINDOW_CAPACITY> source_window{};
    size_t source_window_head{};
    size_t source_window_count{};
    float source_interval{};
    float source_span{};
    float prior_interval{};
    float loop_interval{};
    uint32_t source_samples{};
    uint32_t loop_samples{};
    uint32_t rate_jumps{};
    uint64_t last_drawn{};
    float last_elapsed{};
    float output_credit{};
    size_t limit{};
    size_t cost_limit{};
    size_t probe_from{};
    float raise_delay{0.25f};
    float rate_at_raise{};
    float rate_before_probe{};
    bool probing{};
    std::optional<Clock::time_point> last_cost_change;
    std::optional<Clock::time_point> drop_since;
    int64_t last_now_ns{};
    float last_instant{};
};

}
