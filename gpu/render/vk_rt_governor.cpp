// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_rt_governor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>

#include "render/vk_gpu_busy.h"

namespace Vulkan::RtGovernor {
namespace {

constexpr u32 MaxLevel = 4;
constexpr float FrameMs = 1000.0f / 60.0f; // the game's frame cap
using Clock = std::chrono::steady_clock;

std::atomic<u32> level{0};

struct State {
    u32 last_window = 0;
    u32 over = 0;                 // windows in a row over the budget
    double room_since = -1.0;     // seconds: with room since
    double last_up = -1e9, last_change = -1e9;
    double up_wait = 4.0;
    bool was_enabled = false;
} state;

double Now() {
    static const auto start = Clock::now();
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void Set(u32 to, float utilisation, const char* why) {
    const u32 from = level.exchange(to);
    if (from != to) {
        std::printf("RT governor: level %u -> %u (GPU %.0f %% of the frame, %.1f ms; %s; up after %.0f s)\n",
                    from, to, utilisation * 100.0f, utilisation * FrameMs, why, state.up_wait);
    }
}

} // namespace

Caps ForLevel(u32 l) noexcept {
    Caps c;
    c.level = l;
    if (l >= 1) {
        c.probes = false;
        c.sun_half = true;
        c.fire_rays = 1;
    }
    if (l >= 2) {
        c.gi_rays = 1;
        c.rtao_rays = 2;
    }
    if (l >= 3) {
        c.sun_shadow = false;
        c.rtao_rays = 1;
    }
    if (l >= 4) {
        c.gi_rt = false;
    }
    return c;
}

Caps Current() noexcept {
    return ForLevel(level.load(std::memory_order_relaxed));
}

void Update(bool enabled, float budget_ms) {
    if (!enabled) {
        if (state.was_enabled) {
            Set(0, 0.0f, "off");
            state.was_enabled = false;
        }
        return;
    }
    state.was_enabled = true;
    const auto* busy = GpuBusy::Get();
    if (!busy || busy->Windows() == state.last_window) {
        return;
    }
    state.last_window = busy->Windows();
    const float u = busy->Utilisation();
    if (u < 0.0f) {
        return;
    }
    const double now = Now();
    static double last_print = 0.0;
    if (now - last_print >= 10.0) {
        last_print = now;
        std::printf("RT governor: level %u, GPU %.0f %% of the frame (%.1f ms; budget %.1f)\n",
                    level.load(std::memory_order_relaxed), u * 100.0f, u * FrameMs, budget_ms);
    }
    const float high = std::clamp(budget_ms, 8.0f, FrameMs) / FrameMs;
    const float low = std::clamp(budget_ms - 1.5f, 6.0f, FrameMs) / FrameMs;
    const u32 l = level.load(std::memory_order_relaxed);
    // 20 s held: the wait for a step up shrinks again.
    if (now - state.last_change > 20.0 && state.up_wait > 4.0) {
        state.up_wait = std::max(4.0, state.up_wait * 0.5);
        state.last_change = now;
    }
    if (u > high) {
        state.room_since = -1.0;
        if (++state.over >= 2 && l < MaxLevel) {
            if (now - state.last_up < 3.0) {
                state.up_wait = std::min(64.0, state.up_wait * 2.0); // the step up did not hold
            }
            state.over = 0;
            state.last_change = now;
            Set(l + 1, u, "over the budget");
        }
        return;
    }
    state.over = 0;
    if (u < low && l > 0) {
        if (state.room_since < 0.0) {
            state.room_since = now;
        } else if (now - state.room_since >= state.up_wait) {
            state.room_since = -1.0;
            state.last_up = now;
            state.last_change = now;
            Set(l - 1, u, "room");
        }
    } else {
        state.room_since = -1.0;
    }
}

} // namespace Vulkan::RtGovernor
