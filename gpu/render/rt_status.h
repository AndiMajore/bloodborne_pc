// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: one line that says which ray-traced effects actually ran (ticket 89). Each effect feeds
// these counters where it counts its own passes; RayTracedAO::OnFrameStart prints
// "RT status: ..." every 600 frames in the world (not in menus or while loading), naming only the
// effects that were asked for, each as running with its count or OFF with the reason. The
// launcher shows the last such line.
#pragma once

#include <atomic>
#include <cstdint>

namespace Vulkan::RtStatus {

struct Counters {
    // Asked for this run (set once, at the effect's construction or settings read).
    std::atomic<bool> reflect_wanted{false}, probes_wanted{false};
    // Why an effect asked for is off for the whole run (a static string), or null.
    std::atomic<const char*> rtao_off{nullptr}, reflect_off{nullptr}, probes_off{nullptr};
    // This window's counts (reset with each line).
    std::atomic<std::uint32_t> rtao_passes{0};
    std::atomic<std::uint32_t> gather_rt{0}, gather_march{0}; // the SSGI's ray-traced gather, asked for
    std::atomic<std::uint32_t> sun_wanted{0}, sun_traced{0};  // frames the moon's shadow was on / traced
    std::atomic<std::uint32_t> reflect_passes{0}, probe_passes{0};
};

inline Counters& Get() {
    static Counters counters;
    return counters;
}

} // namespace Vulkan::RtStatus
