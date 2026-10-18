// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the frame-time governor (ticket 71; governor=1 in rtlighting.txt). It reads the GPU's
// utilisation of the whole frame (GpuBusy: the game's passes swing 9-12.7 ms across Central
// Yharnam while the RT passes stay flat, audit section 7) and steps the RT effects down when the
// frame nears the budget, back up after some seconds with room. The effects ask Current() for
// their caps; the files' settings stay what they are (the caps only lower them).
//
//   level 0  the settings as they are
//   level 1  probes off, the sun's rays at half resolution, 1 fire-light ray
//   level 2  + the RT bounce 1 ray, the RTAO at most 2 rays
//   level 3  + the RT sun shadow off, the RTAO 1 ray
//   level 4  + the bounce by the screen march (no rays; no fire lights)
//
// Down one level when two half-second windows in a row are over the budget (1 s apart at most);
// up one after `up_wait` with room (budget - 1.5 ms); a step down within 3 s of a step up doubles
// up_wait (4 s .. 64 s) so the level does not swing at an edge; 20 s held halves it again.

#pragma once

#include "common/types.h"

namespace Vulkan::RtGovernor {

struct Caps {
    u32 level = 0;
    bool probes = true;
    bool sun_half = false;   // true: forced
    bool sun_shadow = true;  // false: forced off
    u32 rtao_rays = 32;
    u32 gi_rays = 8;
    bool gi_rt = true;       // false: the march
    u32 fire_rays = 8;
};

/// The caps of the current level (any thread).
Caps Current() noexcept;

/// Once a frame: reads the utilisation, moves the level. Off: level 0. budget_ms of 16.67.
void Update(bool enabled, float budget_ms);

/// The caps of a level (0..4), for the table above and its test.
Caps ForLevel(u32 level) noexcept;

} // namespace Vulkan::RtGovernor
