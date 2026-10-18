// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (experimental, ticket 59): the game's local lights as a per-frame world-space list, read
// from its light passes' buffers (rt_light_records.h for the layout). Wanted only by a consumer
// (the spot shadow, lights_dump); with none, the light draws are not even recognised.

#pragma once

#include <array>
#include <vector>
#include "common/types.h"
#include "render/rt_light_records.h"

namespace Vulkan {
class CameraMotion;

class RtLights {
public:
    using SceneLight = RtLightRecords::SceneLight;
    using SpotRecord = RtLightRecords::SpotRecord;

    explicit RtLights(CameraMotion& camera_motion) : camera_motion{camera_motion} {}

    /// A light pass bound `size` bytes of light records at `address` (the GPU thread, per tile
    /// draw; a buffer is read once a frame).
    void OnLightBuffer(VAddr address, u64 size, u64 ps_hash);
    /// The frame ended: its list becomes the last complete one; reports (per 600 frames, and the
    /// whole list when `dump` changed).
    void FinishFrame(u32 frame, float dump);
    [[nodiscard]] const std::vector<SceneLight>& FrameLights() const noexcept {
        return lights_last;
    }
    /// The last complete frame's shadowed spot (valid false: none).
    [[nodiscard]] const SpotRecord& FrameSpot() const noexcept {
        return spot_last;
    }

private:
    CameraMotion& camera_motion;
    std::vector<SceneLight> lights_building, lights_last;
    SpotRecord spot_building, spot_last;
    struct BufferKey {
        u64 size;
        std::array<u32, 16> head; // the first record
        bool operator==(const BufferKey&) const = default;
    };
    std::vector<BufferKey> buffers_seen; // this frame's (each read once)
    u32 frames = 0, buffers_total = 0;
    float dump_seen = 0.0f;
};

} // namespace Vulkan
