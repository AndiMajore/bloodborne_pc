// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_rt_lights.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "upscale/vk_camera_motion.h"

namespace Vulkan {

void RtLights::OnLightBuffer(VAddr address, u64 size, u64 ps_hash) {
    if (!RtLightRecords::IsRecordBuffer(size)) {
        return;
    }
    // Every tile draw binds its own copy of its variant's buffer (~960 a frame): a copy is known by
    // its size and first record.
    BufferKey key{.size = size};
    std::memcpy(key.head.data(), reinterpret_cast<const void*>(address), sizeof(key.head));
    if (std::ranges::find(buffers_seen, key) != buffers_seen.end()) {
        return;
    }
    buffers_seen.push_back(key);
    RtLightRecords::Decode({reinterpret_cast<const float*>(address), size_t(size / sizeof(float))},
                           camera_motion.View(), camera_motion.InvView(), ps_hash, lights_building,
                           &spot_building);
}

void RtLights::FinishFrame(u32 frame, float dump) {
    buffers_total += u32(buffers_seen.size());
    lights_last.swap(lights_building);
    lights_building.clear();
    buffers_seen.clear();
    spot_last = spot_building;
    spot_building = {};

    const auto inv_view = camera_motion.InvView();
    const std::array<float, 3> cam{inv_view[3], inv_view[7], inv_view[11]};
    const auto distance = [&](const SceneLight& l) {
        const float dx = l.position[0] - cam[0], dy = l.position[1] - cam[1], dz = l.position[2] - cam[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    const auto print = [&](const SceneLight& l) {
        std::printf("  %s%s ps %08x at (%.2f %.2f %.2f), %.1f m away, radius %.1f m, colour (%.3g %.3g %.3g)",
                    l.spot ? "spot" : "point", l.shadowed ? " (shadowed)" : "", u32(l.ps_hash), l.position[0],
                    l.position[1], l.position[2], distance(l), l.radius, l.color[0], l.color[1], l.color[2]);
        if (l.spot) {
            std::printf(", direction (%.2f %.2f %.2f), cone %.0f deg", l.direction[0], l.direction[1],
                        l.direction[2], l.cone * 2.0f * 57.29578f);
        }
        std::printf("\n");
    };
    // lights_dump=<new value> in rtlighting.txt: the whole list of the frame just finished.
    if (dump != dump_seen) {
        dump_seen = dump;
        std::printf("Lights: frame %u, camera (%.2f %.2f %.2f), %zu lights:\n", frame, cam[0], cam[1], cam[2],
                    lights_last.size());
        for (const auto& l : lights_last) {
            print(l);
        }
    }
    if (++frames == 600) {
        u32 points = 0, spots = 0, shadowed = 0;
        const SceneLight* nearest = nullptr;
        for (const auto& l : lights_last) {
            ++(l.spot ? spots : points);
            shadowed += l.shadowed ? 1 : 0;
            if (!nearest || distance(l) < distance(*nearest)) {
                nearest = &l;
            }
        }
        std::printf("Lights: %zu in the last frame (%u point, %u spot, %u of them shadowed), %.1f light buffers "
                    "per frame; camera (%.1f %.1f %.1f)%s\n",
                    lights_last.size(), points, spots, shadowed, buffers_total / 600.0, cam[0], cam[1], cam[2],
                    nearest ? "; nearest:" : "");
        if (nearest) {
            print(*nearest);
        }
        frames = buffers_total = 0;
    }
}

} // namespace Vulkan
