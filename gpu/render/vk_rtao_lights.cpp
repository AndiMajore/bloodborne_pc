// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's local lights as a per-frame list (ticket 59; vk_rtao.h). The light passes
// draw screen tiles, every tile of a pixel-shader variant binding the same buffer of light records
// (raster_bind.cpp hands it here): read once a frame per buffer, the view-space records become
// world-space lights. The records (from the light shaders), zero-padded to the buffer's size:
//   point, 16 floats: [0..2] position (view), [3] 1 / radius^2 (falloff (1 - sqrt(d^2 [3]))^2),
//     [4..6] colour x intensity, [8..10] specular colour, [12] specular scale, [13] power / 32;
//   spot, 36 floats: [0..2] position, [3] range (metres, >= 1: how the two are told apart),
//     [4..6] colour, [8..10] specular colour, [15] > 0: shadowed (a shadow map), [16..31] the
//     view -> light clip matrix (|[16..18]| the x scale: the cone, [28..30] the beam direction), [32].
#include "render/vk_rtao.h"
#include "render/vk_rtao_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "upscale/vk_camera_motion.h"

namespace Vulkan {

void RayTracedAO::OnLightBuffer(VAddr address, u64 size, u64 ps_hash) {
    if (size == 0 || size % 16 != 0 || size > 64 * LightRecordSize || size == 864) {
        return; // 864: the camera block of the global light pass, not lights
    }
    // Every tile draw binds its own copy of its variant's buffer (~960 a frame): a copy is known by
    // its size and first record.
    LightBufferKey key{.size = size};
    std::memcpy(key.head.data(), reinterpret_cast<const void*>(address), sizeof(key.head));
    if (std::ranges::find(lights_buffers_seen, key) != lights_buffers_seen.end()) {
        return;
    }
    lights_buffers_seen.push_back(key);
    const auto inv_view = camera_motion.InvView(); // view -> world rows
    const auto to_world = [&](const float* v, float w) {
        std::array<float, 3> out{};
        for (int r = 0; r < 3; ++r) {
            out[r] = inv_view[r * 4] * v[0] + inv_view[r * 4 + 1] * v[1] + inv_view[r * 4 + 2] * v[2] +
                     inv_view[r * 4 + 3] * w;
        }
        return out;
    };
    const auto* data = reinterpret_cast<const float*>(address);
    const u64 count = size / sizeof(float);
    for (u64 at = 0; at + 16 <= count;) {
        const float* r = data + at;
        if (r[0] == 0.0f && r[1] == 0.0f && r[2] == 0.0f && r[3] == 0.0f) {
            break; // the padding
        }
        const bool spot = r[3] >= 1.0f;
        at += spot ? 36 : 16;
        if (at > count || !std::isfinite(r[0]) || !std::isfinite(r[1]) || !std::isfinite(r[2]) ||
            !(r[3] > 0.0f)) {
            break;
        }
        SceneLight light{
            .position = to_world(r, 1.0f),
            .radius = spot ? r[3] : 1.0f / std::sqrt(r[3]),
            .color = {r[4], r[5], r[6]},
            .ps_hash = ps_hash,
            .spot = spot,
            .shadowed = spot && r[15] > 0.0f,
        };
        if (spot) {
            const float len = std::sqrt(r[28] * r[28] + r[29] * r[29] + r[30] * r[30]);
            if (len > 0.0f) {
                const float d[3] = {r[28] / len, r[29] / len, r[30] / len};
                light.direction = to_world(d, 0.0f);
            }
            // The matrix's x row is the light's right axis x its projection scale (view -> light is
            // rigid): the half-angle is atan(1 / scale).
            const float scale = std::sqrt(r[16] * r[16] + r[17] * r[17] + r[18] * r[18]);
            light.cone = scale > 0.0f ? std::atan(1.0f / scale) : 0.0f; // half-angle, radians
        }
        // Every tile variant gets its own copy of the same lights: kept once.
        const auto same = [&](const SceneLight& o) {
            return o.spot == light.spot && std::abs(o.position[0] - light.position[0]) < 0.01f &&
                   std::abs(o.position[1] - light.position[1]) < 0.01f &&
                   std::abs(o.position[2] - light.position[2]) < 0.01f && o.color == light.color;
        };
        if (std::ranges::none_of(lights_building, same)) {
            lights_building.push_back(light);
        }
    }
}

void RayTracedAO::FinishLightFrame() {
    lights_buffers_total += u32(lights_buffers_seen.size());
    lights_last.swap(lights_building);
    lights_building.clear();
    lights_buffers_seen.clear();

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
    // lights_dump=<new value> in rtao.txt: the whole list of the frame just finished.
    if (settings.lights_dump != lights_dump_seen) {
        lights_dump_seen = settings.lights_dump;
        std::printf("Lights: frame %u, camera (%.2f %.2f %.2f), %zu lights:\n", frame, cam[0], cam[1], cam[2],
                    lights_last.size());
        for (const auto& l : lights_last) {
            print(l);
        }
    }
    if (++lights_frames == 600) {
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
                    lights_last.size(), points, spots, shadowed, lights_buffers_total / 600.0, cam[0], cam[1],
                    cam[2], nearest ? "; nearest:" : "");
        if (nearest) {
            print(*nearest);
        }
        lights_frames = lights_buffers_total = 0;
    }
}

} // namespace Vulkan
