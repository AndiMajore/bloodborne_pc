// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/rt_light_records.h"

#include <algorithm>
#include <cmath>

namespace RtLightRecords {

bool IsRecordBuffer(uint64_t bytes) {
    return bytes != 0 && bytes % 16 == 0 && bytes <= 64 * SpotFloats * sizeof(float) && bytes != 864;
}

uint32_t Decode(std::span<const float> data, const std::array<float, 12>& view,
                const std::array<float, 12>& inv_view, uint64_t ps_hash, std::vector<SceneLight>& lights,
                SpotRecord* spot) {
    const auto to_world = [&](const float* v, float w) {
        std::array<float, 3> out{};
        for (int r = 0; r < 3; ++r) {
            out[r] = inv_view[r * 4] * v[0] + inv_view[r * 4 + 1] * v[1] + inv_view[r * 4 + 2] * v[2] +
                     inv_view[r * 4 + 3] * w;
        }
        return out;
    };
    uint32_t records = 0;
    for (size_t at = 0; at + PointFloats <= data.size();) {
        const float* r = data.data() + at;
        if (r[0] == 0.0f && r[1] == 0.0f && r[2] == 0.0f && r[3] == 0.0f) {
            break; // the padding
        }
        const bool is_spot = r[3] >= 1.0f;
        at += is_spot ? SpotFloats : PointFloats;
        if (at > data.size() || !std::isfinite(r[0]) || !std::isfinite(r[1]) || !std::isfinite(r[2]) ||
            !(r[3] > 0.0f)) {
            break;
        }
        ++records;
        SceneLight light{
            .position = to_world(r, 1.0f),
            .radius = is_spot ? r[3] : 1.0f / std::sqrt(r[3]),
            .color = {r[4], r[5], r[6]},
            .ps_hash = ps_hash,
            .spot = is_spot,
            .shadowed = is_spot && r[15] > 0.0f,
        };
        if (is_spot) {
            const float len = std::sqrt(r[28] * r[28] + r[29] * r[29] + r[30] * r[30]);
            if (len > 0.0f) {
                const float d[3] = {r[28] / len, r[29] / len, r[30] / len};
                light.direction = to_world(d, 0.0f);
            }
            // The matrix's x row is the light's right axis x its projection scale (view -> light is
            // rigid): the half-angle is atan(1 / scale).
            const float scale = std::sqrt(r[16] * r[16] + r[17] * r[17] + r[18] * r[18]);
            light.cone = scale > 0.0f ? std::atan(1.0f / scale) : 0.0f;
            if (spot && !spot->valid && ps_hash == SpotLightPs) {
                // World -> light clip: the record's view -> light clip x world -> view.
                for (int row = 0; row < 4; ++row) {
                    for (int c = 0; c < 4; ++c) {
                        float v = c == 3 ? r[16 + row * 4 + 3] : 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            v += r[16 + row * 4 + k] * view[k * 4 + c];
                        }
                        spot->world_to_clip[row][c] = v;
                    }
                }
                spot->light = light.position;
                spot->range = r[3];
                spot->valid = true;
            }
        }
        const auto same = [&](const SceneLight& o) {
            return o.spot == light.spot && std::abs(o.position[0] - light.position[0]) < 0.01f &&
                   std::abs(o.position[1] - light.position[1]) < 0.01f &&
                   std::abs(o.position[2] - light.position[2]) < 0.01f && o.color == light.color;
        };
        if (std::ranges::none_of(lights, same)) {
            lights.push_back(light);
        }
    }
    return records;
}

} // namespace RtLightRecords
