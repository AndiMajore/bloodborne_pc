// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's local light records (ticket 59), decoded. Its light passes draw a 16x15
// grid of screen tiles per pixel-shader variant (vs LightQuadVs), every tile of a variant binding
// the same constant buffer of records (ps slot 1; slot 0 the per-frame block of
// LightFrameBlockSize bytes). The records, from the light shaders, zero-padded to the buffer's size:
//   point, 16 floats: [0..2] position (view), [3] 1 / radius^2 (falloff (1 - sqrt(d^2 [3]))^2),
//     [4..6] colour x intensity, [8..10] specular colour, [12] specular scale, [13] power / 32;
//   spot, 36 floats: [0..2] position, [3] range (metres, >= 1: how the two are told apart),
//     [4..6] colour, [8..10] specular colour, [15] > 0: shadowed, [16..31] the view -> light clip
//     matrix (|[16..18]| the x scale: the cone, [28..30] the beam direction), [32] the shadow's strength.
// Pure (tested in rtao-geometry-test from a frame capture's buffers).

#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace RtLightRecords {

constexpr uint64_t LightQuadVs = 0x8747a367;  // the light passes' tile quad
constexpr uint32_t LightFrameBlockSize = 208; // ps slot 0 of a light draw
constexpr uint32_t PointFloats = 16, SpotFloats = 36;
constexpr uint64_t SpotLightPs = 0x82951fa5;  // the shadowed spot (a variance shadow map)

struct SceneLight {
    std::array<float, 3> position{};  // world
    float radius = 0.0f;              // metres (point: 1 / sqrt(record[3]); spot: its range)
    std::array<float, 3> color{};     // x intensity
    std::array<float, 3> direction{}; // world, spots
    float cone = 0.0f;                // half-angle in radians, spots
    uint64_t ps_hash = 0;             // the variant that drew it
    bool spot = false;
    bool shadowed = false;            // record[15] > 0
};

/// The shadowed spot's record (SpotLightPs) in world space: its world -> light clip matrix (rows
/// x, y, z, w), its position and range.
struct SpotRecord {
    std::array<std::array<float, 4>, 4> world_to_clip{};
    std::array<float, 3> light{};
    float range = 0.0f;
    bool valid = false;
};

/// A light-pass buffer's size can hold records (not the 864-byte camera block of the global pass).
[[nodiscard]] bool IsRecordBuffer(uint64_t bytes);

/// Decodes `data` (a light buffer, view space of the camera with `view` world -> view rows and
/// `inv_view` view -> world rows) into world-space lights appended to `lights` (a light already
/// there is not added twice: every tile variant has its own copy of the same lights). The first
/// record of a SpotLightPs buffer fills `spot` when it is not valid yet. Returns the records read.
uint32_t Decode(std::span<const float> data, const std::array<float, 12>& view,
                const std::array<float, 12>& inv_view, uint64_t ps_hash, std::vector<SceneLight>& lights,
                SpotRecord* spot);

} // namespace RtLightRecords
