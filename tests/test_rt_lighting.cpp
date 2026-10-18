// SPDX-License-Identifier: GPL-2.0-or-later
// The RT lighting experiments' logic without Vulkan (ticket 65.5): the game's light records
// decoded from one frame's captured light-pass buffers, the point/spot rule, the padding, the
// shadowed spot's record; the end of the transparents on a recorded draw sequence; the settings.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>
#include "render/rt_emission_trigger.h"
#include "render/rt_light_records.h"
#include "render/rt_lighting_settings.h"

#include "fixtures/rt_light_buffers.inc"

using namespace RtLightRecords;

static constexpr std::array<float, 12> Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

static bool Near(float a, float b, float eps = 1e-3f) {
    return std::abs(a - b) <= eps * std::max(1.0f, std::abs(b));
}

// The buffer as bound: the captured floats, zero-padded to its size (as the game pads it).
static std::vector<float> Bound(std::span<const float> captured, unsigned bytes) {
    std::vector<float> out(bytes / sizeof(float), 0.0f);
    std::copy_n(captured.begin(), std::min(captured.size(), out.size()), out.begin());
    return out;
}

static void Sizes() {
    assert(IsRecordBuffer(432) && IsRecordBuffer(144) && IsRecordBuffer(1296));
    assert(!IsRecordBuffer(0) && !IsRecordBuffer(864)); // the global pass's camera block
    assert(!IsRecordBuffer(9216 + 16) && !IsRecordBuffer(100));
}

static void Points() {
    // ps d22cbfd1 (432 bytes): three points, then the padding.
    const auto data = Bound(Buffer_d22cbfd1, Buffer_d22cbfd1_bytes);
    std::vector<SceneLight> lights;
    assert(Decode(data, Identity, Identity, 0xd22cbfd1, lights, nullptr) == 3);
    assert(lights.size() == 3);
    assert(!lights[0].spot && Near(lights[0].position[0], 7.89731f) && Near(lights[0].position[2], -5.20815f));
    assert(Near(lights[0].radius, 15.0f)); // 1 / sqrt(0.00444444)
    assert(Near(lights[0].color[0], 1.8f) && Near(lights[0].color[2], 0.877066f));
    assert(Near(lights[1].position[1], -0.933655f));
    assert(Near(lights[2].radius, 3.5f) && Near(lights[2].color[1], 0.467128f));
    // Every tile variant has its own copy of the same lights: decoded again, none added.
    assert(Decode(data, Identity, Identity, 0xd22cbfd1, lights, nullptr) == 3 && lights.size() == 3);

    // ps 0b2ace4c (1296 bytes): nine points of two radii, then the padding.
    const auto many = Bound(Buffer_0b2ace4c, Buffer_0b2ace4c_bytes);
    std::vector<SceneLight> more;
    assert(Decode(many, Identity, Identity, 0x0b2ace4c, more, nullptr) == 9);
    assert(more.size() == 9 && Near(more[0].radius, 10.0f) && Near(more[8].radius, 6.0f));
    for (const auto& l : more) {
        assert(!l.spot && !l.shadowed);
    }
}

static void Spots() {
    // ps bae5454e (144 bytes): one spot ([3] = 100 >= 1: the range), the sky's.
    const auto data = Bound(Buffer_bae5454e, Buffer_bae5454e_bytes);
    std::vector<SceneLight> lights;
    SpotRecord spot;
    assert(Decode(data, Identity, Identity, 0xbae5454e, lights, &spot) == 1);
    assert(lights.size() == 1 && lights[0].spot && !lights[0].shadowed);
    assert(Near(lights[0].radius, 100.0f) && Near(lights[0].position[1], 71.5875f));
    // Its cone from the matrix's x row: |(0.265, 0.331, 0.391)| = 0.576, half-angle atan(1 / 0.576).
    assert(Near(lights[0].cone * 2.0f * 57.29578f, 120.1f, 2e-3f));
    // Its beam along [28..30], normalised.
    const float len = std::sqrt(0.151207f * 0.151207f + 0.701687f * 0.701687f + 0.696256f * 0.696256f);
    assert(Near(lights[0].direction[1], -0.701687f / len));
    assert(!spot.valid); // not the shadowed spot's variant

    // ps 82951fa5 (the shadowed spot's variant): its record kept in world space.
    const auto shadowed = Bound(Buffer_82951fa5, Buffer_82951fa5_bytes);
    std::vector<SceneLight> more;
    assert(Decode(shadowed, Identity, Identity, SpotLightPs, more, &spot) == 1);
    assert(spot.valid && Near(spot.range, 76.6f) && Near(spot.light[0], 14.73f));
    assert(Near(spot.world_to_clip[3][0], -0.561477f) && Near(spot.world_to_clip[3][3], 22.0816f));
    assert(Near(spot.world_to_clip[0][0], 3.06725f));
}

static void ViewToWorld() {
    // A camera turned 90 degrees about y and moved: world = R^T (view - t).
    // world -> view rows: x' = -z + 1, y' = y + 2, z' = x + 3; view -> world: x = z' - 3, y = y' - 2, z = 1 - x'.
    const std::array<float, 12> view{0, 0, -1, 1, 0, 1, 0, 2, 1, 0, 0, 3};
    const std::array<float, 12> inv_view{0, 0, 1, -3, 0, 1, 0, -2, -1, 0, 0, 1};
    const auto data = Bound(Buffer_82951fa5, Buffer_82951fa5_bytes);
    std::vector<SceneLight> lights;
    SpotRecord spot;
    Decode(data, view, inv_view, SpotLightPs, lights, &spot);
    // The record's view position (14.73, 8.64051, -14.427) in the world.
    assert(Near(lights[0].position[0], -14.427f - 3.0f) && Near(lights[0].position[1], 8.64051f - 2.0f) &&
           Near(lights[0].position[2], 1.0f - 14.73f));
    // world -> clip applied to the light's world position gives what view -> clip gave its view position.
    const auto& m = spot.world_to_clip;
    const auto& p = lights[0].position;
    const float w_world = m[3][0] * p[0] + m[3][1] * p[1] + m[3][2] * p[2] + m[3][3];
    const float w_view = -0.561477f * 14.73f + -0.33512f * 8.64051f + 0.756597f * -14.427f + 22.0816f;
    assert(Near(w_world, w_view));
}

static void Padding() {
    // Zeros end the list; a non-finite record ends it too (never past the buffer).
    std::vector<float> data(64, 0.0f);
    std::vector<SceneLight> lights;
    assert(Decode(data, Identity, Identity, 0, lights, nullptr) == 0 && lights.empty());
    data[0] = 1.0f; data[3] = 0.25f; // a point of radius 2
    data[16] = NAN; data[19] = 0.25f;
    assert(Decode(data, Identity, Identity, 0, lights, nullptr) == 1 && Near(lights[0].radius, 2.0f));
    // A spot's 36 floats that would run past the buffer: not read.
    std::vector<float> cut(20, 0.0f);
    cut[0] = 1.0f; cut[3] = 50.0f;
    std::vector<SceneLight> none;
    assert(Decode(cut, Identity, Identity, 0, none, nullptr) == 0 && none.empty());
}

static void EmissionEnd() {
    using EmissionRules::Trigger;
    // The draw sequence after the opaque scene at the Central Yharnam burning stakes (emission
    // trace of 2026-10-17): 86 transparents into the scene with its depth, then the fog composite
    // without a depth target, then a draw into another image.
    Trigger t;
    assert(!t.Pending() && !t.OnDraw(1, true));
    t.Begin(1);
    for (int i = 0; i < 86; ++i) {
        assert(!t.OnDraw(1, true));
    }
    assert(t.OnDraw(1, false) && t.Draws() == 87); // the fog: taken before it
    assert(!t.Pending() && !t.OnDraw(2, true));    // once
    // Another target ends them as well; a draw without a target does not.
    t.Begin(1);
    assert(!t.OnDraw(0, false) && t.OnDraw(5, true));
    // Taken another way (emission_at=0): nothing more waits.
    t.Begin(1);
    t.Taken();
    assert(!t.OnDraw(2, false));
}

static void LightingSettings() {
    using RtLightingSettings::ApplyText;
    const RtLightingSettings::Settings defaults;
    assert(!defaults.sun_shadow && !defaults.spot_shadow && defaults.fire_light == 0.0f &&
           defaults.emission == 0.0f && !defaults.EmissionWanted()); // everything off
    RtLightingSettings::Settings s;
    assert(ApplyText(s, "fire_light=0.35 # the fires\nspot_shadow=1\tsun_shadow=0\nunknown=3 fire_smooth=2\n") == 4);
    assert(Near(s.fire_light, 0.35f) && s.spot_shadow && !s.sun_shadow && s.EmissionWanted());
    assert(Near(s.fire_smooth, 0.98f)); // clamped
    assert(ApplyText(s, "# spot_res=4096\nspot_res=100 spot_blur=x") == 1 && s.spot_res == 256);
    // Sticky lights: off by default.
    assert(defaults.fire_switch == 0.0f && defaults.fire_hold == 0);
    assert(ApplyText(s, "fire_switch=0.5 fire_hold=12") == 2 && Near(s.fire_switch, 0.5f) && s.fire_hold == 12);
    // Two-part fires (67): off by default, clamped.
    assert(defaults.fire_base == 0.0f);
    assert(ApplyText(s, "fire_base=0.7 fire_base_smooth=1 fire_spark=0.4") == 3);
    assert(Near(s.fire_base, 0.7f) && Near(s.fire_base_smooth, 0.995f) && Near(s.fire_spark, 0.4f));
    // The RTAO's own keys are not the experiments': rtlighting.txt ignores them.
    assert(ApplyText(s, "rays=4 strength=0.5") == 0);
}

int main() {
    Sizes();
    Points();
    Spots();
    ViewToWorld();
    Padding();
    EmissionEnd();
    LightingSettings();
    std::puts("rt-lighting-test: ok");
    return 0;
}
