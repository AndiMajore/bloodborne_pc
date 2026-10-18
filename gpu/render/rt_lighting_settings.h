// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the settings of the experimental RT lighting effects on the RTAO's scene (ticket 65):
// <user>/rtlighting.txt, re-read while the game runs (vk_rt_lighting.cpp). Everything off by
// default; the RTAO's own settings stay in rtao.txt. Pure (tested in rtao-geometry-test).

#pragma once

#include <cstdint>
#include <string_view>

namespace RtLightingSettings {

struct Settings {
    // The transparents' light (63, vk_rt_emission.cpp): fires, sparks and glows.
    float emission = 0.0f;       // the emission image read by SSGI, the probes and the reflections, x this (0: off)
    uint32_t emission_at = 1;    // 0: the first full-screen pass with the scene depth after the transparents, 1: the first draw after them
    float emission_trace = 0.0f; // a new value: one frame's draws after the opaque scene are logged
    float emission_sat = 0.35f;  // only light this saturated counts in full (fire, embers; not grey smoke); 0: all
    uint32_t emission_blur = 8;  // the emission's Gaussian radius in quarter-size texels; 0: none
    // The fires as lights in the RT SSGI (63).
    float fire_light = 0.0f;      // x their light (0: off)
    float fire_shadow = 0.0f;     // 0..1: their unshadowed light taken away again (the game's own fire light, darkened where blocked)
    float fire_range = 12.0f;     // metres a fire light reaches
    float fire_size = 0.75f;      // metres: a fire light's cylinder's radius
    float fire_height = 1.0f;     // metres: its height, a quarter below the light's point
    float fire_wrap = 0.4f;       // 0..1: the cosine wrapped round (thin swaying leaves stop strobing)
    float fire_smooth = 0.9f;     // 0..0.98: the share of last frame's power (flicker)
    float fire_smooth_pos = 0.5f; // 0..0.98: the share of last frame's place
    float fire_pull = 0.3f;       // metres a light sits in front of the surface behind the flame
    float fire_min = 0.3f;        // a tile's brightest emission texel must reach this to make a light
    // Sticky lights: a light stays at its place unless the tile's new peak is this much brighter
    // than the flame there (0: it follows the peak every frame) ...
    float fire_switch = 0.0f;
    uint32_t fire_hold = 0;       // ... and for this many frames after it moved
    bool fire_debug = false;      // the cylinders drawn into the scene
    // Two-part fires (67): per fire a steady base (the tile lights near each other merged, their
    // summed power averaged over seconds) as a tall slim cylinder, the tile lights as spheres on top.
    float fire_base = 0.0f;         // 0: off (every tile light a cylinder); else x the base's averaged power
    float fire_spark = 0.5f;        // with fire_base: x the tile lights' (sparks') power
    float fire_base_size = 0.3f;    // metres: the base cylinder's radius
    float fire_base_height = 2.5f;  // metres: its height, 40 % below the fire's place
    float fire_base_smooth = 0.98f; // 0..0.995: the share of last frame's base power and place
    float fire_base_fade = 0.985f;  // 0..0.999: per frame without its fire on screen (fades over seconds)
    float fire_base_merge = 1.5f;   // metres (horizontal): tile lights within this are one fire
    float fire_spark_size = 0.3f;   // metres: a spark sphere's radius
    // Ray-traced shadows (64, vk_rt_shadows.cpp).
    bool sun_shadow = false;      // the sun's shadow by ray tracing into the game's shadow mask
    float sun_spread = 0.01f;     // radians the sun rays spread over
    uint32_t sun_erode = 4;       // pixels the game's shadow is shrunk by where it is kept
    float sun_far = 200.0f;       // metres a sun ray reaches
    uint32_t sun_debug = 0;       // 1 the rays alone, 2 reversed, 4 the mask in full shadow (a write test)
    bool spot_shadow = false;     // the shadowed spot's 512x512 map replaced by a ray-traced one
    uint32_t spot_res = 1024;     // its size
    uint32_t spot_blur = 3;       // texels of Gaussian on its moments (soft edges)
    float spot_soft = 0.2f;       // metres: the blurred map's largest depth spread (overlap seams)
    float spot_near = 0.3f;       // metres from the light its rays start
    // The game's light list (59, vk_rt_lights.cpp).
    float lights_dump = 0.0f;     // a new value: the next frame's light list is logged in full

    /// The emission image is wanted (by itself or for the fire lights).
    [[nodiscard]] bool EmissionWanted() const noexcept {
        return emission > 0.0f || fire_light > 0.0f;
    }
};

/// One key=value; false for an unknown key. Values are clamped to their ranges.
bool Apply(Settings& settings, std::string_view key, float value);

/// The live file's text (key=value tokens, `#` comments). Returns how many were applied.
uint32_t ApplyText(Settings& settings, std::string_view text);

} // namespace RtLightingSettings
