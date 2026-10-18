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
    float emission_sat = 0.35f;  // only light this saturated counts in full (fire, embers; not grey smoke); 0: all
    uint32_t emission_blur = 8;  // the emission's Gaussian radius in quarter-size texels; 0: none
    // The fires as lights in the RT SSGI (63, 67). Per fire two parts: a steady base (the tile
    // lights near each other merged, their summed power averaged over seconds, anchored to the
    // game's own fire light) as a large upright cylinder, and the tile lights as spheres (sparks)
    // carrying the flicker. The defaults are the ones chosen at the Central Yharnam stakes
    // (2026-10-18), on by default; they light only while the RT bounce light (ssgi.txt rt=1) reads them.
    float fire_light = 0.3f;      // x their light (0: off)
    float fire_range = 12.0f;     // metres a fire light reaches
    float fire_wrap = 1.0f;       // 0..1: the cosine wrapped round (thin swaying leaves stop strobing)
    uint32_t fire_rays = 1;       // shadow rays to the fires per half-res pixel (1-8, at most SSGI's rays)
    float fire_smooth = 0.9f;     // 0..0.98: the share of last frame's power (flicker)
    float fire_smooth_pos = 0.9f; // 0..0.98: the share of last frame's place
    float fire_pull = 0.3f;       // metres a light sits in front of the surface behind the flame
    float fire_min = 0.4f;        // a tile's brightest emission texel must reach this to make a light
    // Sticky lights: a light stays at its place unless the tile's new peak is this much brighter
    // than the flame there (0: it follows the peak every frame) ...
    float fire_switch = 0.5f;
    uint32_t fire_hold = 15;      // ... and for this many frames after it moved
    float fire_merge = 1.0f;      // metres (horizontally): lights nearer than this merged into the strongest (0: off)
    bool fire_debug = false;      // the bases (cyan cylinders) and the sparks (orange spheres) drawn into the scene
    float fire_base = 1.0f;         // x a base's averaged power (its brightness)
    float fire_base_lift = 0.0f;    // metres: the base moved up (negative: down) from the game's fire light it is anchored to
    float fire_spark = 0.5f;        // x a spark's power
    float fire_base_size = 0.2f;    // metres: the base cylinder's radius
    float fire_base_height = 2.0f;  // metres: its height, 40 % below the fire's place
    float fire_base_smooth = 0.98f; // 0..0.995: the share of last frame's base power and place
    float fire_base_fade = 0.985f;  // 0..0.999: per frame without its fire on screen (fades over seconds)
    float fire_base_merge = 1.5f;   // metres (horizontal): tile lights within this are one fire
    float fire_spark_size = 0.5f;   // metres: a spark sphere's radius
    // Large fires (the pyres): a fire whose own game light is brighter than fire_large (the
    // light's colour as luminance: the Central Yharnam stakes ~4.3, the square's pyre ~9.7, lamps
    // ~2) takes these instead of the base and spark settings above. 0: no large fires.
    float fire_large = 7.0f;
    float fire_large_base = 1.5f;          // x a large base's averaged power
    float fire_large_base_size = 0.8f;     // metres: its cylinder's radius
    float fire_large_base_height = 4.0f;   // metres: its height, 40 % below the fire's place
    float fire_large_base_lift = 0.0f;     // metres: moved up (negative: down)
    float fire_large_spark = 0.5f;         // x a large fire's sparks' power
    float fire_large_spark_size = 0.8f;    // metres: their spheres' radius
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
