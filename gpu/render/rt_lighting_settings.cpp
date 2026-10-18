// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/rt_lighting_settings.h"

#include <algorithm>
#include "render/rtao_geometry.h"

namespace RtLightingSettings {

bool Apply(Settings& s, std::string_view key, float value) {
    const auto flag = [&](bool& target) { target = value != 0.0f; };
    const auto count = [&](uint32_t lo, uint32_t hi) { return std::clamp(uint32_t(std::max(value, 0.0f)), lo, hi); };
    if (key == "emission") s.emission = std::clamp(value, 0.0f, 16.0f);
    else if (key == "emission_at") s.emission_at = count(0u, 1u);
    else if (key == "emission_trace") s.emission_trace = value;
    else if (key == "emission_sat") s.emission_sat = std::clamp(value, 0.0f, 1.0f);
    else if (key == "emission_blur") s.emission_blur = count(0u, 32u);
    else if (key == "fire_light") s.fire_light = std::clamp(value, 0.0f, 64.0f);
    else if (key == "fire_shadow") s.fire_shadow = std::clamp(value, 0.0f, 1.0f);
    else if (key == "fire_range") s.fire_range = std::clamp(value, 0.5f, 100.0f);
    else if (key == "fire_size") s.fire_size = std::clamp(value, 0.0f, 5.0f);
    else if (key == "fire_height") s.fire_height = std::clamp(value, 0.0f, 10.0f);
    else if (key == "fire_wrap") s.fire_wrap = std::clamp(value, 0.0f, 1.0f);
    else if (key == "fire_smooth") s.fire_smooth = std::clamp(value, 0.0f, 0.98f);
    else if (key == "fire_smooth_pos") s.fire_smooth_pos = std::clamp(value, 0.0f, 0.98f);
    else if (key == "fire_pull") s.fire_pull = std::clamp(value, 0.0f, 5.0f);
    else if (key == "fire_min") s.fire_min = std::clamp(value, 0.0f, 100.0f);
    else if (key == "fire_debug") flag(s.fire_debug);
    else if (key == "fire_base") s.fire_base = std::clamp(value, 0.0f, 4.0f);
    else if (key == "fire_spark") s.fire_spark = std::clamp(value, 0.0f, 4.0f);
    else if (key == "fire_base_size") s.fire_base_size = std::clamp(value, 0.0f, 5.0f);
    else if (key == "fire_base_height") s.fire_base_height = std::clamp(value, 0.0f, 10.0f);
    else if (key == "fire_base_smooth") s.fire_base_smooth = std::clamp(value, 0.0f, 0.995f);
    else if (key == "fire_base_fade") s.fire_base_fade = std::clamp(value, 0.0f, 0.999f);
    else if (key == "fire_base_merge") s.fire_base_merge = std::clamp(value, 0.1f, 10.0f);
    else if (key == "fire_spark_size") s.fire_spark_size = std::clamp(value, 0.0f, 5.0f);
    else if (key == "sun_shadow") flag(s.sun_shadow);
    else if (key == "sun_spread") s.sun_spread = std::clamp(value, 0.0f, 0.2f);
    else if (key == "sun_erode") s.sun_erode = count(0u, 32u);
    else if (key == "sun_far") s.sun_far = std::clamp(value, 1.0f, 2000.0f);
    else if (key == "sun_debug") s.sun_debug = count(0u, 7u);
    else if (key == "spot_shadow") flag(s.spot_shadow);
    else if (key == "spot_res") s.spot_res = count(256u, 8192u);
    else if (key == "spot_blur") s.spot_blur = count(0u, 32u);
    else if (key == "spot_soft") s.spot_soft = std::clamp(value, 0.0f, 100.0f);
    else if (key == "spot_near") s.spot_near = std::clamp(value, 0.0f, 10.0f);
    else if (key == "lights_dump") s.lights_dump = value;
    else return false;
    return true;
}

uint32_t ApplyText(Settings& settings, std::string_view text) {
    return Rtao::ForEachSetting(text, [&](std::string_view key, float number) { return Apply(settings, key, number); });
}

} // namespace RtLightingSettings
