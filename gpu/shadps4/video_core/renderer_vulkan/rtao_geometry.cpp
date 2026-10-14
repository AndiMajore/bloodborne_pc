// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/rtao_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace Rtao {

bool Requested() {
    static const bool requested = [] {
        const char* env = std::getenv("BB_RTAO");
        return env && env[0] == '1';
    }();
    return requested;
}

u64 KeyOf(u64 base, u32 occurrence) {
    u64 key = (base ^ (u64(occurrence + 1) * 0x9E3779B97F4A7C15ull)) * 0xBF58476D1CE4E5B9ull;
    key ^= key >> 31;
    return key ? key : 1;
}

bool StatsMatch(const ProbeStats& a, const ProbeStats& b, const std::array<float, 3>& camera) {
    if (a.count != b.count || a.count == 0) {
        return false;
    }
    // World positions are rebuilt from clip space each frame: allow for the rounding, which grows
    // with the distance from the camera.
    const float dx = a.centroid[0] - camera[0], dy = a.centroid[1] - camera[1], dz = a.centroid[2] - camera[2];
    const float tol = 0.02f + 2e-4f * std::sqrt(dx * dx + dy * dy + dz * dz);
    for (int i = 0; i < 3; ++i) {
        if (std::abs(a.centroid[i] - b.centroid[i]) > tol || std::abs(a.min[i] - b.min[i]) > tol ||
            std::abs(a.max[i] - b.max[i]) > tol || std::abs(a.first[i] - b.first[i]) > tol) {
            return false;
        }
    }
    return true;
}

bool TooBig(const ProbeStats& stats, float max_size) {
    if (max_size <= 0.0f) {
        return false;
    }
    const float dx = stats.max[0] - stats.min[0], dy = stats.max[1] - stats.min[1], dz = stats.max[2] - stats.min[2];
    return dx * dx + dy * dy + dz * dz > max_size * max_size;
}

Observation Observe(CacheState& state, const ProbeStats& stats, bool sibling_match,
                    const std::array<float, 3>& camera) {
    if (!state.have_stats) {
        state.stats = stats;
        state.have_stats = true;
        state.stable = 1;
        return Observation::First;
    }
    if (StatsMatch(state.stats, stats, camera)) {
        ++state.stable; // the reference stays the first observation: slow drift is not static
        return Observation::Same;
    }
    if (sibling_match) {
        state.stable = 0; // the order is not stable: no promotion from this slot for now
        return Observation::Reordered;
    }
    const bool was_cached = state.cached;
    if (was_cached) { // cached, then it moved (a door, a lift): back to per frame
        state.cached = false;
        ++state.evictions;
    }
    state.stats = stats;
    state.stable = 1;
    return was_cached ? Observation::ChangedEvict : Observation::Changed;
}

bool Promotable(const CacheState& state, bool skinned, bool too_big, u32 promote_after) {
    return !skinned && !state.cached && state.evictions < 3 && state.stable >= promote_after && !too_big;
}

bool Apply(Settings& s, std::string_view key, float value) {
    const auto flag = [&](bool& target) { target = value != 0.0f; };
    if (key == "enabled") flag(s.active);
    else if (key == "strength") s.strength = std::clamp(value, 0.0f, 1.0f);
    else if (key == "power") s.power = std::clamp(value, 0.1f, 8.0f);
    else if (key == "radius") s.radius = std::max(value, 0.05f);
    else if (key == "history") s.history = std::clamp(value, 0.0f, 0.98f);
    else if (key == "sigma") s.sigma = std::clamp(value, 0.0f, 4.0f);
    else if (key == "rays") s.rays = std::clamp(u32(std::max(value, 0.0f)), 1u, 32u);
    else if (key == "debug") s.debug = std::clamp(u32(std::max(value, 0.0f)), 0u, 4u);
    else if (key == "stage") s.stage = std::clamp(u32(std::max(value, 0.0f)), 0u, 5u);
    else if (key == "denoise") s.denoise = std::clamp(u32(std::max(value, 0.0f)), 1u, 2u);
    else if (key == "blur") flag(s.blur);
    else if (key == "replace") flag(s.replace_mask);
    else if (key == "normals") flag(s.depth_normals);
    else if (key == "pipeline") flag(s.use_rt_pipeline);
    else if (key == "snap") s.snap_far = std::max(value, 0.0f);
    else if (key == "far") s.ao_far = std::max(value, 0.0f);
    else if (key == "offset") s.origin_offset = std::clamp(value, 0.0f, 0.1f);
    else if (key == "maxn") s.max_n = std::clamp(value, 1.0f, 255.0f);
    else if (key == "clipk") s.clip_k = std::clamp(value, 0.1f, 10.0f);
    else if (key == "clipfloor") s.clip_floor = std::clamp(value, 0.0f, 1.0f);
    else if (key == "subpixel") flag(s.subpixel);
    else if (key == "falloff") s.falloff = value != 0.0f ? 1u : 0u;
    else if (key == "period") s.noise_period = std::clamp(u32(std::max(value, 0.0f)), 1u, 1024u);
    else if (key == "cache") flag(s.cache_enabled);
    else if (key == "promote") s.promote_after = std::clamp(u32(std::max(value, 0.0f)), 2u, 600u);
    else if (key == "maxsize") s.cache_max_size = std::max(value, 0.0f);
    else if (key == "alpha") flag(s.include_alpha);
    else if (key == "decals") flag(s.include_nowrite);
    else if (key == "ringtest") flag(s.ring_test);
    else if (key == "build") flag(s.build_fast_trace);
    else if (key == "dump") s.dump_request = value;
    else return false;
    return true;
}

u32 ApplyText(Settings& settings, std::string_view text, bool& history_reset) {
    u32 applied = 0;
    size_t pos = 0;
    const auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (pos < text.size()) {
        while (pos < text.size() && space(text[pos])) ++pos;
        size_t end = pos;
        while (end < text.size() && !space(text[end])) ++end;
        const std::string_view token = text.substr(pos, end - pos);
        pos = end;
        if (token.empty() || token[0] == '#') {
            // A comment runs to the end of its line.
            if (!token.empty()) {
                while (pos < text.size() && text[pos] != '\n') ++pos;
            }
            continue;
        }
        const size_t eq = token.find('=');
        if (eq == std::string_view::npos || eq == 0) {
            continue;
        }
        const std::string value(token.substr(eq + 1));
        char* parse_end = nullptr;
        const float number = std::strtof(value.c_str(), &parse_end);
        if (parse_end == value.c_str()) {
            continue;
        }
        const u32 denoise_before = settings.denoise;
        if (Apply(settings, token.substr(0, eq), number)) {
            ++applied;
            history_reset |= settings.denoise != denoise_before;
        }
    }
    return applied;
}

} // namespace Rtao
