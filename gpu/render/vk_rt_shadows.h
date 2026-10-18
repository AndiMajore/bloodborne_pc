// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (experimental, ticket 64; rtlighting.txt sun_shadow=, spot_shadow=): ray-traced shadows
// on the RTAO's scene, two of the game's shadows replaced where it reads them.
// Sun: the game's sun shadow chain (Central Yharnam): its cascaded shadow map resolved into a
// screen mask (ps f55643de), blurred (11ffdfb5), written with the AO into the final mask
// (a291c498: x the AO; the shadow in z, which the sun's light d92266b9 reads, or in y for the
// other region, which the global pass 3fe9e964 reads, and that one comes first). Before the first
// of the two: one ray per pixel towards the sun (its world direction from f55643de's camera block,
// [80..82]), the game's shadow where the ray leaves the scene without a hit (rt_sun_shadow.comp).
// Spot: the shadowed spot (ps 82951fa5) samples a 512x512 variance shadow map; from the light
// list's record of it (the frame before) a larger map is ray traced before the light passes and
// sampled in the game's place (raster_textures.cpp; rt_spot_map.comp).

#pragma once

#include <array>
#include "common/types.h"
#include "render/rt_lighting_settings.h"
#include "render/vk_rt_view_cache.h"
#include "textures/image.h"
#include "textures/types.h"
#include "vulkan/vk_common.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {
class Instance;
class Scheduler;
class Runtime;
class CameraMotion;
class RayTracedAO;
class RtLights;

class RtShadows {
public:
    RtShadows(const Instance& instance, Scheduler& scheduler, VideoCore::TextureCache& texture_cache,
              Runtime& runtime, CameraMotion& camera_motion, const RayTracedAO& rtao, const RtLights& lights,
              const RtLightingSettings::Settings& settings);
    ~RtShadows();

    /// Fragment-shader constants wanted (the sun's direction and the mask's encoding).
    [[nodiscard]] bool WantsConstants() const noexcept {
        return settings.sun_shadow;
    }
    /// Per-draw notes wanted (the mask pass, the sun's or global light's draw).
    [[nodiscard]] bool WantsDraws() const noexcept {
        return settings.sun_shadow || settings.spot_shadow;
    }
    void OnFrameStart(u32 frame);
    void OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal);
    /// A fragment shader's buffer as bound (by size: the slot numbering differs between paths).
    void OnShaderConstants(u64 hash, VAddr address, u64 size);
    /// Before each draw (its first colour target and pixel shader).
    void OnDraw(VideoCore::ImageId color, u64 ps_hash);
    /// The light passes' textures: the spot's own map is noted (its sixth image); a texture at
    /// that address is then bound as the RT map.
    void OnSpotMapImage(VAddr address, vk::Format format);
    [[nodiscard]] vk::ImageView SpotMapView(VAddr address) const noexcept;

private:
    bool EnsureSun(u32 w, u32 h);
    void RecordSun();
    void RecordSpotMap();

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    const RayTracedAO& rtao;
    const RtLights& lights;
    const RtLightingSettings::Settings& settings;
    RtViewCache views;
    u32 frame = 0;
    VideoCore::ImageId depth_id{}, normal_id{};

    // The sun.
    std::array<float, 3> sun_dir{};  // world, towards the sun
    // The mask pass's regions (its draws): channel (1 y, 2 z), scale, offset, 0.
    std::array<std::array<float, 4>, 2> sun_k{};
    u32 sun_regions = 0;
    u32 sun_dir_frame = ~0u, sun_k_frame = ~0u, sun_mask_frame = ~0u, done_frame = ~0u;
    u32 report_frames = 0, sun_traced = 0, spot_maps = 0; // (per report)
    VideoCore::ImageId sun_mask{};
    VideoCore::HostImage sun_vis; // R8, the visibility (mode 0 -> mode 1)
    vk::UniqueImageView sun_vis_view;
    u32 sun_w = 0, sun_h = 0;
    vk::UniqueDescriptorSetLayout sun_desc_layout;
    vk::UniquePipelineLayout sun_layout;
    vk::UniquePipeline sun_pipeline;
    // The shadowed spot's map.
    u32 spot_map_frame = ~0u, spot_map_size = 0;
    VAddr spot_target = 0; // the game's map the light passes sample
    VideoCore::HostImage spot_map, spot_blur;
    vk::UniqueImageView spot_map_view, spot_blur_view;
    vk::UniqueDescriptorSetLayout spot_desc_layout;
    vk::UniquePipelineLayout spot_layout;
    vk::UniquePipeline spot_pipeline;
};

} // namespace Vulkan
