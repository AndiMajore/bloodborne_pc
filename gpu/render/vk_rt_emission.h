// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (experimental, ticket 63; rtlighting.txt emission=, fire_light=): the transparents' light
// for the RT lighting effects. Fires, sparks and glows are blended over the scene after the
// opaque pass, so SSGI, the probes and the reflections, which read the lit opaque scene, never saw
// them as light. The opaque scene (after SSGI and the probes added their bounce) is kept at a
// quarter of its size at the first transparent draw; at the end of the transparents
// (rt_emission_trigger.h) the scene minus that is the emission image, read the frame after at a
// hit's screen position (rtao_emission.comp). From it the fires as lights: per 64x64 tile of the
// emission the brightest texel at the nearest depth under the flame, matched with last frame's by
// their place in the world (rtao_fire_lights.comp); the RT SSGI lights from them with shadow rays.

#pragma once

#include <array>
#include "common/types.h"
#include "render/rt_emission_trigger.h"
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

class RtEmission {
public:
    RtEmission(const Instance& instance, Scheduler& scheduler, VideoCore::TextureCache& texture_cache,
               Runtime& runtime, CameraMotion& camera_motion, const RayTracedAO& rtao, const RtLights& lights,
               const RtLightingSettings::Settings& settings);
    ~RtEmission();

    /// The emission image and the fire lights of the frame before, for the effects; empty when
    /// off or not made.
    struct Frame {
        vk::ImageView view{}; // RGBA16F quarter size, eGeneral; empty: none (bind something else, strength 0)
        float strength = 0.0f;
        // The fires as lights (RGBA32F, eGeneral; rtao_fire_lights.comp): the gather's list in rows 3
        // (world position, w 1 a spark, 2 a base) and 4 (power, scaled), its length in row 5. A base
        // is an upright cylinder (base_size, base_height), a spark a sphere (spark_size).
        vk::ImageView lights{};
        float light_strength = 0.0f;
        float light_range = 0.0f;
        float light_wrap = 0.0f;
        float base_size = 0.0f;
        float base_height = 0.0f;
        float spark_size = 0.0f;
        u32 light_rays = 1; // shadow rays per pixel
        float large_base_size = 0.0f, large_base_height = 0.0f, large_spark_size = 0.0f; // the pyres
    };
    [[nodiscard]] Frame FrameEmission() const noexcept;

    void OnFrameStart(u32 frame);
    void OnGBufferPass(VideoCore::ImageId depth);
    /// The first transparent draw: the opaque scene (with the bounce of SSGI and the probes) is kept.
    void OnOpaqueSceneLit(VideoCore::ImageId color);
    /// Draws after the opaque scene wait for the end of the transparents.
    [[nodiscard]] bool DrawsWanted() const noexcept {
        return trigger.Pending();
    }
    void OnDrawAfterOpaque(VideoCore::ImageId color, bool scene_depth);

private:
    bool Ensure(u32 w, u32 h);
    void Record(VideoCore::ImageId color, u32 mode);
    void RecordFireLights();
    void RecordFireDebug(VideoCore::ImageId color);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    const RayTracedAO& rtao;
    const RtLights& game_lights; // the game's light list (59): the fires' bases anchored to its fire lights
    const RtLightingSettings::Settings& settings;
    RtViewCache views;
    u32 frame = 0;
    VideoCore::ImageId depth_id{};

    // The opaque scene and the transparents' light, quarter size.
    VideoCore::HostImage opaque_image, emit_image;
    vk::UniqueImageView opaque_view, emit_view;
    u32 emit_w = 0, emit_h = 0;
    u32 snapshot_frame = ~0u, emit_frame = ~0u - 1;
    bool emit_written = false;
    VideoCore::ImageId scene{};       // the scene colour the opaque base was taken from
    EmissionRules::Trigger trigger;
    u32 taken = 0, missed = 0, report_frames = 0; // per 600 frames: emission made / not
    vk::UniqueDescriptorSetLayout emit_desc_layout;
    vk::UniquePipelineLayout emit_layout;
    vk::UniquePipeline emit_pipeline;
    // The fires as lights: one per 64x64 tile. Layers: 0 this frame's per tile; 1 and 2 the lights,
    // matched with the frame before's (they take turns: lights_out is this frame's).
    VideoCore::HostImage lights_image;
    std::array<vk::UniqueImageView, 3> lights_views;
    u32 lights_out = 1;
    u32 lights_count = 0, tiles_x = 0, tiles_y = 0;
    u32 lights_frame = ~0u - 1;
    bool lights_written = false;
    u32 anchor_reports = 0;
    vk::UniqueDescriptorSetLayout lights_desc_layout;
    vk::UniquePipelineLayout lights_layout;
    vk::UniquePipeline lights_pipeline;
    // fire_debug=1: the lights' cylinders drawn into the scene.
    vk::UniqueDescriptorSetLayout debug_desc_layout;
    vk::UniquePipelineLayout debug_layout;
    vk::UniquePipeline debug_pipeline;
};

} // namespace Vulkan
