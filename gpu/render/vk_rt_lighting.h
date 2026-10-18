// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (experimental, ticket 65): the RT lighting experiments that hook the game's draws, behind
// one object the rasterizer talks to. Each effect is a class depending only on the RTAO's scene
// (FrameTlas): the light list (RtLights, 59), the shadows (RtShadows, 64), the transparents' light
// and the fire lights (RtEmission, 63). Settings in <user>/rtlighting.txt (rt_lighting_settings.h),
// everything off by default. The Wants*() flags gate every hook: with no effect on (or the RTAO
// off) the rasterizer does what it did without them.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include "common/types.h"
#include "render/rt_lighting_settings.h"
#include "render/vk_rt_emission.h"
#include "render/vk_rt_lights.h"
#include "render/vk_rt_shadows.h"
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

class RtLighting {
public:
    using Emission = RtEmission::Frame;

    RtLighting(const Instance& instance, Scheduler& scheduler, VideoCore::TextureCache& texture_cache,
               Runtime& runtime, CameraMotion& camera_motion, const RayTracedAO& rtao);
    ~RtLighting();

    void OnFrameStart();
    void OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal);

    /// Fragment-shader constant buffers wanted (OnShaderConstants).
    [[nodiscard]] bool WantsShaderConstants() const noexcept {
        return wants_constants;
    }
    void OnShaderConstants(u64 hash, VAddr address, u64 size) {
        shadows->OnShaderConstants(hash, address, size);
    }
    /// The light passes' draws wanted: their light buffers (OnLightBuffer) and textures (the spot map).
    [[nodiscard]] bool WantsLightDraws() const noexcept {
        return wants_lights;
    }
    void OnLightBuffer(VAddr address, u64 size, u64 ps_hash) {
        lights->OnLightBuffer(address, size, ps_hash);
    }
    void OnSpotMapImage(VAddr address, vk::Format format) {
        shadows->OnSpotMapImage(address, format);
    }
    [[nodiscard]] vk::ImageView SpotMapView(VAddr address) const noexcept {
        return settings.spot_shadow ? shadows->SpotMapView(address) : vk::ImageView{};
    }
    /// Every draw (before it is recorded) wanted: OnDraw.
    [[nodiscard]] bool WantsDraws() const noexcept {
        return wants_draws || emission->DrawsWanted();
    }
    void OnDraw(VideoCore::ImageId color, u64 ps_hash, bool scene_depth, bool blended, bool depth_write, u32 indices,
                u32 instances);
    /// The emission's two scene points (the first transparent draw; the first full-screen pass).
    void OnOpaqueSceneLit(VideoCore::ImageId color) {
        if (wants_emission) {
            emission->OnOpaqueSceneLit(color);
        }
    }
    void OnSceneComposite(VideoCore::ImageId color) {
        if (wants_emission) {
            emission->OnSceneComposite(color);
        }
    }
    /// The emission image and the fire lights of the frame before (SSGI, the probes, the reflections).
    [[nodiscard]] Emission FrameEmission() const noexcept {
        return wants_emission ? emission->FrameEmission() : Emission{};
    }
    [[nodiscard]] const RtLights& Lights() const noexcept {
        return *lights;
    }

private:
    void PollSettings();
    void UpdateWants();

    const RayTracedAO& rtao;
    RtLightingSettings::Settings settings;
    std::unique_ptr<RtLights> lights;
    std::unique_ptr<RtShadows> shadows;
    std::unique_ptr<RtEmission> emission;
    bool wants_constants = false, wants_lights = false, wants_draws = false, wants_emission = false;
    bool lights_collected = false; // the frame that ended collected the light list
    float dump_done = 0.0f;        // the last lights_dump value served
    u32 frame = 0;
    std::string settings_path;
    long long settings_mtime = -1;
    std::chrono::steady_clock::time_point last_poll{};
};

} // namespace Vulkan
