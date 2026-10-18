// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: screen-space diffuse GI (BB_SSGI=1): one bounce gathered from the lit opaque scene at
// half resolution and added to the scene colour before the transparents, with a temporal
// history on the camera motion vectors. Design: .scratch/debug-session/ssgi/DESIGN.md (local).

#pragma once

#include <array>
#include <chrono>
#include <string>
#include "common/types.h"
#include "vulkan/vk_common.h"
#include "textures/image.h"
#include "textures/types.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;
class CameraMotion;
class SceneTargets;
class RayTracedAO;
class RtLighting;

class ScreenSpaceGI {
public:
    explicit ScreenSpaceGI(const Instance& instance, Scheduler& scheduler,
                           VideoCore::TextureCache& texture_cache, Runtime& runtime,
                           CameraMotion& camera_motion, SceneTargets& scene_targets);
    ~ScreenSpaceGI();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }
    /// The G-buffer pass of the frame: its depth, normal (c0) and albedo (c2) targets.
    void OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal,
                       VideoCore::ImageId albedo);
    /// The opaque scene is lit (first blended draw or the fog composite): run once per frame.
    void OnOpaqueScene(VideoCore::ImageId color);
    /// Start of a frame.
    void OnFrameStart();
    /// bbport (experimental): the ray-traced AO whose scene the RT gather traces (rt=1).
    void SetRayTracedAO(const RayTracedAO* rtao_) {
        rtao = rtao_;
    }
    /// bbport (experimental): the transparents' light and the fire lights (rtlighting.txt).
    void SetLighting(const RtLighting* lighting_) {
        lighting = lighting_;
    }
    /// <user>/ssgi.txt (BB_SSGI_FILE): "key=value" lines (strength, radius, history, directions,
    /// steps, debug, enabled) re-read within a second of a change: tuning while the game runs.
    void PollSettings();

private:
    bool EnsureResources(u32 width, u32 height);
    void EnsureDenoiseResources();
    vk::ImageView CachedView(const VideoCore::Image& image, vk::Format format,
                             vk::ImageAspectFlags aspect);
    void Record(VideoCore::ImageId color);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    SceneTargets& scene_targets;

    bool enabled = false;
    u32 debug = 0;      // 1: the bounce alone; 2 (rt): rays seen (green) / hidden (red)
    bool rt = false;    // rt=1: the gather traces rays against the RTAO's scene (experimental)
    u32 rays = 2;       // per half-res pixel (rt)
    float rt_max = 2.0f;  // rt: a ray's radiance capped to this (lamps and candles as fireflies)
    float rt_clip = 1.5f; // rt: history clipped to this frame's 3x3 mean +- rt_clip sigma
    // rt_denoise=1: the colour denoiser instead of the gather's own history (experimental, off):
    // temporal accumulation on the motion vectors, then edge-avoiding a-trous passes.
    bool rt_denoise = false;
    float denoise_frames = 16.0f; // the history's length cap
    float denoise_clip = 2.0f;    // history clipped to the raw 3x3 mean +- that many sigma (0: off)
    u32 denoise_passes = 3;       // a-trous passes (steps 1, 2, 4, ...)
    float denoise_luma = 4.0f, denoise_plane = 0.02f, denoise_normal = 32.0f;
    const RayTracedAO* rtao = nullptr;
    const RtLighting* lighting = nullptr;
    float strength = 1.0f;
    float radius = 2.0f;
    u32 directions = 4, steps = 6;
    float history = 0.9f;
    float adaptive = 4.0f;

    VideoCore::ImageId depth_id{}, normal_id{}, albedo_id{};
    bool done_this_frame = false;
    u32 frame = 0;
    u32 width = 0, height = 0;
    std::string settings_path;
    long long settings_mtime = -1;
    std::chrono::steady_clock::time_point last_poll{};
    bool history_valid = false;
    u32 gi_index = 0;

    vk::UniqueDescriptorSetLayout gather_desc_layout, apply_desc_layout;
    vk::UniquePipelineLayout gather_layout, apply_layout;
    vk::UniquePipeline gather_pipeline, apply_pipeline;
    vk::UniqueDescriptorSetLayout gather_rt_desc_layout; // + binding 6, the TLAS
    vk::UniquePipelineLayout gather_rt_layout;
    vk::UniquePipeline gather_rt_pipeline;               // null without ray queries
    u32 rt_passes = 0, rt_fallbacks = 0;                 // (per report)
    vk::UniqueDescriptorSetLayout temporal_desc_layout, atrous_desc_layout;
    vk::UniquePipelineLayout temporal_layout, atrous_layout;
    vk::UniquePipeline temporal_pipeline, atrous_pipeline;
    // The denoiser's images (made at its first use): accumulated rgb + frames and moments + view z
    // + normal ping-pong between frames; two scratch images for the a-trous passes.
    std::array<VideoCore::HostImage, 2> acc_images, mom_images, atrous_images;
    std::array<vk::UniqueImageView, 2> acc_views, mom_views, atrous_views;
    u32 denoise_width = 0, denoise_height = 0;
    bool denoise_valid = false;
    u32 denoise_index = 0;
    std::array<VideoCore::HostImage, 2> gi_images;
    std::array<vk::UniqueImageView, 2> gi_views;
    VideoCore::HostImage motion_image;
    vk::UniqueImageView motion_view;

    struct ViewEntry {
        vk::Image image{};
        u64 uid = 0;
        vk::Format format{};
        vk::ImageAspectFlags aspect{};
        vk::ImageView view{};
        u64 last_use = 0;
    };
    std::array<ViewEntry, 6> view_cache{};
    u64 view_uses = 0;
};

} // namespace Vulkan
