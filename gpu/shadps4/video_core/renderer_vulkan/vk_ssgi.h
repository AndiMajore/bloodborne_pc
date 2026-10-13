// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: screen-space diffuse GI (BB_SSGI=1): one bounce gathered from the lit opaque scene at
// half resolution and added to the scene colour before the transparents, with a temporal
// history on the camera motion vectors. Design: .scratch/debug-session/ssgi/DESIGN.md (local).

#pragma once

#include <array>
#include <chrono>
#include <string>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;
class CameraMotion;
class SceneTargets;

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
    /// <user>/ssgi.txt (BB_SSGI_FILE): "key=value" lines (strength, radius, history, directions,
    /// steps, debug, enabled) re-read within a second of a change: tuning while the game runs.
    void PollSettings();

private:
    bool EnsureResources(u32 width, u32 height);
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
    bool debug = false;
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
    std::array<VideoCore::UniqueImage, 2> gi_images;
    std::array<vk::UniqueImageView, 2> gi_views;
    VideoCore::UniqueImage motion_image;
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
