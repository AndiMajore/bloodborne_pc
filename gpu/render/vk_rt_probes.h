// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ray-traced probe GI (BB_RT_PROBES=1, experimental; ticket 62 6b, a first DDGI-style
// version). A grid of irradiance probes around the camera (16x8x16, `spacing` metres apart,
// scrolling with it) traces `rays` rays per probe each frame against the ray-traced AO's scene;
// hits are shaded from the lit screen or the RTAO's surface cache, misses with a sky term; the
// radiance is kept as L1 spherical harmonics with a hysteresis. Each pixel blends its eight
// probes and adds albedo x irradiance to the lit opaque scene, as SSGI does (use one or the other).
// No probe depth maps yet: probes inside geometry are switched off, light can still leak through
// thin walls. <user>/rtprobes.txt: enabled, strength, spacing, rays, hysteresis, sky, bias, debug.

#pragma once

#include <array>
#include <chrono>
#include <string>
#include "common/types.h"
#include "textures/types.h"
#include "vulkan/vk_common.h"
#include <vk_mem_alloc.h>

namespace VideoCore {
class TextureCache;
struct Image;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;
class CameraMotion;
class RayTracedAO;

class RtProbes {
public:
    explicit RtProbes(const Instance& instance, Scheduler& scheduler,
                      VideoCore::TextureCache& texture_cache, Runtime& runtime,
                      CameraMotion& camera_motion, RayTracedAO& rtao);
    ~RtProbes();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }
    void OnFrameStart();
    /// The G-buffer pass: depth, normal (c0) and albedo (c2).
    void OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal, VideoCore::ImageId albedo);
    /// The lit opaque scene of this frame: probes updated, then added to it.
    void OnOpaqueScene(VideoCore::ImageId color);

    static constexpr std::array<s32, 3> Dims{16, 8, 16};

private:
    void PollSettings();
    vk::ImageView CachedView(const VideoCore::Image& image, vk::Format format,
                             vk::ImageAspectFlags aspect);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    RayTracedAO& rtao;

    bool enabled = false;
    bool active = true;
    float strength = 4.0f; // ~ SSGI's brightness at the Doll (its own strength 8 + damping)
    float spacing = 3.0f;
    u32 rays = 64;
    float hysteresis = 0.95f;
    float sky = 0.0f;
    float bias = 0.3f;
    u32 debug = 0;
    bool reset = true; // the probes are cleared before the next update (start, spacing change)
    std::string settings_path;
    long long settings_mtime = -1;
    std::chrono::steady_clock::time_point last_poll{};

    VideoCore::ImageId depth_id{}, normal_id{}, albedo_id{};
    bool done_this_frame = false;
    u32 frame = 0;
    u32 passes = 0, skipped = 0;

    vk::Buffer probe_buffer{};
    VmaAllocation probe_allocation{};
    vk::UniqueDescriptorSetLayout update_desc_layout, apply_desc_layout;
    vk::UniquePipelineLayout update_layout, apply_layout;
    vk::UniquePipeline update_pipeline, apply_pipeline;

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
