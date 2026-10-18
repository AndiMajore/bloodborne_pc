// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ray-traced reflections, phase 1 (BB_RT_REFLECT=1, experimental; ticket 60). After the
// game's SSR pass (ps 6c2323c2, a half-resolution RGBA16F target) a mirror ray per texel is traced
// against the ray-traced AO's scene; a hit the camera sees is shaded with last frame's lit image
// at its reprojected position and written into the SSR target, which the game's lighting pass
// weights by the surface's own specular as it does SSR's result. mode 1 fills only SSR's misses,
// mode 2 replaces SSR wherever the ray gives a value. Needs the RTAO (its TLAS).
// replace_ssr=1 (default): the game's SSR draw is left out (~1.5 ms) and the trace writes its whole target
// (as mode 2, nothing where the ray finds no visible hit).
// <user>/rtreflect.txt: enabled, mode, far, strength, debug, replace_ssr (re-read while the game runs).

#pragma once

#include <array>
#include <chrono>
#include <string>
#include "common/types.h"
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
class RtLighting;

class RtReflections {
public:
    explicit RtReflections(const Instance& instance, Scheduler& scheduler,
                           VideoCore::TextureCache& texture_cache, Runtime& runtime,
                           CameraMotion& camera_motion, const RayTracedAO& rtao, const RtLighting& lighting);
    ~RtReflections();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }
    void OnFrameStart();
    /// The G-buffer pass: its depth and normal (c0) targets.
    void OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal, VideoCore::ImageId material);
    /// Every draw's first colour target and fragment shader: the end of the SSR pass is where
    /// the reflections are traced.
    void OnColorTarget(VideoCore::ImageId color, u64 ps_hash);
    /// The lit opaque scene of this frame (before the transparents): kept as next frame's radiance.
    void OnOpaqueScene(VideoCore::ImageId color);
    static constexpr u64 SsrShader = 0x6c2323c2ull;
    /// The game's SSR draw (before it is recorded): true when it is to be left out this frame
    /// (replace_ssr=1 and the trace will run right after it, writing the whole target).
    [[nodiscard]] bool SkipGameSsr();

private:
    void PollSettings();
    void Record(VideoCore::ImageId ssr);
    bool EnsureCopy(u32 w, u32 h);
    vk::ImageView CachedView(const VideoCore::Image& image, vk::Format format,
                             vk::ImageAspectFlags aspect);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    const RayTracedAO& rtao;
    const RtLighting& lighting; // the transparents' light (FrameEmission)

    bool enabled = false;
    bool active = true;     // enabled=0 in the file: off live
    u32 mode = 1;
    bool replace_ssr = true;
    float soften = 1.0f; // the rays' cone x the material's lobe (G-buffer c1.x: gloss, power 2^(13 gloss))
    bool replaced_this_frame = false;
    u32 replaced = 0; // (per report)
    u32 debug = 0;
    float far = 40.0f;
    float strength = 1.0f;
    std::string settings_path;
    long long settings_mtime = -1;
    std::chrono::steady_clock::time_point last_poll{};

    VideoCore::ImageId depth_id{}, normal_id{}, material_id{}; // material: the G-buffer's c1 (x: gloss)
    VideoCore::ImageId last_color{}, ssr_target{};
    bool done_this_frame = false, copied_this_frame = false;
    bool copy_valid = false; // the copy holds a frame (the previous one when traced)
    u32 frame = 0;
    u32 passes = 0, skipped = 0; // (per report)

    VideoCore::HostImage copy_image;
    vk::UniqueImageView copy_view;
    u32 copy_w = 0, copy_h = 0;

    vk::UniqueDescriptorSetLayout trace_desc_layout, copy_desc_layout;
    vk::UniquePipelineLayout trace_layout, copy_layout;
    vk::UniquePipeline trace_pipeline, copy_pipeline;

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
