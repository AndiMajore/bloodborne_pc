// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ray-traced ambient occlusion (BB_RTAO=1). The G-buffer draws' clip positions, captured
// by the object-motion feature, are expanded into a view-space triangle list, built into an
// acceleration structure every frame, and traced with ray queries at half resolution; the result
// replaces the game's own SSAO in its R8 mask (min of both). Plan: .scratch/debug-session/rt/PLAN.md.

#pragma once

#include <array>
#include <chrono>
#include <string>
#include <vector>
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
class ObjectMotion;

class RayTracedAO {
public:
    /// One captured indexed draw of the frame (the object-motion store of its positions).
    struct Geometry {
        u32 store;
        u32 vertices;
        u32 instances;
        u32 first_vertex;
        u64 index_address;
        u32 index_count;
        bool index16;
        s32 base_vertex;
        bool restart;
        bool triangle_list;
        bool non_indexed; // index_address 0, vertex i of the draw is index i
        bool alpha_test;  // the pixel shader discards (foliage, vines): solid quads in the BVH
        bool depth_write; // off: decals and overlays hugging other surfaces
    };

    explicit RayTracedAO(const Instance& instance, Scheduler& scheduler,
                         VideoCore::TextureCache& texture_cache, Runtime& runtime,
                         CameraMotion& camera_motion, ObjectMotion& object_motion);
    ~RayTracedAO();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }
    void OnFrameStart();
    /// <user>/rtao.txt (BB_RTAO_FILE): key=value lines (strength, radius, rays, history, debug,
    /// enabled) re-read within a second of a change.
    void PollSettings();
    void OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal);
    void AddGeometry(const Geometry& geometry);
    /// Every draw's first color target: the end of the game's AO upsample pass (an R8 target of
    /// the scene size) is where the mask is rewritten.
    void OnColorTarget(VideoCore::ImageId color);

private:
    struct GpuGeometry {
        u32 store, vertices, instances, first_vertex;
        u64 index_address;
        u32 index_count, index16;
        s32 base_vertex;
        u32 restart, triangle_offset, triangles, pad0, pad1;
    };
    static_assert(sizeof(GpuGeometry) == 56);

    struct Buffer {
        vk::Buffer buffer{};
        VmaAllocation allocation{};
        vk::DeviceAddress address{};
        void* mapped{};
        vk::DeviceSize size{};
    };
    bool CreateBuffer(Buffer& buffer, vk::DeviceSize size, vk::BufferUsageFlags usage, bool host,
                      vk::DeviceSize alignment = 0);
    void DestroyBuffer(Buffer& buffer);
    bool EnsureResources(u32 width, u32 height);
    bool CreateAccelerationStructures();
    vk::ImageView CachedView(const VideoCore::Image& image, vk::Format format,
                             vk::ImageAspectFlags aspect);
    void Record(VideoCore::ImageId mask);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    ObjectMotion& object_motion;

    bool enabled = false;
    u32 debug = 0; // 1: raw AO into the mask, 2: primary-ray test, 3: normal facing, 4: AO with t_min 0.1
    float strength = 1.0f;
    float power = 2.0f; // ao^power before the apply (Andreas: 2 at radius 1.5; 1 = faithful)
    float radius = 1.5f;
    float history = 0.5f; // 0.85 lagged visibly behind the camera (the Doll's cap)
    bool blur = true;     // 4x4 depth-aware tent in the apply (blur=0 for the plain upsample)
    bool replace_mask = false; // settings "replace=1": the game's AO (or the GTAO pack) is ignored
    float sigma = 1.5f;   // separable depth-aware blur of the half-res AO, in texels (0 = off)
    u32 rays = 8;
    u32 max_triangles = 1500000;
    u32 stage = 5; // stop after this stage: 1 expand, 2 blas, 3 tlas, 4 trace, 5 apply
    u32 max_geometries = 8192;

    std::vector<GpuGeometry> geometries;
    u32 frame_triangles = 0, dropped_triangles = 0, dropped_draws = 0;
    u32 recorded = 0;         // passes recorded since the last report
    u32 skipped_topology = 0, skipped_no_address = 0, skipped_alpha = 0, skipped_nowrite = 0;
    bool include_alpha = false;   // settings "alpha=1": keep alpha-tested draws
    bool include_nowrite = false; // settings "decals=1": keep draws without depth write
    bool readback_pending = false;
    u32 readback_frame = 0;
    u32 readback_triangles = 0;
    Buffer dump_buffer;       // host: the half-res AO image (rg16f), "dump=" in the settings
    bool dump_pending = false;
    u32 dump_frame = 0;
    float dump_request = 0.0f, dump_done = 0.0f;
    u32 recorded_total = 0;
    VideoCore::ImageId depth_id{}, normal_id{};
    VideoCore::ImageId last_color{}, mask_candidate{};
    bool done_this_frame = false;
    u32 frame = 0;
    u32 width = 0, height = 0;
    std::string settings_path;
    long long settings_mtime = -1;
    std::chrono::steady_clock::time_point last_poll{};
    bool active = true; // the file's "enabled" (the pass stays constructed)
    bool history_valid = false;
    u32 ao_index = 0;

    // Pipelines
    vk::UniqueDescriptorSetLayout expand_desc_layout, trace_desc_layout, apply_desc_layout, blur_desc_layout;
    vk::UniquePipelineLayout expand_layout, trace_layout, apply_layout, blur_layout;
    vk::UniquePipeline expand_pipeline, trace_pipeline, apply_pipeline, blur_pipeline;
    // Geometry and acceleration structures
    static constexpr u32 TableSlots = 3;
    Buffer table_buffer;      // host: GpuGeometry[max_geometries] x TableSlots
    Buffer readback_buffer;   // host: the first ReadbackTriangles triangles of a frame
    static constexpr u32 ReadbackTriangles = 4096;
    Buffer triangle_buffer;   // device: float3 x 3 x max_triangles
    Buffer blas_buffer, blas_scratch, tlas_buffer, tlas_scratch, instance_buffer;
    vk::AccelerationStructureKHR blas{}, tlas{};
    vk::DeviceSize scratch_alignment = 256;
    // Images
    std::array<VideoCore::UniqueImage, 2> ao_images;
    std::array<vk::UniqueImageView, 2> ao_views;
    VideoCore::UniqueImage motion_image;
    vk::UniqueImageView motion_view;
    VideoCore::UniqueImage blur_image; // half res, the horizontal blur pass
    vk::UniqueImageView blur_view;
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
