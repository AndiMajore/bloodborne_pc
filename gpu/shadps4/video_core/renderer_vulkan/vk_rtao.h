// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ray-traced ambient occlusion (BB_RTAO=1). The G-buffer draws' clip positions, captured
// by the object-motion feature, are expanded into a view-space triangle list, built into an
// acceleration structure every frame, and traced with ray queries at half resolution; the result
// replaces the game's own SSAO in its R8 mask (min of both). Design, settings, measurements and
// the debugging procedure: docs/rtao.md.

#pragma once

#include <array>
#include <chrono>
#include <string>
#include <unordered_map>
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
        u64 key;          // identity across frames (shader, buffers, ranges); the occurrence is added
        bool skinned;     // a bone palette: never cached
    };

    explicit RayTracedAO(const Instance& instance, Scheduler& scheduler,
                         VideoCore::TextureCache& texture_cache, Runtime& runtime,
                         CameraMotion& camera_motion, ObjectMotion& object_motion);
    ~RayTracedAO();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }
    void OnFrameStart();
    /// A cut (load, warp, upscaler reset): the next pass starts without temporal history.
    void InvalidateHistory() {
        history_valid = false;
    }
    /// <user>/rtao.txt (BB_RTAO_FILE): key=value lines (strength, radius, rays, history, debug,
    /// enabled) re-read within a second of a change.
    void PollSettings();
    void OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal);
    void AddGeometry(const Geometry& geometry);
    /// Every draw's first color target and fragment shader: the end of the game's AO upsample
    /// pass (pixel shader 0xc45c51cb into an R8 target of the scene size) is where the mask is
    /// rewritten. Before that shader is seen, any R8 scene-size target qualifies.
    void OnColorTarget(VideoCore::ImageId color, u64 ps_hash);
    static constexpr u64 AoUpsampleShader = 0xc45c51cbull;

private:
    struct GpuGeometry {
        u32 store, vertices, instances, first_vertex;
        u64 index_address;
        u32 index_count, index16;
        s32 base_vertex;
        u32 restart, triangle_offset, triangles;
        u32 out_base; // first output triangle in the triangle buffer
        u32 flags;    // GeoWorld: world space (a cache build); GeoProbe: measured for the cache
    };
    static_assert(sizeof(GpuGeometry) == 56);
    static constexpr u32 GeoWorld = 1, GeoProbe = 2;

    /// Per probed draw, written by rtao_probe.comp: world-space summary of its captured vertices.
    struct ProbeStats {
        u32 id, count;
        float centroid[3], min[3], max[3], first[3];
        u32 pad[2];
    };
    static_assert(sizeof(ProbeStats) == 64);

    /// A draw seen over several frames: static ones get their own world-space BLAS.
    struct CacheEntry {
        u64 base = 0;          // the mesh (all copies of it share this)
        u32 stable = 0;        // consecutive direct matching observations (stable draw order)
        u32 evictions = 0;     // static, then changed: three times and the draw is never cached
        bool have_stats = false;
        ProbeStats stats{};
        u32 last_seen = 0;     // frame
        vk::AccelerationStructureKHR blas{};
        VmaAllocation allocation{};
        vk::Buffer buffer{};
        vk::DeviceAddress address = 0;
        u64 bytes = 0;
        u32 triangles = 0;
        u32 ready_from = 0;    // frame from which the BLAS may be instanced
    };
    /// One draw of the frame before classification (dynamic, cached instance or promotion).
    struct Draw {
        GpuGeometry gpu;
        u64 key;  // base and occurrence
        u64 base; // the mesh: copies drawn several times a frame share it (the game sorts them by
                  // distance, so an occurrence is not one copy for good)
        u32 occurrence;
        bool skinned;
    };
    static u64 KeyOf(u64 base, u32 occurrence) {
        u64 key = (base ^ (u64(occurrence + 1) * 0x9E3779B97F4A7C15ull)) * 0xBF58476D1CE4E5B9ull;
        key ^= key >> 31;
        return key ? key : 1;
    }

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
    void ReadProbes(u32 ring);
    bool StatsMatch(const ProbeStats& a, const ProbeStats& b) const;
    void Evict(CacheEntry& entry);
    void TrimCache();

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    ObjectMotion& object_motion;

    bool enabled = false;
    u32 debug = 0; // 1: raw AO into the mask, 2: primary-ray test, 3: normal facing, 4: AO with t_min 0.1
    float strength = 1.0f;
    float power = 2.0f; // ao^power before the apply (the chosen look: 2 at radius 1.5; 1 = plain)
    float radius = 1.5f;
    float history = 0.5f; // 0.85 lagged visibly behind the camera (the Doll's cap)
    bool blur = true;     // 4x4 depth-aware tent in the apply (blur=0 for the plain upsample)
    bool replace_mask = false; // settings "replace=1": the game's AO (or the GTAO pack) is ignored
    bool build_fast_trace = false; // settings "build=1": BVH built for trace speed
    bool data_access = true;       // BB_RTAO_DATA_ACCESS=0: the per-frame BLAS without ALLOW_DATA_ACCESS
    bool depth_normals = true;   // settings "normals=0": the G-buffer shading normal instead
    float origin_offset = 0.0f;  // settings "offset=": metres along the normal on top of the ulp offset
    float snap_far = 30.0f;   // settings "snap=": the surface snap up to this depth (0 = off)
    float ao_far = 40.0f;     // settings "far=": no AO and no BVH beyond (0 = everything); 40 m saves ~2 ms
    float sigma = 1.5f;   // separable depth-aware blur of the half-res AO, in texels (0 = off)
    u32 rays = 6; // 8 cost 4.3 ms at 1.5M triangles (below 60 FPS)
    u32 max_triangles = 3000000; // 1.5M dropped ~10 draws per frame in the Hunter's Dream (blinking occluders)
    u32 stage = 5; // stop after this stage: 1 expand, 2 blas, 3 tlas, 4 trace, 5 apply
    u32 max_geometries = 8192;

    std::vector<GpuGeometry> geometries; // the frame's table (built in Record)
    std::vector<Draw> draws;             // the frame's eligible draws (AddGeometry)
    std::unordered_map<u64, u32> occurrences;
    std::unordered_map<u64, u32> max_copies; // per mesh: the most copies seen in one frame
    // Static cache
    bool cache_enabled = true;           // settings "cache=0": everything per frame
    u32 promote_after = 8;               // matching observations before a draw is cached
    float cache_max_size = 20.0f;        // settings "maxsize=": bigger draws stay per frame (far cut)
    u64 cache_budget = 1536ull << 20;    // bytes of cached BLASes (least recently seen go first)
    std::unordered_map<u64, CacheEntry> cache;
    u64 cache_bytes = 0;
    u32 cache_instances = 0, cache_instance_tris = 0, promotions = 0, evictions_total = 0;
    u32 probes_read = 0, probes_stale = 0, probes_matched = 0, probes_changed = 0, mismatch_prints = 0;
    u32 probes_reordered = 0;
    std::array<float, 3> camera_world{};
    static constexpr u32 ProbeSlots = 4;
    std::array<std::vector<u64>, ProbeSlots> probe_keys; // table index -> key (0: not probed)
    std::array<u32, ProbeSlots> probe_ids{};
    // Scheduler ticks of the last pass that used each host ring slot: a slot is reused (or read
    // back) only once the GPU has passed that tick; otherwise the pass skips the frame.
    std::array<u64, ProbeSlots> probe_ticks{};
    u64 readback_tick = 0, dump_tick = 0;
    u32 skipped_busy = 0;   // passes skipped because the slot was still in use (per report)
    bool ring_test = false; // settings "ringtest=1": every pass on slot 0 (exercises the skip)
    static constexpr u32 MaxInstances = 8192;
    static constexpr u32 PromoTriangles = 400000; // world-space triangles built per frame at most
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
    bool upsample_seen = false;   // the game's upsample shader was seen: the heuristic is off
    bool mask_by_heuristic = false;
    bool missing_pass_reported = false;
    bool done_this_frame = false;
    u32 frame = 0;
    u32 width = 0, height = 0;
    std::string settings_path;
    long long settings_mtime = -1;
    std::chrono::steady_clock::time_point last_poll{};
    bool active = true; // the file's "enabled" (the pass stays constructed)
    bool history_valid = false;
    u32 last_recorded_frame = 0;            // frame of the previous traced pass
    std::array<float, 3> last_camera_world{}; // its camera position (cuts: > 1 m between passes)
    u32 history_resets = 0;
    u32 ao_index = 0;

    // Pipelines
    vk::UniqueDescriptorSetLayout expand_desc_layout, trace_desc_layout, apply_desc_layout, blur_desc_layout,
        probe_desc_layout;
    vk::UniquePipelineLayout expand_layout, trace_layout, apply_layout, blur_layout, probe_layout;
    vk::UniquePipeline expand_pipeline, trace_pipeline, apply_pipeline, blur_pipeline, probe_pipeline;
    // Geometry and acceleration structures
    static constexpr u32 TableSlots = 3;
    std::array<u64, TableSlots> slot_ticks{};
    Buffer table_buffer;      // host: GpuGeometry[max_geometries] x TableSlots
    Buffer readback_buffer;   // host: the first ReadbackTriangles triangles of a frame
    static constexpr u32 ReadbackTriangles = 4096;
    Buffer triangle_buffer;   // device: float3 x 3 x (max_triangles + PromoTriangles)
    Buffer probe_buffer;      // host (read back): ProbeStats[max_geometries] x ProbeSlots
    Buffer promo_scratch;     // device: scratch for the frame's cache builds
    Buffer constants_buffer;  // host: the trace's per-frame constants x TableSlots
    static constexpr vk::DeviceSize ConstantsStride = 256; // >= minUniformBufferOffsetAlignment
    Buffer blas_buffer, blas_scratch, tlas_buffer, tlas_scratch;
    Buffer instance_buffer;   // host: AccelerationStructureInstanceKHR[MaxInstances] x TableSlots
    vk::DeviceAddress blas_address = 0;
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
