// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ray-traced ambient occlusion (BB_RTAO=1). The G-buffer draws' clip positions, captured
// by the object-motion feature, are expanded into a view-space triangle list and built into an
// acceleration structure every frame; draws that do not move are kept in world space in one static
// BLAS, rebuilt when the camera moved. Traced at half resolution; the result replaces the game's
// own SSAO in its R8 mask (min of both). Design, settings, measurements and
// the debugging procedure: docs/rtao.md.

#pragma once

#include <array>
#include <chrono>
#include <future>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/types.h"
#include "video_core/renderer_vulkan/rtao_geometry.h"

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
        // Small index ranges (the buffer cache streams them through a ring that is reused once the
        // draw's tick is done): the guest address, copied at draw time into the RTAO's own ring
        // through the runtime's backing view (Core::MemoryManager::ReadBacking; 0: none).
        VAddr index_guest = 0;
        u32 index_bytes = 0;
    };
    /// True once per frame, after the pass was recorded (the caller releases what it held).
    bool TakeRecorded() {
        const bool value = recorded_this_frame;
        recorded_this_frame = false;
        return value;
    }

    explicit RayTracedAO(const Instance& instance, Scheduler& scheduler,
                         VideoCore::TextureCache& texture_cache, Runtime& runtime,
                         CameraMotion& camera_motion, ObjectMotion& object_motion);
    ~RayTracedAO();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }
    /// Running now: not switched off live (enabled=0) and its buffers made. The capture of
    /// positions follows this, not Enabled().
    [[nodiscard]] bool Active() const noexcept {
        return enabled && resources_ready && settings.active;
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
    /// rewritten. Nothing else is ever written.
    void OnColorTarget(VideoCore::ImageId color, u64 ps_hash);
    static constexpr u64 AoUpsampleShader = 0xc45c51cbull;

private:
    using GpuGeometry = Rtao::GpuGeometry;
    using ProbeStats = Rtao::ProbeStats;
    static constexpr u32 GeoWorld = Rtao::GeoWorld, GeoProbe = Rtao::GeoProbe;

    /// A draw seen over several frames: static ones get their own world-space BLAS.
    /// A cached draw's world-space triangles in the static pool: the whole draw, or a piece of a
    /// big one.
    struct CachePart {
        u32 pool_offset = 0;   // triangles
        u32 triangles = 0;
        Rtao::Box box;         // world space: in the static BLAS only within reach
    };
    struct CacheEntry : Rtao::CacheState {
        u32 last_seen = 0;     // frame
        std::vector<CachePart> parts; // empty: not cached
        bool chunked = false;  // parts are pieces of a mesh bigger than maxsize
        bool split_pending = false; // its triangles are being read back and split
        bool in_static = false;     // the static BLAS holds all its parts within reach
        bool static_skip = false;   // left out of the last static build (its triangle budget)
        u32 triangles = 0;
        u32 ready_from = 0;    // frame from which it may join the static BLAS
    };
    /// Big meshes (chunk > 0) are cached in pieces: their world-space triangles are copied to the
    /// host once, split into compact pieces on a worker thread (Rtao::SplitTriangles), written to
    /// a staging buffer, and the pieces are built from there. One batch at a time.
    struct SplitMesh {
        u64 key;
        Rtao::ProbeStats stats;   // the draw's summary when it was read: built only if unchanged
        u32 offset, triangles;    // in the fetched batch (triangles)
        u32 staging_offset = 0;   // in the staging buffer (triangles), after the split
        std::vector<Rtao::ChunkGroup> groups;
        bool done = false;
    };
    enum class SplitState { Idle, Fetching, Splitting, Building };
    void UpdateSplit();       // advance the batch: fetched -> worker -> pieces to build
    SplitState split_state = SplitState::Idle;
    std::vector<SplitMesh> split_meshes;
    u64 split_tick = 0;       // Fetching: the copy's tick; Building: the last build's
    std::future<std::pair<std::vector<SplitMesh>, float>> split_result; // the pieces, worker ms
    /// One draw of the frame before classification (dynamic, cached instance or promotion).
    struct Draw {
        GpuGeometry gpu;
        u64 key;  // base and occurrence
        u64 base; // the mesh: copies drawn several times a frame share it (the game sorts them by
                  // distance, so an occurrence is not one copy for good)
        u32 occurrence;
        bool skinned;
    };

    struct Buffer {
        vk::Buffer buffer{};
        VmaAllocation allocation{};
        vk::DeviceAddress address{};
        void* mapped{};
        vk::DeviceSize size{};
    };
    bool CreateBuffer(Buffer& buffer, vk::DeviceSize size, vk::BufferUsageFlags usage, bool host,
                      vk::DeviceSize alignment = 0, bool quiet = false, bool readback = false);
    void RebuildStatic(float reach); // choose the static BLAS's parts; built in this pass
    /// The static BLAS's build flags (staticbuild=; the buffers are sized for all of them).
    static vk::BuildAccelerationStructureFlagsKHR StaticFlags(u32 mode) {
        using F = vk::BuildAccelerationStructureFlagBitsKHR;
        return mode == 0 ? vk::BuildAccelerationStructureFlagsKHR{F::ePreferFastTrace}
               : mode == 1 ? vk::BuildAccelerationStructureFlagsKHR{F::ePreferFastBuild}
                           : F::ePreferFastBuild | F::eAllowDataAccess;
    }
    void DestroyBuffer(Buffer& buffer);
    bool EnsureResources(u32 width, u32 height);
    bool CreateAccelerationStructures();
    void CreateTracePipeline(); // the ray-tracing-pipeline variant (pipeline=1)
    vk::ImageView CachedView(const VideoCore::Image& image, vk::Format format,
                             vk::ImageAspectFlags aspect);
    void Record(VideoCore::ImageId mask);
    void ReadProbes(u32 ring);
    void WriteDump();      // vk_rtao_diagnostics.cpp
    void ReportReadback(); // vk_rtao_diagnostics.cpp
    void Evict(CacheEntry& entry);
    void TrimCache(); // forget draws not seen for a minute

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    ObjectMotion& object_motion;

    bool enabled = false;
    bool resources_ready = false; // the buffers and acceleration structures (made in the world)
    Rtao::Settings settings; // environment at start, <user>/rtao.txt while the game runs
    bool fast_trace_allowed = false;
    bool data_access = true;       // BB_RTAO_DATA_ACCESS=0: the per-frame BLAS without ALLOW_DATA_ACCESS
    // The per-frame BLAS's budget. With the static cache it holds the characters (~0.8M at the
    // Doll); without it (cache=0, chunk=0) 1.5M dropped ~10 draws per frame in the Hunter's Dream
    // (BB_RTAO_MAX_TRIANGLES=3000000 then).
    u32 max_triangles = 1500000;
    u32 max_geometries = 8192;

    std::vector<GpuGeometry> geometries; // the frame's table (built in Record)
    std::vector<Draw> draws;             // the frame's eligible draws (AddGeometry)
    std::unordered_map<u64, u32> occurrences;
    std::unordered_map<u64, u32> max_copies; // per mesh: the most copies seen in one frame
    // Static cache: cached draws' world-space triangles in a pool, and one BLAS over the pool's
    // triangles within reach of the camera (static_*), rebuilt when the camera moved or the set
    // changed. The TLAS holds two instances: the per-frame BLAS (view space) and the static one.
    u64 static_bytes = 0;                 // the RTAO's fixed buffers (footprint report)
    std::unordered_map<u64, CacheEntry> cache;
    Rtao::RangeAllocator pool;            // triangles of static_pool
    static constexpr u32 PoolTriangles = 3000000;
    static constexpr u32 StaticTriangles = 2000000;  // the static BLAS's budget
    static constexpr u32 StaticSeenFrames = 120;     // draws not seen for longer leave it
    Buffer static_pool;                   // device: float3 x 3 x PoolTriangles
    Buffer static_blas_buffer;
    vk::AccelerationStructureKHR static_blas{};
    vk::DeviceAddress static_blas_address = 0;
    std::vector<std::array<u32, 2>> static_build; // (pool offset, triangles) per geometry, this pass
    bool static_valid = false;            // built (instanced from the pass that builds it)
    bool static_dirty = false;            // a draw in it was evicted: rebuild now
    std::array<float, 3> static_anchor{}; // the camera when it was built
    u32 static_frame = 0;
    u32 static_triangles = 0, static_geometries = 0, static_rebuilds = 0; // (rebuilds per report)
    u32 promotions = 0, evictions_total = 0, pool_evictions = 0;
    u32 chunk_groups_built = 0, chunked_meshes_built = 0, split_skipped = 0; // (per report)
    float split_ms = 0.0f;                                                    // (per report)
    u32 probes_read = 0, probes_stale = 0, probes_matched = 0, probes_changed = 0;
    u32 probes_reordered = 0;
    // Why per-frame triangles are not cached (the last pass): skinned, too big, blacklisted,
    // not stable yet, part of a mesh whose copies are not all cached, cache off.
    std::array<u32, 6> per_frame_reasons{};
    std::array<float, 3> camera_world{};
    static constexpr u32 ProbeSlots = 4;
    std::array<std::vector<u64>, ProbeSlots> probe_keys; // table index -> key (0: not probed)
    std::array<u32, ProbeSlots> probe_ids{};
    // Scheduler ticks of the last pass that used each host ring slot: a slot is reused (or read
    // back) only once the GPU has passed that tick; otherwise the pass skips the frame.
    std::array<u64, ProbeSlots> probe_ticks{};
    u64 readback_tick = 0, dump_tick = 0;
    u32 skipped_busy = 0;   // passes skipped because the slot was still in use (per report)
    u64 index_bytes_frame = 0, index_bytes_small = 0; // measurement: index data per frame (all, < 16 KiB ranges)
    static constexpr u32 MaxInstances = 8192;
    static constexpr u32 PromoTriangles = 400000; // world-space triangles cached per frame at most
    static constexpr u32 ChunkGroupTriangles = 65536; // a piece of a big mesh: triangles at most
    static constexpr u32 MinPieceTriangles = 256;     // ... and pieces are not split below this
    u32 frame_triangles = 0, dropped_triangles = 0, dropped_draws = 0;
    u32 recorded = 0;         // passes recorded since the last report
    u32 skipped_topology = 0, skipped_no_address = 0, skipped_alpha = 0, skipped_nowrite = 0;
    bool readback_pending = false;
    u32 readback_frame = 0;
    u32 readback_triangles = 0;
    Buffer dump_buffer;       // host: DumpFrames consecutive half-res AO images (rg16f), "dump="
    static constexpr u32 DumpFrames = 8;
    u32 dump_index = 0;       // frames copied so far
    bool dump_pending = false;
    u32 dump_frame = 0;
    float dump_done = 0.0f;   // the last dump= value served
    float recreate_done = 0.0f; // the last recreate= value served
    u32 recorded_total = 0;
    VideoCore::ImageId depth_id{}, normal_id{};
    VideoCore::ImageId last_color{}, mask_candidate{};
    bool upsample_seen = false;   // the game's upsample shader was seen
    bool heuristic_reported = false; // an R8 scene-size target without it (reported once)
    u32 world_frames = 0;            // frames with G-buffer draws
    bool missing_pass_reported = false;
    bool done_this_frame = false;
    u32 frame = 0;
    u32 width = 0, height = 0;
    std::string settings_path;
    long long settings_mtime = -1;
    std::chrono::steady_clock::time_point last_poll{};
    bool history_valid = false;
    u32 last_recorded_frame = 0;            // frame of the previous traced pass
    std::array<float, 3> last_camera_world{}; // its camera position (cuts: > 1 m between passes)
    u32 history_resets = 0;
    u32 ao_index = 0;

    // Pipelines
    vk::UniqueDescriptorSetLayout expand_desc_layout, trace_desc_layout, apply_desc_layout, blur_desc_layout,
        probe_desc_layout, temporal_desc_layout;
    vk::UniquePipelineLayout expand_layout, trace_layout, apply_layout, blur_layout, probe_layout, temporal_layout;
    vk::UniquePipeline expand_pipeline, trace_pipeline, apply_pipeline, blur_pipeline, probe_pipeline,
        temporal_pipeline;
    vk::UniquePipeline rt_trace_pipeline; // ray-tracing pipeline variant of the trace
    Buffer sbt_buffer;
    vk::StridedDeviceAddressRegionKHR sbt_rgen{}, sbt_miss{}, sbt_hit{};
    // Geometry and acceleration structures
    static constexpr u32 TableSlots = 3;
    std::array<u64, TableSlots> slot_ticks{};
    Buffer table_buffer;      // host: GpuGeometry[max_geometries] x TableSlots
    Buffer readback_buffer;   // host: the first ReadbackTriangles triangles of a frame
    static constexpr u32 ReadbackTriangles = 4096;
    Buffer triangle_buffer;   // device: float3 x 3 x (max_triangles + PromoTriangles)
    Buffer probe_buffer;      // host (read back): ProbeStats[max_geometries] x ProbeSlots
    Buffer probe_chunks;      // host: (geometry, first index) per 4096-index chunk x TableSlots
    Buffer probe_ranges;      // host: (first chunk, chunks) per table entry x TableSlots
    Buffer probe_partials;    // device: per chunk sum/count, min, max
    Buffer split_fetch;       // host (read back, cached): a batch of big meshes' world triangles
    Buffer split_staging;     // host: the same, reordered piece by piece (build input)
    static constexpr u32 MaxProbeChunks = 32768;
    Buffer constants_buffer;  // host: the trace's per-frame constants x TableSlots
    Buffer index_ring;        // host: small index ranges copied at draw time, IndexRingSlot x TableSlots
    static constexpr vk::DeviceSize IndexRingSlot = 4ull << 20;
    vk::DeviceSize index_ring_used = 0;
    int index_ring_state = -1; // this frame: -1 not checked, 0 slot busy, 1 usable
    u32 skipped_copy = 0;      // draws left out: the index ring slot was busy or full (per report)
    bool recorded_this_frame = false;
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
    VideoCore::UniqueImage raw_image;  // half res, denoiser 2: this frame's trace (ao, z)
    vk::UniqueImageView raw_view;
    std::array<VideoCore::UniqueImage, 2> hist_images; // half res, denoiser 2: ao, z, N (ping-pong)
    std::array<vk::UniqueImageView, 2> hist_views;
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
