// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_rtao.h"
#include "video_core/renderer_vulkan/vk_rtao_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <sys/stat.h>
#include "common/path_util.h"

#include <vk_mem_alloc.h>

#include "video_core/host_shaders/rtao_apply_comp.h"
#include "video_core/host_shaders/rtao_blur_comp.h"
#include "video_core/host_shaders/rtao_expand_comp.h"
#include "video_core/host_shaders/rtao_probe_comp.h"
#include "video_core/host_shaders/rtao_temporal_comp.h"
#include "video_core/host_shaders/rtao_trace_comp.h"
#include "video_core/renderer_vulkan/vk_camera_motion.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_object_motion.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

using namespace RtaoInternal;

RayTracedAO::RayTracedAO(const Instance& instance_, Scheduler& scheduler_,
                         VideoCore::TextureCache& texture_cache_, VideoCore::BufferCache& buffer_cache_,
                         Runtime& runtime_, CameraMotion& camera_motion_, ObjectMotion& object_motion_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, buffer_cache{buffer_cache_},
      runtime{runtime_}, camera_motion{camera_motion_}, object_motion{object_motion_} {
    if (!Rtao::Requested()) {
        return;
    }
    if (!instance.IsRayQuerySupported()) {
        std::puts("RTAO: ray queries unsupported by the driver, off");
        return;
    }
    if (!object_motion.Enabled() || !ObjectMotion::RtaoCapture() || !camera_motion.Enabled()) {
        std::puts("RTAO: needs the object motion vectors (menu: object motion, with an upscaler), off");
        return;
    }
    settings.debug = std::min(EnvU32("BB_RTAO_DEBUG", 0), 4u);
    settings.strength = std::clamp(EnvFloat("BB_RTAO_STRENGTH", 1.0f), 0.0f, 1.0f);
    settings.radius = std::max(EnvFloat("BB_RTAO_RADIUS", 1.5f), 0.05f);
    settings.power = std::clamp(EnvFloat("BB_RTAO_POWER", 2.0f), 0.1f, 8.0f);
    settings.history = std::clamp(EnvFloat("BB_RTAO_HISTORY", 0.5f), 0.0f, 0.98f);
    settings.sigma = std::clamp(EnvFloat("BB_RTAO_SIGMA", 1.5f), 0.0f, 4.0f);
    settings.rays = std::clamp(EnvU32("BB_RTAO_RAYS", 2), 1u, 32u);
    max_triangles = std::clamp(EnvU32("BB_RTAO_MAX_TRIANGLES", 1500000), 10000u, 16000000u);
    settings.stage = std::clamp(EnvU32("BB_RTAO_STAGE", 5), 0u, 5u);
    settings.cache_enabled = EnvU32("BB_RTAO_CACHE", 1) != 0;
    settings.denoise = std::clamp(EnvU32("BB_RTAO_DENOISE", 2), 1u, 2u);
    // ALLOW_DATA_ACCESS on the per-frame BLAS although nothing reads positions back: on RADV the
    // flag turns RDNA4's compressed triangle-pair leaves off, and their build cost more than
    // their traversal saved (Doll, three sessions: build 1.65 vs 2.41 ms, trace 1.79 vs 1.73 ms).
    // The static BLAS: staticbuild= (fast build with the flag by default: a rebuild costs 1.4 ms
    // instead of 2.75 for fast trace, and traces 0.03 ms slower).
    data_access = EnvU32("BB_RTAO_DATA_ACCESS", 1) != 0;
    hold_enabled = EnvU32("BB_RTAO_HOLD", 1) != 0;
    test_move = EnvU32("BB_RTAO_TEST_MOVE", 0) != 0;

    const auto device = instance.GetDevice();
    const auto make_layouts = [&](std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                  u32 push_size, vk::UniqueDescriptorSetLayout& desc_layout,
                                  vk::UniquePipelineLayout& layout) {
        desc_layout = Check(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                          .offset = 0,
                                          .size = push_size};
        layout = Check(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &*desc_layout,
            .pushConstantRangeCount = push_size ? 1u : 0u,
            .pPushConstantRanges = push_size ? &range : nullptr,
        }));
    };
    const auto make_pipeline = [&](std::span<const u32> code, vk::PipelineLayout layout) {
        const auto module = CompileSPV(code, device);
        auto pipeline = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = module,
                              .pName = "main"},
                    .layout = layout,
                }));
        device.destroyShaderModule(module);
        return pipeline;
    };
    const auto binding = [](u32 index, vk::DescriptorType type) {
        return vk::DescriptorSetLayoutBinding{.binding = index,
                                              .descriptorType = type,
                                              .descriptorCount = 1,
                                              .stageFlags = vk::ShaderStageFlagBits::eCompute};
    };
    using DT = vk::DescriptorType;
    const std::array expand_bindings = {binding(0, DT::eStorageBuffer), binding(1, DT::eStorageBuffer)};
    const bool rt_pipeline_supported = instance.IsRayTracingPipelineSupported();
    const auto trace_binding = [&](u32 index, vk::DescriptorType type) {
        auto b = binding(index, type);
        if (rt_pipeline_supported) {
            b.stageFlags |= vk::ShaderStageFlagBits::eRaygenKHR;
        }
        return b;
    };
    const std::array trace_bindings = {trace_binding(0, DT::eSampledImage), trace_binding(1, DT::eSampledImage),
                                       trace_binding(2, DT::eSampledImage), trace_binding(3, DT::eSampledImage),
                                       trace_binding(4, DT::eStorageImage),
                                       trace_binding(5, DT::eAccelerationStructureKHR),
                                       trace_binding(6, DT::eUniformBuffer)};
    const std::array apply_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eSampledImage),
                                       binding(2, DT::eStorageImage)};
    make_layouts(expand_bindings, sizeof(ExpandConstants), expand_desc_layout, expand_layout);
    make_layouts(trace_bindings, 0, trace_desc_layout, trace_layout); // constants: a uniform buffer
    const std::array blur_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eStorageImage),
                                      binding(2, DT::eSampledImage)};
    const std::array temporal_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eSampledImage),
                                          binding(2, DT::eSampledImage), binding(3, DT::eStorageImage),
                                          binding(4, DT::eStorageImage)};
    make_layouts(temporal_bindings, sizeof(TemporalConstants), temporal_desc_layout, temporal_layout);
    temporal_pipeline = make_pipeline(RTAO_TEMPORAL_COMP, *temporal_layout);
    make_layouts(apply_bindings, sizeof(ApplyConstants), apply_desc_layout, apply_layout);
    make_layouts(blur_bindings, sizeof(BlurConstants), blur_desc_layout, blur_layout);
    blur_pipeline = make_pipeline(RTAO_BLUR_COMP, *blur_layout);
    const std::array probe_bindings = {binding(0, DT::eStorageBuffer), binding(1, DT::eStorageBuffer),
                                       binding(2, DT::eStorageBuffer), binding(3, DT::eStorageBuffer),
                                       binding(4, DT::eStorageBuffer)};
    make_layouts(probe_bindings, sizeof(ProbeConstants), probe_desc_layout, probe_layout);
    probe_pipeline = make_pipeline(RTAO_PROBE_COMP, *probe_layout);
    expand_pipeline = make_pipeline(RTAO_EXPAND_COMP, *expand_layout);
    trace_pipeline = make_pipeline(RTAO_TRACE_COMP, *trace_layout);
    apply_pipeline = make_pipeline(RTAO_APPLY_COMP, *apply_layout);
    if (rt_pipeline_supported) {
        CreateTracePipeline();
        // Measured at the Doll: trace 1.40-1.44 ms as a pipeline vs 1.62-1.64 ms with ray queries.
        settings.use_rt_pipeline = rt_trace_pipeline && EnvU32("BB_RTAO_PIPELINE", 1) != 0;
    }
    // Names for validation messages and captures.
    SetObjectName(device, *expand_pipeline, "rtao expand");
    SetObjectName(device, *probe_pipeline, "rtao probe");
    SetObjectName(device, *trace_pipeline, "rtao trace");
    SetObjectName(device, *blur_pipeline, "rtao blur");
    SetObjectName(device, *apply_pipeline, "rtao apply");
    SetObjectName(device, *temporal_pipeline, "rtao temporal");

    // The buffers (over a GiB) are made at the first frame with a scene camera, not here: made
    // here, before the game's own resources, past ~1.3 GiB the game's lights went out (lanterns,
    // candles; reproducibly, VRAM far from full). See docs/rtao.md.
    enabled = true;
    std::printf("RTAO: on, %u rays, denoiser %u, radius %.2f, strength %.2f, up to %u triangles%s\n",
                settings.rays, settings.denoise, settings.radius, settings.strength, max_triangles, settings.debug ? ", debug view" : "");
    if (const char* file = std::getenv("BB_RTAO_FILE")) {
        settings_path = file;
    } else {
        settings_path = (Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "rtao.txt").string();
    }
    std::printf("RTAO: settings re-read from %s while the game runs\n", settings_path.c_str());
}

void RayTracedAO::PollSettings() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_poll < std::chrono::seconds(1)) {
        return;
    }
    last_poll = now;
    struct stat st{};
    if (::stat(settings_path.c_str(), &st) != 0) {
        return;
    }
    const long long mtime = static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
    if (mtime == settings_mtime) {
        return;
    }
    settings_mtime = mtime;
    std::ifstream file(settings_path);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    bool history_reset = false;
    Rtao::ApplyText(settings, text, history_reset);
    if (history_reset) {
        history_valid = false; // the two denoisers' histories differ
    }
    settings.use_rt_pipeline = settings.use_rt_pipeline && rt_trace_pipeline;
    settings.build_fast_trace = settings.build_fast_trace && fast_trace_allowed;
    std::printf("RTAO: settings: enabled %d, %u rays, denoiser %u (maxn %.0f, clip %.1f / %.2f), radius %.2f, "
                "strength %.2f, power %.2f, far %.0f m, snap %.0f m, normals %s, trace %s, cache %d (maxsize %.0f m), "
                "debug %u, stage %u\n",
                settings.active ? 1 : 0, settings.rays, settings.denoise, settings.max_n, settings.clip_k,
                settings.clip_floor, settings.radius, settings.strength, settings.power, settings.ao_far,
                settings.snap_far, settings.depth_normals ? "depth" : "shading",
                settings.use_rt_pipeline ? "pipeline" : "ray query", settings.cache_enabled ? 1 : 0,
                settings.cache_max_size, settings.debug, settings.stage);
}

RayTracedAO::~RayTracedAO() {
    if (split_result.valid()) {
        split_result.wait(); // it writes into the staging buffer
    }
    const auto device = instance.GetDevice();
    for (auto& entry : view_cache) {
        if (entry.view) {
            device.destroyImageView(entry.view);
        }
    }
    if (tlas) {
        device.destroyAccelerationStructureKHR(tlas);
    }
    if (blas) {
        device.destroyAccelerationStructureKHR(blas);
    }
    if (static_blas) {
        device.destroyAccelerationStructureKHR(static_blas);
    }
    for (auto* buffer : {&table_buffer, &triangle_buffer, &blas_buffer, &blas_scratch, &tlas_buffer,
                         &tlas_scratch, &instance_buffer, &readback_buffer, &dump_buffer, &probe_buffer,
                         &static_pool, &static_blas_buffer, &constants_buffer, &sbt_buffer, &index_ring, &probe_chunks,
                         &probe_ranges, &probe_partials, &split_fetch, &split_staging}) {
        DestroyBuffer(*buffer);
    }
}






void RayTracedAO::OnFrameStart() {
    if (!enabled) {
        return;
    }
    if (!resources_ready) {
        // The buffers, once the game is in the world (its own resources exist by then).
        const auto proj = camera_motion.Projection();
        if (proj[0] == 0.0f || proj[1] == 0.0f) {
            ++frame;
            return;
        }
        if (!CreateAccelerationStructures()) {
            std::puts("RTAO: acceleration structure setup failed, off");
            enabled = false;
            return;
        }
        resources_ready = true;
    }
    static u32 reports = 0;
    WriteDump();
    ReportReadback();
    if (frame % 600 == 0 && frame) {
        std::printf("RTAO: frame %u: %zu draws, %u triangles (%u draws, %u triangles dropped: budget %u; "
                    "%u strips/other, %u alpha-tested, %u without depth write, %u without index address "
                    "skipped), %u passes recorded in 600 frames, %u skipped (ring slot still in use)\n",
                    frame, geometries.size(), frame_triangles, dropped_draws, dropped_triangles,
                    max_triangles, skipped_topology, skipped_alpha, skipped_nowrite, skipped_no_address,
                    recorded, skipped_busy);
        if (skipped_copy) {
            std::printf("RTAO: %u draws left out (index ring slot busy or full) in 600 frames\n", skipped_copy);
            skipped_copy = 0;
        }
        std::printf("RTAO: index data of the last frame: %llu KiB (%llu KiB in ranges of <= 16 KiB)\n",
                    (unsigned long long)(index_bytes_frame >> 10), (unsigned long long)(index_bytes_small >> 10));
        if (history_resets) {
            std::printf("RTAO: history reset %u times (gaps or camera jumps) in 600 frames\n", history_resets);
            history_resets = 0;
        }
        skipped_busy = 0;
        std::printf("RTAO: cache: %zu draws known, pool %u of %u triangles; %u cached, %u evicted (%u for room) "
                    "in 600 frames; static BLAS %u triangles in %u parts, %u rebuilds in 600 frames; probes %u read, "
                    "%u not written, %u same, %u reordered, %u changed; camera (%.1f %.1f %.1f)\n",
                    cache.size(), pool.Used(), pool.Size(), promotions, evictions_total, pool_evictions,
                    static_triangles, static_geometries, static_rebuilds, probes_read, probes_stale,
                    probes_matched, probes_reordered, probes_changed, camera_world[0], camera_world[1],
                    camera_world[2]);
        static_rebuilds = pool_evictions = 0;
        probes_read = probes_stale = probes_matched = probes_changed = probes_reordered = 0;
        if (chunked_meshes_built || split_skipped) {
            std::printf("RTAO: big meshes in pieces: %u cached (%u pieces, split %.1f ms on the worker), %u dropped "
                        "(changed meanwhile, or no room) in 600 frames\n",
                        chunked_meshes_built, chunk_groups_built, split_ms, split_skipped);
            chunked_meshes_built = chunk_groups_built = split_skipped = 0;
            split_ms = 0.0f;
        }
        std::printf("RTAO: per-frame triangles by reason: %u skinned, %u bigger than maxsize, %u kept changing, "
                    "%u not stable yet, %u copies of a mesh not all cached, %u cache off\n",
                    per_frame_reasons[0], per_frame_reasons[1], per_frame_reasons[2], per_frame_reasons[3],
                    per_frame_reasons[4], per_frame_reasons[5]);
        std::printf("RTAO: memory: fixed %llu MiB (the static pool and BLAS included)\n",
                    (unsigned long long)(static_bytes >> 20));
        promotions = 0;
        evictions_total = 0;
        if (recorded == 0 && !draws.empty() && settings.active && !upsample_seen) {
            pass_missing = true; // no capture for a pass that never runs
        }
        if (recorded == 0 && !draws.empty() && settings.active && !missing_pass_reported) {
            missing_pass_reported = true;
            std::puts("RTAO: OFF in practice: the game's AO pass never ran, so there is no AO mask to "
                      "write into. The \"Disable SSAO\" game patch (launcher: Game effects, SSAO "
                      "unchecked) removes it; RTAO needs that pass. (Reported once.)");
        }
        recorded = 0;
        ++reports;
    }
    TrimCache();
    world_frames += draws.empty() ? 0 : 1;
    if (frame % 600 != 599) { // keep the last frame's totals for the report
        index_bytes_frame = index_bytes_small = 0;
    }
    draws.clear();
    occurrences.clear();
    index_ring_state = -1;
    index_ring_used = 0;
    dropped_triangles = 0;
    dropped_draws = 0;
    skipped_topology = 0;
    skipped_no_address = 0;
    skipped_alpha = 0;
    skipped_nowrite = 0;
    done_this_frame = false;
    last_color = {};
    mask_candidate = {};
    ++frame;
    PollSettings();
    // Mirrors over the index ranges read by address this frame wait for the pass (it reads them
    // when the game's AO pass ends, draws later); nothing else is held, and only while it is on.
    buffer_cache.ReleaseRetirements(scheduler.CurrentTick());
    if (Active() && hold_enabled) {
        buffer_cache.HoldRetirements();
    }
}

void RayTracedAO::OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal) {
    depth_id = depth;
    normal_id = normal;
}

void RayTracedAO::AddGeometry(const Geometry& g) {
    if (!enabled || !resources_ready || g.index_count < 3 || !g.vertices || !g.instances) {
        return;
    }
    if (!g.triangle_list) { // strips and the rest: not expanded (yet), counted
        ++skipped_topology;
        return;
    }
    if (g.alpha_test && !settings.include_alpha) {
        ++skipped_alpha;
        return;
    }
    if (!g.depth_write && !settings.include_nowrite) {
        ++skipped_nowrite;
        return;
    }
    if (g.index_address == 0 && !g.non_indexed) {
        ++skipped_no_address; // an index buffer without a device address
        return;
    }
    const u32 tris = (g.index_count / 3) * g.instances;
    if (g.non_indexed && g.index_count > g.vertices) {
        ++dropped_draws; // the capture holds fewer vertices than the draw
        dropped_triangles += tris;
        return;
    }
    // The positions must lie inside the capture (two halves of the object-motion buffer).
    const u64 last = u64(g.store) + u64(g.vertices) * g.instances;
    if (last > object_motion.PositionsEntries()) {
        ++dropped_draws;
        dropped_triangles += tris;
        return;
    }
    if (draws.size() >= max_geometries) {
        ++dropped_draws;
        dropped_triangles += tris;
        return;
    }
    u64 index_address = g.index_address;
    bool index_mid_word = (g.index_address & 2) != 0;
    // Small ranges are copied now (the stream ring may be reused before the pass); the others
    // are held by the buffer cache until the pass is recorded.
    const bool copy_indices = g.index_guest && g.index_bytes &&
                              g.index_bytes <= VideoCore::BufferCache::STREAM_THRESHOLD &&
                              !buffer_cache.IsRegionGpuModified(g.index_guest, g.index_bytes) &&
                              Core::Memory::Instance()->IsValidGpuMapping(g.index_guest, g.index_bytes) &&
                              Core::Memory::Instance()->ClampRangeSize(g.index_guest, g.index_bytes) == g.index_bytes;
    if (g.index_guest && g.index_bytes && !copy_indices) {
        buffer_cache.HoldRange(g.index_guest, g.index_bytes);
    }
    if (copy_indices) {
        // The frame's ring slot is the one the pass will use; only once the GPU is done with it.
        const u32 copy_slot = settings.ring_test ? 0u : frame % TableSlots;
        if (index_ring_state < 0) {
            index_ring_state = (!slot_ticks[copy_slot] || scheduler.IsFree(slot_ticks[copy_slot])) ? 1 : 0;
            index_ring_used = 0;
        }
        const vk::DeviceSize at = (index_ring_used + 3) & ~vk::DeviceSize(3);
        if (index_ring_state == 0 || at + g.index_bytes > IndexRingSlot) {
            ++skipped_copy;
            return;
        }
        const vk::DeviceSize offset = vk::DeviceSize(copy_slot) * IndexRingSlot + at;
        // Under the runtime's memory lock, from the backing store: the game may unmap or rewrite
        // the range meanwhile (an unmapped page reads as zeros, which the expand drops as
        // degenerate, not as a host fault).
        Core::Memory::Instance()->ReadBacking(g.index_guest, static_cast<u8*>(index_ring.mapped) + offset,
                                              g.index_bytes);
        index_ring_used = at + g.index_bytes;
        index_address = index_ring.address + offset;
        index_mid_word = false;
    }
    // The same buffers may be drawn more than once a frame: the occurrence keeps them apart.
    {
        const u64 bytes = g.non_indexed ? 0 : u64(g.index_count) * (g.index16 ? 2 : 4);
        index_bytes_frame += bytes;
        index_bytes_small += bytes <= 16384 ? bytes : 0;
    }
    const u32 occurrence = occurrences[g.key]++;
    u32& copies = max_copies[g.key];
    copies = std::max(copies, occurrence + 1);
    Draw& draw = draws.emplace_back();
    draw.key = Rtao::KeyOf(g.key, occurrence);
    draw.base = g.key;
    draw.occurrence = occurrence;
    draw.skinned = g.skinned;
    draw.gpu = GpuGeometry{.store = g.store,
                          .vertices = g.vertices,
                          .instances = g.instances,
                          .first_vertex = g.first_vertex,
                          // 16-bit index buffers may start 2 bytes into a word: read words
                          // from the aligned address and skip the first half-word.
                          .index_address = g.index16 ? (index_address & ~u64(3)) : index_address,
                          .index_count = g.index_count,
                          .index16 = g.index16 ? (1u | (index_mid_word ? 2u : 0u)) : 0u,
                          .base_vertex = g.base_vertex,
                          .restart = g.restart ? 1u : 0u,
                          .triangle_offset = 0,
                          .triangles = tris,
                          .out_base = 0,
                          .flags = 0};
}





void RayTracedAO::OnColorTarget(VideoCore::ImageId color, u64 ps_hash) {
    if (!enabled || !resources_ready) {
        return;
    }
    if (test_move_pending) { // the draw after the AO upsample's first one, before the pass
        test_move_pending = false;
        if (++test_moves % 600 == 1) {
            std::printf("RTAO test: layer mirrors moved back and submitted before the AO pass (%u)\n",
                        test_moves);
        }
        buffer_cache.DemoteAllMirrors();
        scheduler.Flush();
    }
    if (test_move && ps_hash == AoUpsampleShader && Active()) {
        test_move_pending = true;
    }
    if (color != last_color) {
        // The previous pass ended. Was it the game's AO upsample (into the R8 mask)?
        if (last_color && last_color == mask_candidate && !done_this_frame && depth_id && settings.active &&
            !draws.empty()) {
            done_this_frame = true;
            Record(mask_candidate);
            if (recorded_this_frame) { // what the pass reads is in its commands: release the hold
                recorded_this_frame = false;
                buffer_cache.ReleaseRetirements(scheduler.CurrentTick());
            }
        }
        last_color = color;
        mask_candidate = {};
    }
    if (!color || !depth_id || mask_candidate) {
        return;
    }
    if (ps_hash == AoUpsampleShader) {
        pass_missing = false;
        if (!upsample_seen) {
            upsample_seen = true;
            std::printf("RTAO: the game's AO pass found by its upsample shader 0x%llx\n",
                        (unsigned long long)AoUpsampleShader);
        }
        mask_candidate = color;
        return;
    }
    // Only the identified pass is written. An R8 scene-size target while the shader was never
    // seen is reported once (a game update or a shader pack that changed the upsample shader's
    // hash), and left alone.
    if (upsample_seen || heuristic_reported) {
        return;
    }
    const auto& image = texture_cache.GetImage(color);
    const auto& depth = texture_cache.GetImage(depth_id);
    if (image.info.pixel_format == vk::Format::eR8Unorm &&
        image.info.size.width == depth.info.size.width &&
        image.info.size.height == depth.info.size.height && world_frames > 120) {
        heuristic_reported = true;
        std::printf("RTAO: an R8 scene-size target (pixel shader 0x%llx) but the game's AO upsample "
                    "shader 0x%llx was not seen: not written (has its hash changed?)\n",
                    (unsigned long long)ps_hash, (unsigned long long)AoUpsampleShader);
    }
}



void RayTracedAO::Record(VideoCore::ImageId mask_id) {
    auto& mask = texture_cache.GetImage(mask_id);
    auto& depth = texture_cache.GetImage(depth_id);
    auto& normal = texture_cache.GetImage(normal_id);
    const u32 w = depth.info.size.width, h = depth.info.size.height;
    if (settings.recreate_request != recreate_done) { // recreate=: as a resize would
        recreate_done = settings.recreate_request;
        width = 0;
    }
    if (normal.info.size.width != w || normal.info.size.height != h || !EnsureResources(w, h)) {
        return;
    }
    // The host ring slot (geometry table, instances, constants): only once the GPU is done with it.
    const u32 slot = settings.ring_test ? 0u : frame % TableSlots;
    if (slot_ticks[slot] && !scheduler.IsFree(slot_ticks[slot])) {
        ++skipped_busy; // the game's own AO stays this frame
        return;
    }

    const auto depth_view = CachedView(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const auto normal_view = CachedView(normal, normal.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const auto mask_view = CachedView(mask, vk::Format::eR8Unorm, vk::ImageAspectFlagBits::eColor);

    scheduler.EndRendering();
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, TraceStages, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&normal, vk::ImageLayout::eGeneral, TraceStages, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&mask, vk::ImageLayout::eGeneral, Compute, rw);
    runtime.FlushBarriers();

    // The history pair swaps once the trace is recorded (the stage limiter may stop before it).
    const u32 read = ao_index, write = ao_index ^ 1;
    const auto image_barrier = [&](vk::Image image, vk::ImageLayout old_layout,
                                   vk::AccessFlags2 src_access, vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 b{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = src_access,
            .dstStageMask = TraceStages,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        scheduler.Record([b](vk::CommandBuffer cmdbuf) {
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
        });
    };
    image_barrier(vk::Image(motion_image), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
                  vk::AccessFlagBits2::eShaderWrite);
    image_barrier(vk::Image(ao_images[write]), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
                  vk::AccessFlagBits2::eShaderWrite);
    image_barrier(vk::Image(blur_image), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
                  vk::AccessFlagBits2::eShaderWrite);
    // Denoiser 2: this frame's raw trace and the history pair (same read/write indices).
    const bool v2 = settings.denoise == 2 && settings.debug < 2; // the debug views keep the plain path
    image_barrier(vk::Image(raw_image), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
                  vk::AccessFlagBits2::eShaderWrite);
    image_barrier(vk::Image(hist_images[write]), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
                  vk::AccessFlagBits2::eShaderWrite);
    image_barrier(vk::Image(hist_images[read]),
                  history_valid ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined, rw,
                  vk::AccessFlagBits2::eShaderRead);
    image_barrier(vk::Image(ao_images[read]),
                  history_valid ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined, rw,
                  vk::AccessFlagBits2::eShaderRead);
    camera_motion.RecordMotion(depth_view, *motion_view, w, h);
    image_barrier(vk::Image(motion_image), vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
                  vk::AccessFlagBits2::eShaderRead);

    const auto cam_proj = camera_motion.Projection();
    if (cam_proj[0] == 0.0f || cam_proj[1] == 0.0f || cam_proj[3] == 0.0f) {
        return; // no camera this frame (menus): nothing sensible to build
    }
    ++recorded;
    ++recorded_total;

    // Static cache. The probes of the pass that last used this ring slot are read back first
    // (four passes ago: long finished), then the frame's draws are sorted: cached ones become
    // instances of their world-space BLAS, the rest is per-frame geometry, and draws that stayed
    // the same for promote_after observations get their BLAS built this frame.
    const u32 probe_id = recorded_total;
    const u32 ring = probe_id % ProbeSlots;
    if (probe_ids[ring] && probe_ticks[ring] && !scheduler.IsFree(probe_ticks[ring])) {
        probe_ids[ring] = 0; // not finished: those results are dropped, not read half-written
    }
    ReadProbes(ring);
    const auto view = camera_motion.View();
    const auto inv_view = camera_motion.InvView();
    camera_world = {inv_view[3], inv_view[7], inv_view[11]};
    // Cuts the upscaler does not report: frames without this pass (loading screens, menus) or a
    // camera jump no walk or sprint makes in one frame (warps, scripted cuts).
    {
        const float dx = camera_world[0] - last_camera_world[0], dy = camera_world[1] - last_camera_world[1],
                    dz = camera_world[2] - last_camera_world[2];
        if (history_valid && (frame != last_recorded_frame + 1 || dx * dx + dy * dy + dz * dz > 1.0f)) {
            history_valid = false;
            ++history_resets;
        }
        last_recorded_frame = frame;
        last_camera_world = camera_world;
    }
    const bool use_cache = settings.cache_enabled && settings.stage >= 3;
    const float cull = settings.ao_far > 0.0f ? settings.ao_far + settings.radius : 0.0f;
    const auto device = instance.GetDevice();
    // Copies into the static pool: world-space triangles from the triangle list (promotions) or the
    // split staging buffer (pieces of big meshes).
    struct PoolCopy {
        bool from_staging;
        vk::DeviceSize src, dst, bytes;
    };
    std::vector<GpuGeometry> dynamic_list, verify_list, promo_list;
    std::vector<u64> dynamic_keys, verify_keys;
    std::vector<PoolCopy> pool_copies;
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    u32 promo_used = 0;
    frame_triangles = 0;
    per_frame_reasons = {};
    const auto make_instance = [&](vk::DeviceAddress address, bool world) {
        vk::AccelerationStructureInstanceKHR as_instance{
            .instanceCustomIndex = world ? 1u : 0u,
            .mask = 0xFF,
            .instanceShaderBindingTableRecordOffset = 0,
            .flags = VkGeometryInstanceFlagsKHR(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable),
            .accelerationStructureReference = address,
        };
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                as_instance.transform.matrix[r][c] = world ? view[r * 4 + c] : (r == c ? 1.0f : 0.0f);
            }
        }
        return as_instance;
    };
    // Huge draws (terrain, building blocks): whole, their far triangles came back into the BVH
    // (the far cut works per frame) and rays slowed down. They stay per frame, or (chunk > 0) are
    // split into pieces, each in the static BLAS only within reach.
    const auto too_big = [&](const CacheEntry& e) {
        return e.have_stats && Rtao::TooBig(e.stats, settings.cache_max_size);
    };
    const bool chunking = settings.chunk_cell > 0.0f;
    for (const Draw& d : draws) {
        CacheEntry& e = cache[d.key];
        e.base = d.base;
        e.last_seen = frame;
        if (!e.parts.empty() && (!use_cache || (e.chunked ? !chunking || !too_big(e) : too_big(e)))) {
            Evict(e); // the settings changed: cached the other way, or not at all
        }
    }
    UpdateSplit();
    // The static BLAS: every pooled draw seen lately, its parts within reach of the camera (the far
    // cut plus a margin), in one tree. Rebuilt when the camera left the margin, a draw in it was
    // evicted, or (at most every `rebuild` frames) the set changed.
    {
        const float reach = cull > 0.0f ? cull + settings.static_margin : 0.0f;
        bool set_changed = false;
        for (const auto& [key, e] : cache) {
            const bool candidate = !e.parts.empty() && frame >= e.ready_from &&
                                   frame - e.last_seen <= StaticSeenFrames && !e.static_skip;
            if (candidate != e.in_static) {
                set_changed = true;
                break;
            }
        }
        const float dx = camera_world[0] - static_anchor[0], dy = camera_world[1] - static_anchor[1],
                    dz = camera_world[2] - static_anchor[2];
        const bool moved = cull > 0.0f && dx * dx + dy * dy + dz * dz > settings.static_margin * settings.static_margin;
        const bool due = frame - static_frame >= settings.static_interval;
        const bool always = settings.static_interval == 0; // measurement: the rebuild's cost
        if (use_cache && (always || static_dirty || (static_valid && moved) || (set_changed && (due || !static_valid)))) {
            RebuildStatic(reach);
        } else if (!use_cache && static_valid) {
            static_valid = false;
            for (auto& [key, e] : cache) {
                e.in_static = false;
            }
        }
    }
    // A mesh leaves the per-frame BVH only when every copy drawn this frame is in the static BLAS:
    // which draw is which copy changes with the draw order, but the set covers them all.
    std::unordered_map<u64, bool> mesh_cached;
    for (const Draw& d : draws) {
        const CacheEntry& e = cache[d.key];
        auto [it, inserted] = mesh_cached.try_emplace(d.base, true);
        it->second = it->second && use_cache && e.in_static;
    }
    // Probing costs a pass over the draw's indices: cached draws are rechecked every 8th pass,
    // draws waiting for a build slot every 4th; characters and draws that kept changing never.
    const auto probe_now = [&](const Draw& d, const CacheEntry& e) {
        if (d.skinned || e.evictions >= 3) {
            return false;
        }
        const u32 phase = u32(d.key >> 7) + probe_id;
        if (!e.parts.empty()) {
            return phase % 8 == 0;
        }
        if (too_big(e) && !chunking) {
            return phase % 64 == 0; // it will stay per frame; an occasional look is enough
        }
        if (e.stable >= settings.promote_after) {
            return phase % 4 == 0;
        }
        return true;
    };
    // Room in the pool: least recently seen draws (not seen this frame) go first.
    const auto pool_alloc = [&](u32 triangles) -> std::optional<u32> {
        if (auto offset = pool.Alloc(triangles)) {
            return offset;
        }
        std::vector<std::pair<u32, u64>> order; // last seen, key
        for (const auto& [key, e] : cache) {
            if (!e.parts.empty() && e.last_seen != frame) {
                order.emplace_back(e.last_seen, key);
            }
        }
        std::sort(order.begin(), order.end());
        for (const auto& [seen, key] : order) {
            Evict(cache[key]);
            ++pool_evictions;
            if (auto offset = pool.Alloc(triangles)) {
                return offset;
            }
        }
        return std::nullopt;
    };
    // Big meshes are collected for a split only while no batch is in flight.
    const bool collect_split = chunking && use_cache && split_state == SplitState::Idle;
    if (collect_split) {
        split_meshes.clear();
    }
    for (const Draw& d : draws) {
        CacheEntry& e = cache[d.key];
        GpuGeometry gpu = d.gpu;
        const u32 probe_flag = probe_now(d, e) ? GeoProbe : 0u;
        if (mesh_cached[d.base]) {
            if (probe_flag) { // still checked: a door that starts moving leaves the cache
                gpu.triangles = 0;
                gpu.flags = probe_flag;
                verify_list.push_back(gpu);
                verify_keys.push_back(d.key);
            }
            continue;
        }
        if (frame_triangles + gpu.triangles > max_triangles) {
            ++dropped_draws;
            dropped_triangles += gpu.triangles;
            if (probe_flag) { // measured all the same: once cached it no longer needs budget
                gpu.triangles = 0;
                gpu.flags = probe_flag;
                verify_list.push_back(gpu);
                verify_keys.push_back(d.key);
            }
            continue;
        }
        {
            const int reason = !use_cache ? 5 : d.skinned ? 0 : too_big(e) && !chunking ? 1 : e.evictions >= 3 ? 2
                               : e.parts.empty() ? 3 : 4;
            per_frame_reasons[reason] += gpu.triangles;
        }
        gpu.triangle_offset = frame_triangles;
        gpu.out_base = frame_triangles;
        gpu.flags = probe_flag;
        frame_triangles += gpu.triangles;
        dynamic_list.push_back(gpu);
        dynamic_keys.push_back(d.key);
        const bool big = too_big(e);
        if (!use_cache || e.split_pending || !e.parts.empty() ||
            !Rtao::Promotable(e, d.skinned, big && !chunking, settings.promote_after) ||
            gpu.triangles == 0 || promo_used + gpu.triangles > PromoTriangles) {
            continue;
        }
        { // the same copy cached under another slot (before a reorder): not twice
            bool duplicate = false;
            const auto copies = max_copies.find(d.base);
            for (u32 o = 0; copies != max_copies.end() && o < copies->second && !duplicate; ++o) {
                const auto other = cache.find(Rtao::KeyOf(d.base, o));
                duplicate = other != cache.end() && &other->second != &e && !other->second.parts.empty() &&
                            Rtao::StatsMatch(other->second.stats, e.stats, camera_world);
            }
            if (duplicate) {
                continue;
            }
        }
        const u32 out_base = max_triangles + promo_used;
        if (big) {
            // A big mesh: its world-space triangles go to the host for the split (one batch at a time).
            if (!collect_split) {
                continue;
            }
            split_meshes.push_back({.key = d.key, .stats = e.stats, .offset = promo_used, .triangles = gpu.triangles});
            e.split_pending = true;
        } else {
            // Promotion: a world-space copy of the triangles into the pool, for the static BLAS.
            const auto offset = pool_alloc(gpu.triangles);
            if (!offset) {
                continue;
            }
            CachePart& part = e.parts.emplace_back();
            part.pool_offset = *offset;
            part.triangles = gpu.triangles;
            for (int i = 0; i < 3; ++i) {
                part.box.min[i] = e.stats.min[i];
                part.box.max[i] = e.stats.max[i];
            }
            e.triangles = gpu.triangles;
            e.cached = true;
            e.ready_from = frame + 1; // copied in this pass, in the static BLAS from a later one
            ++promotions;
            pool_copies.push_back({false, vk::DeviceSize(out_base) * 36, vk::DeviceSize(*offset) * 36,
                                   vk::DeviceSize(gpu.triangles) * 36});
        }
        GpuGeometry world = gpu;
        world.out_base = out_base;
        world.flags = GeoWorld;
        promo_list.push_back(world);
        promo_used += gpu.triangles;
    }
    const bool split_fetch_now = collect_split && !split_meshes.empty();
    if (split_fetch_now) {
        split_state = SplitState::Fetching;
    }
    // Pieces of split big meshes: copied from the staging buffer into the pool.
    if (split_state == SplitState::Building) {
        for (auto& mesh : split_meshes) {
            if (mesh.done) {
                continue;
            }
            mesh.done = true;
            const auto it = cache.find(mesh.key);
            CacheEntry* e = it != cache.end() ? &it->second : nullptr;
            u32 active = 0;
            for (const auto& g : mesh.groups) {
                active += g.triangles;
            }
            const auto offset = e && e->parts.empty() && chunking && use_cache && e->have_stats &&
                                        Rtao::StatsMatch(mesh.stats, e->stats, camera_world) && active
                                    ? pool_alloc(active)
                                    : std::nullopt;
            if (e) {
                e->split_pending = false;
            }
            if (!offset) {
                ++split_skipped; // changed or gone since it was read, or no room
                continue;
            }
            for (const auto& g : mesh.groups) {
                CachePart& part = e->parts.emplace_back();
                part.pool_offset = *offset + g.first_triangle;
                part.triangles = g.triangles;
                part.box = g.box;
            }
            e->triangles = active;
            e->chunked = true;
            e->cached = true;
            e->ready_from = frame + 1;
            ++promotions;
            ++chunked_meshes_built;
            chunk_groups_built += u32(mesh.groups.size());
            pool_copies.push_back({true, vk::DeviceSize(mesh.staging_offset) * 36, vk::DeviceSize(*offset) * 36,
                                   vk::DeviceSize(active) * 36});
            split_tick = scheduler.CurrentTick(); // the staging buffer is read until this pass is done
        }
    }
    // The table: per-frame entries, then the world-space copies (expanded after them), then the
    // entries that are only measured (no triangles; their offset lies past the expand range).
    geometries.clear();
    probe_keys[ring].clear();
    geometries.insert(geometries.end(), dynamic_list.begin(), dynamic_list.end());
    for (size_t i = 0; i < dynamic_list.size(); ++i) {
        probe_keys[ring].push_back((dynamic_list[i].flags & GeoProbe) ? dynamic_keys[i] : 0);
    }
    u32 dispatch_triangles = frame_triangles;
    for (auto& world : promo_list) {
        world.triangle_offset = dispatch_triangles;
        dispatch_triangles += world.triangles;
        geometries.push_back(world);
        probe_keys[ring].push_back(0);
    }
    for (size_t i = 0; i < verify_list.size() && geometries.size() < max_geometries; ++i) {
        verify_list[i].triangle_offset = dispatch_triangles;
        geometries.push_back(verify_list[i]);
        probe_keys[ring].push_back(verify_keys[i]);
    }
    // Probe chunks: 4096 indices each (pass 1), then one combine per draw (pass 2).
    u32 num_chunks = 0;
    {
        auto* chunk_map = static_cast<std::array<u32, 2>*>(probe_chunks.mapped) + size_t(slot) * MaxProbeChunks;
        auto* range_map = static_cast<std::array<u32, 2>*>(probe_ranges.mapped) + size_t(slot) * max_geometries;
        for (size_t i = 0; i < geometries.size(); ++i) {
            auto& g = geometries[i];
            range_map[i] = {0, 0};
            if (!(g.flags & GeoProbe)) {
                continue;
            }
            const u32 total = g.index_count * g.instances;
            const u32 n = (total + 4095) / 4096;
            if (num_chunks + n > MaxProbeChunks) { // too much this frame: next time
                g.flags &= ~GeoProbe;
                probe_keys[ring][i] = 0;
                continue;
            }
            range_map[i] = {num_chunks, n};
            for (u32 k = 0; k < n; ++k) {
                chunk_map[num_chunks + k] = {u32(i), k * 4096};
            }
            num_chunks += n;
        }
        vmaFlushAllocation(instance.GetAllocator(), probe_chunks.allocation, vk::DeviceSize(slot) * MaxProbeChunks * 8,
                           std::max<vk::DeviceSize>(num_chunks, 1) * 8);
        vmaFlushAllocation(instance.GetAllocator(), probe_ranges.allocation, vk::DeviceSize(slot) * max_geometries * 8,
                           std::max<vk::DeviceSize>(geometries.size(), 1) * 8);
    }
    probe_ids[ring] = probe_id;
    if (index_ring_used) {
        vmaFlushAllocation(instance.GetAllocator(), index_ring.allocation,
                           vk::DeviceSize(slot) * IndexRingSlot, index_ring_used);
    }
    recorded_this_frame = true;
    // This pass's commands go out with the current tick: the slots are busy until it is reached.
    slot_ticks[slot] = scheduler.CurrentTick();
    if (split_fetch_now) {
        split_tick = slot_ticks[slot]; // the copy of the batch to the host
    }
    const vk::Buffer split_fetch_handle = split_fetch.buffer, split_staging_handle = split_staging.buffer,
                     pool_handle = static_pool.buffer;
    const vk::DeviceAddress pool_address = static_pool.address;
    std::vector<std::array<u32, 2>> static_parts;
    if (static_frame == frame && static_valid && use_cache) {
        static_parts.swap(static_build); // built in this pass
    }
    const vk::DeviceSize split_fetch_bytes = vk::DeviceSize(promo_used) * 36;
    const vk::DeviceSize split_fetch_src = vk::DeviceSize(max_triangles) * 36; // the promotions' region
    probe_ticks[ring] = slot_ticks[slot];
    auto* table = static_cast<GpuGeometry*>(table_buffer.mapped) + size_t(slot) * max_geometries;
    std::memcpy(table, geometries.data(), geometries.size() * sizeof(GpuGeometry));
    vmaFlushAllocation(instance.GetAllocator(), table_buffer.allocation,
                       vk::DeviceSize(slot) * max_geometries * sizeof(GpuGeometry),
                       std::max<vk::DeviceSize>(geometries.size(), 1) * sizeof(GpuGeometry));
    const u32 num_geometries = u32(geometries.size());
    const u32 total_triangles = frame_triangles; // the per-frame BLAS
    // Instances: the per-frame BLAS (view space, when it has triangles) and the static BLAS
    // (world space).
    if (total_triangles) {
        instances.push_back(make_instance(blas_address, false));
    }
    const bool static_now = use_cache && static_valid;
    if (static_now) {
        instances.push_back(make_instance(static_blas_address, true));
    }
    const u32 num_instances = u32(instances.size());
    const vk::DeviceSize instance_offset =
        vk::DeviceSize(slot) * MaxInstances * sizeof(vk::AccelerationStructureInstanceKHR);
    std::memcpy(static_cast<u8*>(instance_buffer.mapped) + instance_offset, instances.data(),
                instances.size() * sizeof(vk::AccelerationStructureInstanceKHR));
    vmaFlushAllocation(instance.GetAllocator(), instance_buffer.allocation, instance_offset,
                       std::max<vk::DeviceSize>(instances.size(), 1) *
                           sizeof(vk::AccelerationStructureInstanceKHR));
    const bool do_readback = !readback_pending && total_triangles > 0 &&
                             (recorded_total == 1 || frame % 600 == 300);
    if (do_readback) {
        readback_pending = true;
        readback_frame = frame;
        readback_triangles = total_triangles;
        readback_tick = scheduler.CurrentTick();
    }
    // dump=: DumpFrames consecutive frames of the AO the apply gets (after the blur).
    if (settings.dump_request != dump_done && !dump_pending && settings.stage >= 4 && dump_buffer.mapped) {
        dump_pending = true;
        dump_index = 0;
        dump_frame = frame;
        dump_done = settings.dump_request;
    }
    const bool do_dump = dump_pending && dump_index < DumpFrames;
    const vk::DeviceSize dump_offset = vk::DeviceSize(dump_index) * ((w + 1) / 2) * ((h + 1) / 2) * 4;
    const vk::Buffer dump_handle = dump_buffer.buffer;
    if (recorded_total <= 2) {
        std::printf("RTAO: pass %u: proj %g %g %g %g, %u geometries, %u triangles, %ux%u\n",
                    recorded_total, cam_proj[0], cam_proj[1], cam_proj[2], cam_proj[3], num_geometries,
                    total_triangles, w, h);
        for (u32 i = 0; i < std::min<u32>(num_geometries, 4); ++i) {
            const auto& g = geometries[i];
            std::printf("RTAO:   geometry %u: store %u, %u vertices x %u instances, first %u, indices 0x%llx "
                        "count %u %s base %d restart %u, triangles %u at %u\n",
                        i, g.store, g.vertices, g.instances, g.first_vertex,
                        (unsigned long long)g.index_address, g.index_count,
                        (g.index16 & 1) ? "16-bit" : "32-bit", g.base_vertex, g.restart, g.triangles,
                        g.triangle_offset);
        }
    }
    const vk::Buffer readback_handle = readback_buffer.buffer;
    const ExpandConstants expand{
        .positions_address = instance.GetDevice().getBufferAddress({.buffer = object_motion.PositionsBuffer()}),
        .proj = camera_motion.Projection(),
        .num_geometries = num_geometries,
        .total_triangles = dispatch_triangles,
        .far_cut = settings.ao_far > 0.0f ? settings.ao_far + settings.radius : 0.0f,
        .inv_view = inv_view,
    };
    const ProbeConstants probe{
        .positions_address = expand.positions_address,
        .num_geometries = num_geometries,
        .id = probe_id,
        .proj = expand.proj,
        .inv_view = inv_view,
        .pass_index = 1,
        .num_chunks = num_chunks,
    };
    ProbeConstants probe_combine = probe;
    probe_combine.pass_index = 2;
    const vk::Buffer chunks_handle = probe_chunks.buffer, ranges_handle = probe_ranges.buffer,
                     partials_handle = probe_partials.buffer;
    const vk::DeviceSize chunks_offset = vk::DeviceSize(slot) * MaxProbeChunks * 8,
                         ranges_offset = vk::DeviceSize(slot) * max_geometries * 8;
    const vk::Buffer probe_handle = probe_buffer.buffer;
    const vk::DeviceSize probe_offset = vk::DeviceSize(ring) * max_geometries * sizeof(ProbeStats);
    const vk::DeviceSize probe_bytes = vk::DeviceSize(max_geometries) * sizeof(ProbeStats);
    const auto jitter = camera_motion.Jitter();
    const TraceConstants trace{
        .view = camera_motion.View(),
        .proj = expand.proj,
        .full_size = {float(w), float(h)},
        .half_size = {float((w + 1) / 2), float((h + 1) / 2)},
        .jitter = jitter,
        .radius = settings.radius,
        .history = history_valid ? settings.history : 0.0f,
        .rays = settings.rays,
        .frame = frame,
        .mode = settings.debug,
        .snap_far = settings.snap_far,
        .ao_far = settings.ao_far,
        .normals = settings.depth_normals ? 1u : 0u,
        .offset = settings.origin_offset,
        .denoise = v2 ? 2u : 1u,
        .subpixel = v2 && settings.subpixel ? 1u : 0u,
        .falloff = settings.falloff,
        .period = settings.noise_period,
    };
    const TemporalConstants temporal{
        .half_size = {float((w + 1) / 2), float((h + 1) / 2)},
        .sub = settings.subpixel ? std::array<s32, 2>{s32(frame & 1), s32((frame >> 1) & 1)} : std::array<s32, 2>{1, 1},
        .max_n = settings.max_n,
        .clip_k = settings.clip_k,
        .history_ok = history_valid ? 1u : 0u,
        .clip_floor = settings.clip_floor,
    };
    const vk::ImageView raw_rw = *raw_view, hist_read = *hist_views[read], hist_write = *hist_views[write];
    const vk::DeviceSize constants_offset = vk::DeviceSize(slot) * ConstantsStride;
    std::memcpy(static_cast<u8*>(constants_buffer.mapped) + constants_offset, &trace, sizeof(trace));
    vmaFlushAllocation(instance.GetAllocator(), constants_buffer.allocation, constants_offset, sizeof(trace));
    const vk::Buffer constants_handle = constants_buffer.buffer;
    const ApplyConstants apply{
        .proj = expand.proj,
        .full_size = trace.full_size,
        .half_size = trace.half_size,
        .strength = settings.strength,
        .debug = settings.debug ? 1u : 0u,
        .power = settings.power,
        .blur = settings.blur ? 1u : 0u,
        .replace = settings.replace_mask ? 1u : 0u,
    };

    const vk::Buffer table_handle = table_buffer.buffer;
    const vk::DeviceSize table_offset = vk::DeviceSize(slot) * max_geometries * sizeof(GpuGeometry);
    const vk::DeviceSize table_bytes = vk::DeviceSize(std::max<u32>(num_geometries, 1)) * sizeof(GpuGeometry);
    const vk::Buffer triangles_handle = triangle_buffer.buffer;
    const vk::DeviceSize triangles_bytes = triangle_buffer.size;
    const vk::DeviceAddress triangles_address = triangle_buffer.address;
    const vk::DeviceAddress blas_scratch_address = blas_scratch.address;
    const vk::DeviceAddress tlas_scratch_address = tlas_scratch.address;
    const vk::DeviceAddress instances_address = instance_buffer.address + instance_offset;
    const vk::ImageView ao_read = *ao_views[read], ao_write = *ao_views[write];
    const vk::ImageView blur_rw = *blur_view;
    const BlurConstants blur_h{.half_size = {float((w + 1) / 2), float((h + 1) / 2)}, .dir = {1, 0},
                               .sigma = settings.sigma, .use_n = settings.denoise == 2 && settings.debug < 2 ? 1u : 0u};
    const BlurConstants blur_v{.half_size = blur_h.half_size, .dir = {0, 1}, .sigma = settings.sigma, .use_n = blur_h.use_n};
    const vk::ImageView motion = *motion_view;

    // Five crumbs, so a GPU hang names the stage; "stage" in the settings stops after that stage.
    const u32 last_stage = settings.stage;
    const bool fast_trace = settings.build_fast_trace;
    const vk::BuildAccelerationStructureFlagsKHR extra_flags =
        data_access ? vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess
                    : vk::BuildAccelerationStructureFlagsKHR{};
    const auto memory_barrier = [](vk::CommandBuffer cmdbuf, vk::PipelineStageFlags2 src_stage,
                                   vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
                                   vk::AccessFlags2 dst) {
        const vk::MemoryBarrier2 b{.srcStageMask = src_stage,
                                   .srcAccessMask = src,
                                   .dstStageMask = dst_stage,
                                   .dstAccessMask = dst};
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
    };
    const auto info = [](vk::ImageView view) {
        return vk::DescriptorImageInfo{.imageView = view, .imageLayout = vk::ImageLayout::eGeneral};
    };
    // BB_GPU_PROFILE=1: one segment per stage.
    const auto profile = [](u64 key, const char* name) {
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(key, [name] { return std::string{name}; });
        }
    };

    // 1. Expand: the vertex shaders wrote the positions (vertex stage), the table was written by
    // the host, the index buffers by uploads: wait for all of it.
    if (last_stage < 1) {
        return;
    }
    profile(0x47A01ull, "rtao expand");
    scheduler.RecordCrumb(
        {.name = "rtao expand"},
        [=, expand_pipeline = *expand_pipeline, expand_layout = *expand_layout,
         probe_pipeline = *probe_pipeline, probe_layout = *probe_layout](vk::CommandBuffer cmdbuf) {
            // The inputs: captured positions (vertex shaders), index buffers (uploads, or game
            // compute), the table and constants (host). Not ALL_COMMANDS: no render-target writes.
            memory_barrier(cmdbuf,
                           vk::PipelineStageFlagBits2::eVertexShader | Compute |
                               vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eHost,
                           vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite |
                               vk::AccessFlagBits2::eHostWrite,
                           Compute, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
            if (dispatch_triangles) {
                const vk::DescriptorBufferInfo table_info{table_handle, table_offset, table_bytes};
                const vk::DescriptorBufferInfo tri_info{triangles_handle, 0, triangles_bytes};
                const std::array writes = {
                    vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &table_info},
                    vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &tri_info}};
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, expand_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, expand_layout, 0, writes);
                cmdbuf.pushConstants(expand_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(expand),
                                     &expand);
                cmdbuf.dispatch((dispatch_triangles + 63) / 64, 1, 1);
            }

            memory_barrier(cmdbuf, Compute, vk::AccessFlagBits2::eShaderWrite,
                           AsBuild | vk::PipelineStageFlagBits2::eTransfer,
                           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);
            if (do_readback) {
                const vk::BufferCopy region{0, 0, vk::DeviceSize(std::min(ReadbackTriangles, total_triangles)) * 9 * sizeof(float)};
                cmdbuf.copyBuffer(triangles_handle, readback_handle, region);
            }
            if (!pool_copies.empty()) {
                // Into the static pool: after the static builds recorded before (they read it).
                memory_barrier(cmdbuf, AsBuild, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer,
                               vk::AccessFlagBits2::eTransferWrite);
                for (const auto& copy : pool_copies) {
                    const vk::BufferCopy region{copy.src, copy.dst, copy.bytes};
                    cmdbuf.copyBuffer(copy.from_staging ? split_staging_handle : triangles_handle, pool_handle, region);
                }
                memory_barrier(cmdbuf, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                               AsBuild, vk::AccessFlagBits2::eShaderRead);
            }
            if (split_fetch_now) { // the world-space copies of this frame, big meshes among them
                const vk::BufferCopy region{split_fetch_src, 0, split_fetch_bytes};
                cmdbuf.copyBuffer(triangles_handle, split_fetch_handle, region);
                memory_barrier(cmdbuf, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                               vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
            }
        });

    // 1b. The static cache's measurements (its own crumb and profiler segment).
    if (num_geometries && num_chunks) {
        profile(0x47A08ull, "rtao probe");
        scheduler.RecordCrumb(
            {.name = "rtao probe"},
            [=, probe_pipeline = *probe_pipeline, probe_layout = *probe_layout](vk::CommandBuffer cmdbuf) {
                const vk::DescriptorBufferInfo table_info{table_handle, table_offset, table_bytes};
                const vk::DescriptorBufferInfo probe_info{probe_handle, probe_offset, probe_bytes};
                const vk::DescriptorBufferInfo chunks_info{chunks_handle, chunks_offset,
                                                           vk::DeviceSize(MaxProbeChunks) * 8};
                const vk::DescriptorBufferInfo partials_info{partials_handle, 0, vk::DeviceSize(MaxProbeChunks) * 48};
                const vk::DescriptorBufferInfo ranges_info{ranges_handle, ranges_offset,
                                                           vk::DeviceSize(max_geometries) * 8};
                const auto storage = [](u32 binding_index, const vk::DescriptorBufferInfo* buffer_info) {
                    return vk::WriteDescriptorSet{.dstBinding = binding_index, .descriptorCount = 1,
                                                  .descriptorType = vk::DescriptorType::eStorageBuffer,
                                                  .pBufferInfo = buffer_info};
                };
                const std::array writes = {storage(0, &table_info), storage(1, &probe_info),
                                           storage(2, &chunks_info), storage(3, &partials_info),
                                           storage(4, &ranges_info)};
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, probe_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, probe_layout, 0, writes);
                cmdbuf.pushConstants(probe_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(probe), &probe);
                cmdbuf.dispatch(num_chunks, 1, 1); // pass 1: chunks
                memory_barrier(cmdbuf, Compute, vk::AccessFlagBits2::eShaderWrite, Compute,
                               vk::AccessFlagBits2::eShaderRead);
                cmdbuf.pushConstants(probe_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(probe_combine),
                                     &probe_combine);
                cmdbuf.dispatch(num_geometries, 1, 1); // pass 2: per draw
                memory_barrier(cmdbuf, Compute, vk::AccessFlagBits2::eShaderWrite,
                               vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
            });
    }

    // 2. BLAS over the triangle list.
    if (last_stage < 2) {
        return;
    }
    // The static BLAS, when it was rebuilt in this pass (its own profiler segment: a rebuild is a
    // one-frame cost).
    if (!static_parts.empty()) {
        profile(0x47A09ull, "rtao static");
        scheduler.RecordCrumb({.name = "rtao static"}, [=, static_blas = static_blas,
                                                        static_flags = StaticFlags(settings.static_build),
                                                        static_parts = std::move(static_parts)](vk::CommandBuffer cmdbuf) {
        {
            // The last pass's trace still reads the old tree.
            memory_barrier(cmdbuf, TraceStages, vk::AccessFlagBits2::eNone, AsBuild,
                           vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
            std::vector<vk::AccelerationStructureGeometryKHR> geos(static_parts.size());
            std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(static_parts.size());
            for (size_t i = 0; i < static_parts.size(); ++i) {
                const auto [offset, count] = static_parts[i];
                geos[i] = vk::AccelerationStructureGeometryKHR{
                    .geometryType = vk::GeometryTypeKHR::eTriangles,
                    .geometry = {.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                                     .vertexFormat = vk::Format::eR32G32B32Sfloat,
                                     .vertexData = {.deviceAddress = pool_address + vk::DeviceSize(offset) * 36},
                                     .vertexStride = 3 * sizeof(float),
                                     .maxVertex = count * 3 - 1,
                                     .indexType = vk::IndexType::eNoneKHR,
                                 }},
                    .flags = vk::GeometryFlagBitsKHR::eOpaque,
                };
                ranges[i] = vk::AccelerationStructureBuildRangeInfoKHR{.primitiveCount = count};
            }
            const vk::AccelerationStructureBuildGeometryInfoKHR info{
                .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
                .flags = static_flags,
                .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
                .dstAccelerationStructure = static_blas,
                .geometryCount = u32(geos.size()),
                .pGeometries = geos.data(),
                .scratchData = {.deviceAddress = blas_scratch_address},
            };
            const vk::AccelerationStructureBuildRangeInfoKHR* range_ptr = ranges.data();
            cmdbuf.buildAccelerationStructuresKHR(1, &info, &range_ptr);
            // The scratch is the per-frame build's next.
            memory_barrier(cmdbuf, AsBuild, vk::AccessFlagBits2::eAccelerationStructureWriteKHR, AsBuild,
                           vk::AccessFlagBits2::eAccelerationStructureReadKHR |
                               vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
        }
        });
    }
    profile(0x47A02ull, "rtao blas");
    scheduler.RecordCrumb({.name = "rtao blas"}, [=, blas = blas](vk::CommandBuffer cmdbuf) {
        if (!total_triangles) {
            memory_barrier(cmdbuf, AsBuild, vk::AccessFlagBits2::eAccelerationStructureWriteKHR, AsBuild,
                           vk::AccessFlagBits2::eAccelerationStructureReadKHR);
            return;
        }
        vk::AccelerationStructureGeometryKHR blas_geometry{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = {.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                             .vertexFormat = vk::Format::eR32G32B32Sfloat,
                             .vertexData = {.deviceAddress = triangles_address},
                             .vertexStride = 3 * sizeof(float),
                             .maxVertex = std::max<u32>(total_triangles * 3, 1) - 1,
                             .indexType = vk::IndexType::eNoneKHR,
                         }},
            .flags = vk::GeometryFlagBitsKHR::eOpaque,
        };
        const vk::AccelerationStructureBuildGeometryInfoKHR blas_info{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags = (fast_trace ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                                 : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild) |
                     extra_flags,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .dstAccelerationStructure = blas,
            .geometryCount = 1,
            .pGeometries = &blas_geometry,
            .scratchData = {.deviceAddress = blas_scratch_address},
        };
        const vk::AccelerationStructureBuildRangeInfoKHR blas_range{.primitiveCount = total_triangles};
        cmdbuf.buildAccelerationStructuresKHR(blas_info, &blas_range);
        memory_barrier(cmdbuf, AsBuild, vk::AccessFlagBits2::eAccelerationStructureWriteKHR, AsBuild,
                       vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    });

    // 3. TLAS: the one instance.
    if (last_stage < 3) {
        return;
    }
    profile(0x47A03ull, "rtao tlas");
    scheduler.RecordCrumb({.name = "rtao tlas"}, [=, tlas = tlas](vk::CommandBuffer cmdbuf) {
        vk::AccelerationStructureGeometryKHR tlas_geometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = {.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                             .arrayOfPointers = false,
                             .data = {.deviceAddress = instances_address},
                         }},
        };
        const vk::AccelerationStructureBuildGeometryInfoKHR tlas_info{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .dstAccelerationStructure = tlas,
            .geometryCount = 1,
            .pGeometries = &tlas_geometry,
            .scratchData = {.deviceAddress = tlas_scratch_address},
        };
        const vk::AccelerationStructureBuildRangeInfoKHR tlas_range{.primitiveCount = num_instances};
        cmdbuf.buildAccelerationStructuresKHR(tlas_info, &tlas_range);
        memory_barrier(cmdbuf, AsBuild, vk::AccessFlagBits2::eAccelerationStructureWriteKHR, TraceStages,
                       vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    });

    // 4. Trace at half resolution.
    if (last_stage < 4) {
        return;
    }
    profile(0x47A04ull, "rtao trace");
    scheduler.RecordCrumb(
        {.name = "rtao trace"},
        [=, tlas = tlas, trace_pipeline = *trace_pipeline, trace_layout = *trace_layout,
         rt = settings.use_rt_pipeline && rt_trace_pipeline ? *rt_trace_pipeline : vk::Pipeline{},
         sbt_rgen = sbt_rgen, sbt_miss = sbt_miss, sbt_hit = sbt_hit,
         ao_image = vk::Image(v2 ? raw_image : ao_images[write])](vk::CommandBuffer cmdbuf) {
            const auto d = info(depth_view), n = info(normal_view), m = info(motion), p = info(ao_read),
                       o = info(v2 ? raw_rw : ao_write);
            const vk::WriteDescriptorSetAccelerationStructureKHR as_write{
                .accelerationStructureCount = 1, .pAccelerationStructures = &tlas};
            const vk::DescriptorBufferInfo constants_info{constants_handle, constants_offset, sizeof(trace)};
            const std::array writes = {
                vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &n},
                vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m},
                vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &p},
                vk::WriteDescriptorSet{.dstBinding = 4, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &o},
                vk::WriteDescriptorSet{.pNext = &as_write, .dstBinding = 5, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
                vk::WriteDescriptorSet{.dstBinding = 6, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &constants_info}};
            if (rt) {
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, rt);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eRayTracingKHR, trace_layout, 0, writes);
                cmdbuf.traceRaysKHR(sbt_rgen, sbt_miss, sbt_hit, vk::StridedDeviceAddressRegionKHR{},
                                    (w + 1) / 2, (h + 1) / 2, 1);
            } else {
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, trace_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, trace_layout, 0, writes);
                cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 3) / 4, 1); // 8x4 groups
            }
            const vk::ImageMemoryBarrier2 ao_barrier{
                .srcStageMask = TraceStages,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = Compute | vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = ao_image,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ao_barrier});
        });

    // 4a. Denoiser 2: temporal accumulation into the history and the AO image.
    if (v2) {
        profile(0x47A07ull, "rtao temporal");
        scheduler.RecordCrumb(
            {.name = "rtao temporal"},
            [=, temporal_pipeline = *temporal_pipeline, temporal_layout = *temporal_layout,
             ao_image = vk::Image(ao_images[write]), hist_image = vk::Image(hist_images[write])](
                vk::CommandBuffer cmdbuf) {
                const auto r = info(raw_rw), hr = info(hist_read), m = info(motion), hw = info(hist_write),
                           a = info(ao_write);
                const std::array writes = {
                    vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &r},
                    vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &hr},
                    vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m},
                    vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &hw},
                    vk::WriteDescriptorSet{.dstBinding = 4, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &a}};
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, temporal_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, temporal_layout, 0, writes);
                cmdbuf.pushConstants(temporal_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(temporal),
                                     &temporal);
                cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 7) / 8, 1);
                std::array<vk::ImageMemoryBarrier2, 2> barriers;
                for (int i = 0; i < 2; ++i) {
                    barriers[i] = vk::ImageMemoryBarrier2{
                        .srcStageMask = Compute,
                        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                        .dstStageMask = Compute | vk::PipelineStageFlagBits2::eTransfer,
                        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
                        .oldLayout = vk::ImageLayout::eGeneral,
                        .newLayout = vk::ImageLayout::eGeneral,
                        .image = i == 0 ? ao_image : hist_image,
                        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                    };
                }
                cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = barriers.data()});
            });
    }

    ao_index = write;
    history_valid = true;

    // 4b. Separable depth-aware blur of the AO image (in place through blur_image).
    if (settings.sigma > 0.0f && settings.debug < 2) { // the debug views stay unblurred
        profile(0x47A06ull, "rtao blur");
        scheduler.RecordCrumb(
            {.name = "rtao blur"},
            [=, blur_pipeline = *blur_pipeline, blur_layout = *blur_layout, ao_image = vk::Image(ao_images[write]),
             blur_img = vk::Image(blur_image)](vk::CommandBuffer cmdbuf) {
                const auto barrier = [&](vk::Image image) {
                    const vk::ImageMemoryBarrier2 b{
                        .srcStageMask = Compute,
                        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                        .dstStageMask = Compute | vk::PipelineStageFlagBits2::eTransfer, // + dump copy
                        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
                        .oldLayout = vk::ImageLayout::eGeneral,
                        .newLayout = vk::ImageLayout::eGeneral,
                        .image = image,
                        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                    };
                    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
                };
                const auto pass = [&](vk::ImageView src, vk::ImageView dst, const BlurConstants& constants) {
                    // Denoiser 2: the history's frame count steers the width (else any image).
                    const auto si = info(src), di = info(dst), ni = info(v2 ? hist_write : ao_read);
                    const std::array writes = {
                        vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &si},
                        vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &di},
                        vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &ni}};
                    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, blur_pipeline);
                    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, blur_layout, 0, writes);
                    cmdbuf.pushConstants(blur_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants),
                                         &constants);
                    cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 7) / 8, 1);
                };
                pass(ao_write, blur_rw, blur_h);
                barrier(blur_img);
                pass(blur_rw, ao_write, blur_v);
                barrier(ao_image);
            });
    }

    // The AO the apply gets, for dump= (DumpFrames consecutive frames).
    if (do_dump) {
        scheduler.Record([=, ao_image = vk::Image(ao_images[write])](vk::CommandBuffer cmdbuf) {
            const vk::BufferImageCopy region{
                .bufferOffset = dump_offset,
                .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .imageExtent = {(w + 1) / 2, (h + 1) / 2, 1},
            };
            cmdbuf.copyImageToBuffer(ao_image, vk::ImageLayout::eGeneral, dump_handle, region);
        });
        if (++dump_index == DumpFrames) {
            dump_tick = scheduler.CurrentTick();
        }
    }

    // 5. Into the game's mask.
    if (last_stage < 5) {
        return;
    }
    profile(0x47A05ull, "rtao apply");
    scheduler.RecordCrumb(
        {.name = "rtao apply"},
        [=, apply_pipeline = *apply_pipeline, apply_layout = *apply_layout](vk::CommandBuffer cmdbuf) {
            const auto a = info(ao_write), d = info(depth_view), mk = info(mask_view);
            const std::array writes = {
                vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &a},
                vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &mk}};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, apply_pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, apply_layout, 0, writes);
            cmdbuf.pushConstants(apply_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(apply), &apply);
            cmdbuf.dispatch((w + 7) / 8, (h + 7) / 8, 1);
        });
}

} // namespace Vulkan
