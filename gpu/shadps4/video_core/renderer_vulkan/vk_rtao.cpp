// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_rtao.h"
#include "video_core/renderer_vulkan/rtao_bluenoise.h"
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
    if (!object_motion.Enabled() || !camera_motion.Enabled()) {
        std::puts("RTAO: needs the object motion vectors (menu: object motion, with an upscaler), off");
        return;
    }
    if (!instance.IsRayQuerySupported() || !ObjectMotion::RtaoCapture()) {
        std::puts("RTAO: ray queries unsupported by the driver, off");
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
                                       trace_binding(6, DT::eUniformBuffer), trace_binding(7, DT::eSampledImage)};
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
                         &static_pool, &static_blas_buffer, &noise_staging, &constants_buffer, &sbt_buffer, &index_ring, &probe_chunks,
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
    if (!noise_uploaded) { // the blue-noise texture, once (rtao_bluenoise.h)
        noise_uploaded = true;
        scheduler.Record([image = vk::Image(noise_image), staging = noise_staging.buffer](vk::CommandBuffer cmdbuf) {
            const auto to = [&](vk::ImageLayout from_layout, vk::ImageLayout layout, vk::PipelineStageFlags2 src,
                                vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst, vk::AccessFlags2 dst_access) {
                const vk::ImageMemoryBarrier2 b{.srcStageMask = src, .srcAccessMask = src_access, .dstStageMask = dst,
                                                .dstAccessMask = dst_access, .oldLayout = from_layout, .newLayout = layout,
                                                .image = image,
                                                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
                cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
            };
            to(vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits2::eNone,
               vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
            const vk::BufferImageCopy region{.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                             .imageExtent = {Rtao::BlueNoiseSize, Rtao::BlueNoiseSize, 1}};
            cmdbuf.copyBufferToImage(staging, image, vk::ImageLayout::eTransferDstOptimal, region);
            to(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eTransfer,
               vk::AccessFlagBits2::eTransferWrite, TraceStages, vk::AccessFlagBits2::eShaderRead);
        });
    }
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
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
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
    FrameDraws classified;
    ClassifyDraws(classified, use_cache, probe_id);
    auto& dynamic_list = classified.dynamic_list;
    auto& verify_list = classified.verify_list;
    auto& promo_list = classified.promo_list;
    auto& dynamic_keys = classified.dynamic_keys;
    auto& verify_keys = classified.verify_keys;
    auto& pool_copies = classified.pool_copies;
    const u32 promo_used = classified.promo_used;
    const bool split_fetch_now = classified.split_fetch_now;
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
        .noise = settings.blue_noise ? 1u : 0u,
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

    PassInputs in;
    in.ao_read = ao_read;
    in.ao_write = ao_write;
    in.blur_rw = blur_rw;
    in.depth_view = depth_view;
    in.hist_read = hist_read;
    in.hist_write = hist_write;
    in.normal_view = normal_view;
    in.mask_view = mask_view;
    in.motion = motion;
    in.raw_rw = raw_rw;
    in.w = w;
    in.h = h;
    in.write = write;
    in.dispatch_triangles = dispatch_triangles;
    in.num_chunks = num_chunks;
    in.num_geometries = num_geometries;
    in.num_instances = num_instances;
    in.total_triangles = total_triangles;
    in.do_dump = do_dump;
    in.do_readback = do_readback;
    in.split_fetch_now = split_fetch_now;
    in.v2 = v2;
    in.chunks_handle = chunks_handle;
    in.ranges_handle = ranges_handle;
    in.partials_handle = partials_handle;
    in.constants_handle = constants_handle;
    in.dump_handle = dump_handle;
    in.probe_handle = probe_handle;
    in.readback_handle = readback_handle;
    in.split_fetch_handle = split_fetch_handle;
    in.split_staging_handle = split_staging_handle;
    in.pool_handle = pool_handle;
    in.table_handle = table_handle;
    in.triangles_handle = triangles_handle;
    in.chunks_offset = chunks_offset;
    in.ranges_offset = ranges_offset;
    in.constants_offset = constants_offset;
    in.dump_offset = dump_offset;
    in.probe_bytes = probe_bytes;
    in.probe_offset = probe_offset;
    in.split_fetch_bytes = split_fetch_bytes;
    in.split_fetch_src = split_fetch_src;
    in.table_bytes = table_bytes;
    in.table_offset = table_offset;
    in.triangles_bytes = triangles_bytes;
    in.blas_scratch_address = blas_scratch_address;
    in.tlas_scratch_address = tlas_scratch_address;
    in.instances_address = instances_address;
    in.pool_address = pool_address;
    in.triangles_address = triangles_address;
    in.apply = apply;
    in.blur_h = blur_h;
    in.blur_v = blur_v;
    in.expand = expand;
    in.probe = probe;
    in.probe_combine = probe_combine;
    in.temporal = temporal;
    in.trace = trace;
    in.pool_copies = std::move(pool_copies);
    in.static_parts = std::move(static_parts);
    RecordPasses(in);
}

} // namespace Vulkan
