// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/rt_status.h"
#include "render/vk_rtao.h"
#include "render/vk_rtao_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <iterator>

#include <vk_mem_alloc.h>

#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "buffers/buffer_cache.h"
#include "core/memory.h"
#include "upscale/vk_object_motion.h"
#include "vulkan/platform.h"
#include "render/vk_gpu_profiler.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"

namespace Vulkan {

using namespace RtaoInternal;

void RayTracedAO::OnFrameStart() {
    ReportStatus();
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
            RtStatus::Get().rtao_off = "its acceleration structures could not be made";
            enabled = false;
            return;
        }
        resources_ready = true;
    }
    static u32 reports = 0;
    WriteDump();
    ReportReadback();
    ReportExtents();
    if (frame % 600 == 0 && frame) {
        std::printf("RTAO: frame %u: %zu draws, %u triangles (%u draws, %u triangles dropped: budget %u; "
                    "%u strips/other, %u alpha-tested, %u without depth write, %u without index address "
                    "skipped), %u passes recorded in 600 frames, %u skipped (ring slot still in use)\n",
                    frame, geometries.size(), frame_triangles, dropped_draws, dropped_triangles,
                    max_triangles, skipped_topology, skipped_alpha, skipped_nowrite, skipped_no_address,
                    recorded, skipped_busy);
        if (settings.foliage > 0.0f) {
            std::printf("RTAO: foliage %.2f: %u triangles (%u draws over its budget of %u)\n", settings.foliage,
                        foliage_triangles, foliage_dropped, FoliageMax());
        }
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
        std::printf("RTAO: last pass's draws by G-buffer pass (first, second, more): %u / %u / %u draws, "
                    "%u / %u / %u triangles, the camera's pass %d; %u small-skeleton draws (%u triangles); "
                    "left out: %u triangles of other passes (capture_pass=%u), %u of small skeletons "
                    "(skip_weapons=%d, cache_weapons=%d)\n",
                    pass_draws[0], pass_draws[1], pass_draws[2], pass_triangles[0], pass_triangles[1],
                    pass_triangles[2], main_pass == ~0u ? -1 : int(main_pass + 1), small_draws, small_triangles,
                    side_skipped_triangles, settings.capture_pass, weapon_skipped_triangles,
                    int(settings.skip_weapons), int(settings.cache_weapons));
        std::printf("RTAO: memory: fixed %llu MiB (the static pool and BLAS included)\n",
                    (unsigned long long)(static_bytes >> 20));
        promotions = 0;
        evictions_total = 0;
        if (recorded == 0 && !draws.empty() && settings.active && !upsample_seen) {
            pass_missing = true; // no capture for a pass that never runs
        }
        if (recorded == 0 && !draws.empty() && settings.active && !missing_pass_reported) {
            missing_pass_reported = true;
            std::puts("RTAO: OFF in practice: the game's AO pass never ran (neither its upsample shader "
                      "c45c51cb nor its half-size shader cbbca5dc), so there is no AO mask to write into. "
                      "The \"Disable SSAO\" game patch (launcher: Game effects, SSAO unchecked) and some "
                      "mods (bb_enhanced) remove it; RTAO needs that pass. (Reported once.)");
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
    gbuffer_passes = 0;
    pass_depth = {};
    index_ring_state = -1;
    index_ring_used = 0;
    dropped_triangles = 0;
    dropped_draws = 0;
    skipped_topology = 0;
    skipped_no_address = 0;
    skipped_alpha = 0;
    foliage_dropped = 0;
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

/// "RT status:" (ticket 89, render/rt_status.h): every 600 frames in the world, the effects asked
/// for, each running with its count or OFF with the reason. RTAO off for the whole run (driver,
/// memory): said once. Nothing while RTAO is not asked for, nor in menus and loading.
void RayTracedAO::ReportStatus() {
    if (!Rtao::Requested()) {
        return;
    }
    auto& st = RtStatus::Get();
    if (const char* off = st.rtao_off.load(); off && !enabled) {
        if (!status_off_reported) {
            status_off_reported = true;
            std::printf("RT status: RTAO OFF (%s): no ray-traced effect runs\n", off);
        }
        return;
    }
    status_world += draws.empty() ? 0u : 1u; // the previous frame's G-buffer draws
    if (++status_frames < 600) {
        return;
    }
    const u32 world = std::exchange(status_world, 0u);
    status_frames = 0;
    const u32 ao = st.rtao_passes.exchange(0), gather = st.gather_rt.exchange(0), march = st.gather_march.exchange(0);
    const u32 sun_on = st.sun_wanted.exchange(0), sun = st.sun_traced.exchange(0);
    const u32 reflect = st.reflect_passes.exchange(0), probes = st.probe_passes.exchange(0);
    if (world < 60) {
        return; // menus, loading screens: nothing to say
    }
    std::string line = "RT status: ";
    char part[200];
    if (ao > 0) {
        std::snprintf(part, sizeof(part), "RTAO on (%u of %u frames)", ao, world);
    } else if (!upsample_seen) {
        std::snprintf(part, sizeof(part), "RTAO OFF (the game's AO pass never ran: SSAO patched out, or a mod "
                                          "removes it)");
    } else {
        std::snprintf(part, sizeof(part), "RTAO OFF (no pass in %u frames)", world);
    }
    line += part;
    const bool scene = ao > 0;
    if (gather + march > 0) {
        if (gather > 0) {
            std::snprintf(part, sizeof(part), "; RT bounce on (%u of %u gathers)", gather, gather + march);
        } else {
            std::snprintf(part, sizeof(part), "; RT bounce OFF (%s)", scene ? "fell back to the screen march"
                                                                             : "no RTAO scene");
        }
        line += part;
    }
    if (sun_on > 0) {
        if (sun > 0) {
            std::snprintf(part, sizeof(part), "; moon shadow on (%u of %u frames)", sun, sun_on);
        } else {
            std::snprintf(part, sizeof(part), "; moon shadow OFF (%s)", scene ? "not traced" : "no RTAO scene");
        }
        line += part;
    }
    if (const char* off = st.reflect_off.load()) {
        line += std::string("; reflections OFF (") + off + ")";
    } else if (st.reflect_wanted.load()) {
        if (reflect > 0) {
            std::snprintf(part, sizeof(part), "; reflections on (%u passes)", reflect);
            line += part;
        } else {
            line += "; reflections OFF (the game's SSR pass did not run: turn on SSR)";
        }
    }
    if (const char* off = st.probes_off.load()) {
        line += std::string("; probes OFF (") + off + ")";
    } else if (st.probes_wanted.load()) {
        if (probes > 0) {
            std::snprintf(part, sizeof(part), "; probes on (%u passes)", probes);
            line += part;
        } else {
            line += "; probes OFF (no RTAO scene)";
        }
    }
    std::printf("%s\n", line.c_str());
}

void RayTracedAO::OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal) {
    if (depth != pass_depth) { // a new G-buffer pass (a planar reflection draws its own first)
        pass_depth = depth;
        ++gbuffer_passes;
    }
    depth_id = depth;
    normal_id = normal;
}

void RayTracedAO::OnColorTarget(VideoCore::ImageId color, u64 ps_hash) {
    // Which gate the game's AO pass meets (ticket 86): once, the first times its shader comes by.
    static int ao_pass_reports = 0;
    if (IsAoShader(ps_hash) && ao_pass_reports < 3) {
        ++ao_pass_reports;
        std::printf("RTAO: the AO shader 0x%llx at a pass: enabled %d, resources %d, color %d, "
                    "depth %d, candidate %d, active %d\n", (unsigned long long)ps_hash, int(enabled),
                    int(resources_ready), int(bool(color)), int(bool(depth_id)), int(bool(mask_candidate)),
                    int(settings.active));
    }
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
    if (test_move && IsAoShader(ps_hash) && Active()) {
        test_move_pending = true;
    }
    if (color != last_color) {
        // The previous pass ended. Was it the game's AO pass (the R8 mask or the half-size RGBA8)?
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
    if (IsAoShader(ps_hash)) {
        pass_missing = false;
        const bool half = ps_hash == AoHalfShader;
        if (!upsample_seen || (half && !half_path_seen)) {
            // Keep the wording: playtests look for "the game's AO pass found by its upsample shader".
            std::printf("RTAO: the game's AO pass found by its upsample shader 0x%llx (%s)\n",
                        (unsigned long long)ps_hash,
                        half ? "the half-size path, RGBA8: the unmodded game" : "the temporal path, R8 scene size");
        }
        upsample_seen = true;
        half_path_seen |= half;
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
    // The game's AO target: R8 of the scene size, or (the half-size path, 87) RGBA8 of half the
    // depth's size rounded up, its red channel the AO. Anything else is left alone.
    const bool mask_half = mask.info.pixel_format == vk::Format::eR8G8B8A8Unorm;
    const u32 mask_scale = mask_half ? 2u : 1u;
    const u32 mask_w = (w + mask_scale - 1) / mask_scale, mask_h = (h + mask_scale - 1) / mask_scale;
    if ((mask.info.pixel_format != vk::Format::eR8Unorm && !mask_half) ||
        !(mask.usage_flags & vk::ImageUsageFlagBits::eStorage) || mask.info.size.width < mask_w ||
        mask.info.size.height < mask_h) {
        if (!mask_unusable_reported) {
            mask_unusable_reported = true;
            std::printf("RTAO: the game's AO target is %s %ux%u%s (depth %ux%u): not written\n",
                        vk::to_string(mask.info.pixel_format).c_str(), mask.info.size.width,
                        mask.info.size.height,
                        (mask.usage_flags & vk::ImageUsageFlagBits::eStorage) ? "" : " without storage use", w, h);
        }
        return;
    }
    const auto mask_view = CachedView(mask, mask.info.pixel_format, vk::ImageAspectFlagBits::eColor);

    // The history pair swaps once the trace is recorded (the stage limiter may stop before it).
    const u32 read = ao_index, write = ao_index ^ 1;
    RecordImageSetup(depth, normal, mask, depth_view, read, write, w, h);
    const bool v2 = settings.denoise == 2 && settings.debug < 2; // denoiser 2; the debug views keep the plain path

    const auto cam_proj = camera_motion.Projection();
    if (cam_proj[0] == 0.0f || cam_proj[1] == 0.0f || cam_proj[3] == 0.0f) {
        return; // no camera this frame (menus): nothing sensible to build
    }
    ++recorded;
    ++recorded_total;
    ++RtStatus::Get().rtao_passes;

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
            .mask = world ? 0x02u : 0x01u, // 1 per-frame, 2 world-space (cached); 0xFF hits both
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
    auto& pool_copies = classified.pool_copies;
    const u32 promo_used = classified.promo_used;
    const bool split_fetch_now = classified.split_fetch_now;
    u32 dispatch_triangles = 0;
    const u32 num_chunks = BuildTable(classified, slot, ring, dispatch_triangles);
    probe_ids[ring] = probe_id;
    if (index_ring_used) {
        vmaFlushAllocation(instance.GetAllocator(), index_ring.allocation,
                           vk::DeviceSize(slot) * IndexRingSlot, index_ring_used);
    }
    recorded_this_frame = true;
    tlas_frame = settings.stage >= 3 ? frame : ~0u;
    // This pass's commands go out with the current tick: the slots are busy until it is reached.
    slot_ticks[slot] = scheduler.CurrentTick();
    if (split_fetch_now) {
        split_tick = slot_ticks[slot]; // the copy of the batch to the host
    }
    PassInputs in;
    in.w = w;
    in.h = h;
    in.write = write;
    in.v2 = v2;
    in.depth_view = depth_view;
    in.normal_view = normal_view;
    in.mask_view = mask_view;
    in.mask_half = mask_half;
    in.mask_w = mask_w;
    in.mask_h = mask_h;
    in.split_fetch_now = split_fetch_now;
    in.split_fetch_handle = split_fetch.buffer;
    in.split_staging_handle = split_staging.buffer;
    in.pool_handle = static_pool.buffer;
    in.pool_address = static_pool.address;
    if (static_frame == frame && static_valid && use_cache) {
        in.static_parts.swap(static_build); // built in this pass
        SurfaceTable(in);
    }
    in.split_fetch_bytes = vk::DeviceSize(promo_used) * 36;
    in.split_fetch_src = vk::DeviceSize(max_triangles) * 36; // the promotions' region
    probe_ticks[ring] = slot_ticks[slot];
    auto* table = static_cast<GpuGeometry*>(table_buffer.mapped) + size_t(slot) * max_geometries;
    std::memcpy(table, geometries.data(), geometries.size() * sizeof(GpuGeometry));
    vmaFlushAllocation(instance.GetAllocator(), table_buffer.allocation,
                       vk::DeviceSize(slot) * max_geometries * sizeof(GpuGeometry),
                       std::max<vk::DeviceSize>(geometries.size(), 1) * sizeof(GpuGeometry));
    in.num_geometries = u32(geometries.size());
    in.total_triangles = frame_triangles; // the per-frame BLAS
    in.opaque_triangles = std::min(classified.opaque_triangles, frame_triangles);
    in.dispatch_triangles = dispatch_triangles;
    in.num_chunks = num_chunks;
    // Instances: the per-frame BLAS (view space, when it has triangles) and the static BLAS
    // (world space).
    if (in.total_triangles) {
        instances.push_back(make_instance(blas_address, false));
    }
    const bool static_now = use_cache && static_valid;
    static_in_frame = static_now;
    if (static_now) {
        instances.push_back(make_instance(static_blas_address, true));
    }
    in.num_instances = u32(instances.size());
    const vk::DeviceSize instance_offset =
        vk::DeviceSize(slot) * MaxInstances * sizeof(vk::AccelerationStructureInstanceKHR);
    std::memcpy(static_cast<u8*>(instance_buffer.mapped) + instance_offset, instances.data(),
                instances.size() * sizeof(vk::AccelerationStructureInstanceKHR));
    vmaFlushAllocation(instance.GetAllocator(), instance_buffer.allocation, instance_offset,
                       std::max<vk::DeviceSize>(instances.size(), 1) *
                           sizeof(vk::AccelerationStructureInstanceKHR));
    in.instances_address = instance_buffer.address + instance_offset;
    in.do_readback = !readback_pending && in.total_triangles > 0 &&
                     (recorded_total == 1 || frame % 600 == 300);
    if (in.do_readback) {
        readback_pending = true;
        readback_frame = frame;
        readback_triangles = in.total_triangles;
        readback_tick = scheduler.CurrentTick();
    }
    // dump=: DumpFrames consecutive frames of the AO the apply gets (after the blur).
    if (settings.dump_request != dump_done && !dump_pending && settings.stage >= 4 && dump_buffer.mapped) {
        dump_pending = true;
        dump_index = 0;
        dump_frame = frame;
        dump_done = settings.dump_request;
    }
    in.do_dump = dump_pending && dump_index < DumpFrames;
    in.dump_offset = vk::DeviceSize(dump_index) * ((w + 1) / 2) * ((h + 1) / 2) * 4;
    in.dump_handle = dump_buffer.buffer;
    if (recorded_total <= 2) {
        LogFirstPasses(in.num_geometries, in.total_triangles, w, h);
    }
    in.readback_handle = readback_buffer.buffer;
    in.chunks_handle = probe_chunks.buffer;
    in.ranges_handle = probe_ranges.buffer;
    in.partials_handle = probe_partials.buffer;
    in.chunks_offset = vk::DeviceSize(slot) * MaxProbeChunks * 8;
    in.ranges_offset = vk::DeviceSize(slot) * max_geometries * 8;
    in.probe_handle = probe_buffer.buffer;
    in.probe_offset = vk::DeviceSize(ring) * max_geometries * sizeof(ProbeStats);
    in.probe_bytes = vk::DeviceSize(max_geometries) * sizeof(ProbeStats);
    in.table_handle = table_buffer.buffer;
    in.table_offset = vk::DeviceSize(slot) * max_geometries * sizeof(GpuGeometry);
    in.table_bytes = vk::DeviceSize(std::max<u32>(in.num_geometries, 1)) * sizeof(GpuGeometry);
    in.triangles_handle = triangle_buffer.buffer;
    in.triangles_bytes = triangle_buffer.size;
    in.triangles_address = triangle_buffer.address;
    in.blas_scratch_address = blas_scratch.address;
    in.tlas_scratch_address = tlas_scratch.address;
    in.pool_copies = std::move(pool_copies);
    in.surface_cache_handle = settings.surface ? surface_cache.buffer : vk::Buffer{};
    SampleExtents(in, classified);
    FillConstants(in, slot, read, probe_id);
    RecordPasses(in);
}

} // namespace Vulkan
