// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ray-traced AO, the static cache's classification of a frame's draws (see vk_rtao.cpp and
// docs/rtao.md): per-frame geometry, draws covered by the static BLAS, new pool entries, the
// static BLAS's rebuild and the split of big meshes.
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

void RayTracedAO::ClassifyDraws(FrameDraws& out, bool use_cache, u32 probe_id) {
    // The results under the names the classification uses.
    auto& dynamic_list = out.dynamic_list;
    auto& verify_list = out.verify_list;
    auto& promo_list = out.promo_list;
    auto& dynamic_keys = out.dynamic_keys;
    auto& verify_keys = out.verify_keys;
    auto& pool_copies = out.pool_copies;
    u32& promo_used = out.promo_used;
    const float cull = settings.ao_far > 0.0f ? settings.ao_far + settings.radius : 0.0f;
    const auto device = instance.GetDevice();
    frame_triangles = 0;
    per_frame_reasons = {};
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
    out.split_fetch_now = split_fetch_now;
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
}

} // namespace Vulkan
