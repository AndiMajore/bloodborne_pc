// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ray-traced AO, debug output: the dump= frames and the expanded-triangle readback (see vk_rtao.cpp).
#include "video_core/renderer_vulkan/vk_rtao.h"
#include "video_core/renderer_vulkan/vk_rtao_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {
using namespace RtaoInternal;

void RayTracedAO::WriteDump() {
    if (!(dump_pending && dump_index == DumpFrames && dump_buffer.mapped && scheduler.IsFree(dump_tick))) {
        return;
    }
        dump_pending = false;
        vmaInvalidateAllocation(instance.GetAllocator(), dump_buffer.allocation, 0, VK_WHOLE_SIZE);
        const u32 dw = (width + 1) / 2, dh = (height + 1) / 2;
        const size_t texels_per_frame = size_t(dw) * dh;
        const auto* texels = static_cast<const u16*>(dump_buffer.mapped);
        const auto half_to_float = [](u16 hf) {
            const u32 sign = (hf >> 15) & 1, exp = (hf >> 10) & 0x1f, mant = hf & 0x3ff;
            float value;
            if (exp == 0) value = std::ldexp(float(mant), -24);
            else if (exp == 31) value = mant ? NAN : INFINITY;
            else value = std::ldexp(float(mant | 0x400), int(exp) - 25);
            return sign ? -value : value;
        };
        std::vector<float> prev(texels_per_frame), cur(texels_per_frame);
        double mean = 0.0, change = 0.0;
        u64 changed = 0, zeros = 0;
        for (u32 k = 0; k < DumpFrames; ++k) {
            const u16* frame_texels = texels + k * texels_per_frame * 2;
            for (size_t i = 0; i < texels_per_frame; ++i) {
                const float r = half_to_float(frame_texels[i * 2]);
                cur[i] = std::isfinite(r) ? std::clamp(r, 0.0f, 2.0f) : 0.0f;
                mean += cur[i];
                zeros += cur[i] == 0.0f;
                if (k) {
                    const float d = std::abs(cur[i] - prev[i]);
                    change += d;
                    changed += d > 0.05f;
                }
            }
            const std::string path =
                "out/rtao_dump_" + std::to_string(dump_frame) + "_" + std::to_string(k) + ".pgm";
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fprintf(f, "P5\n%u %u\n255\n", dw, dh);
                std::vector<u8> row(dw);
                for (u32 y = 0; y < dh; ++y) {
                    for (u32 x = 0; x < dw; ++x) {
                        row[x] = u8(cur[size_t(y) * dw + x] * 127.5f + 0.5f); // 1.0 -> 128
                    }
                    std::fwrite(row.data(), 1, dw, f);
                }
                std::fclose(f);
            }
            prev.swap(cur);
        }
        const double all = double(texels_per_frame) * DumpFrames, pairs = double(texels_per_frame) * (DumpFrames - 1);
        std::printf("RTAO: dumped out/rtao_dump_%u_0..%u.pgm (%u consecutive frames, %ux%u, 1.0 = 128): mean %.3f, "
                    "%.1f%% zero; frame to frame: mean |change| %.4f, %.2f%% of pixels change > 0.05\n",
                    dump_frame, DumpFrames - 1, DumpFrames, dw, dh, mean / all, 100.0 * zeros / all,
                    change / pairs, 100.0 * changed / pairs);
}

void RayTracedAO::ReportReadback() {
    if (!(readback_pending && scheduler.IsFree(readback_tick))) {
        return;
    }
        // Copied 120 frames ago: long finished. How do the expanded triangles look?
        readback_pending = false;
        vmaInvalidateAllocation(instance.GetAllocator(), readback_buffer.allocation, 0, VK_WHOLE_SIZE);
        const float* v = static_cast<const float*>(readback_buffer.mapped);
        u32 valid = 0, finite = 0;
        const u32 count = std::min<u32>(ReadbackTriangles, readback_triangles);
        float zmin = 1e30f, zmax = -1e30f, xymax = 0.0f;
        for (u32 i = 0; i < count; ++i) {
            const float* tri = v + i * 9;
            bool nonzero = false, all_finite = true;
            for (u32 k = 0; k < 9; ++k) {
                nonzero |= tri[k] != 0.0f;
                all_finite &= std::isfinite(tri[k]);
            }
            if (!nonzero) continue;
            ++valid;
            if (!all_finite) continue;
            ++finite;
            for (u32 k = 0; k < 3; ++k) {
                zmin = std::min(zmin, tri[k * 3 + 2]);
                zmax = std::max(zmax, tri[k * 3 + 2]);
                xymax = std::max({xymax, std::abs(tri[k * 3]), std::abs(tri[k * 3 + 1])});
            }
        }
        std::printf("RTAO: readback (frame %u): of the first %u triangles %u non-zero, %u valid (finite), "
                    "%u invalid (NaN); z %.2f..%.2f, |xy| up to %.2f; first: (%.2f %.2f %.2f) (%.2f %.2f %.2f) "
                    "(%.2f %.2f %.2f)\n",
                    readback_frame, count, valid, finite, valid - finite, zmin, zmax, xymax, v[0], v[1], v[2],
                    v[3], v[4], v[5], v[6], v[7], v[8]);
}

} // namespace Vulkan
