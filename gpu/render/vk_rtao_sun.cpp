// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the sun's shadow by ray tracing (experimental, sun_shadow= in rtao.txt; vk_rtao.h,
// rt_sun_shadow.comp). The game's sun shadow chain (Central Yharnam, ticket 59): the cascaded
// shadow map resolved into a screen mask (ps f55643de), blurred (11ffdfb5), then written with the
// AO into the final mask (a291c498: x the AO; the sun's shadow in z, which the sun's light shader
// d92266b9 reads, or in y for the other region, which the global pass 3fe9e964 reads, and that one
// comes first). Before the first of the two the shadow is replaced:
// one ray per pixel towards the sun (world direction from the camera block of f55643de, [80..82]),
// the game's shadow where the ray leaves the RT scene without a hit.
#include "render/vk_rtao.h"
#include "render/vk_rtao_internal.h"

#include <cmath>
#include <cstdio>
#include <vk_mem_alloc.h>
#include "bbgpu/shaders/rt_sun_shadow_comp.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

namespace Vulkan {
using namespace RtaoInternal;

namespace {
constexpr u64 SunResolvePs = 0xf55643de; // the cascaded shadow map into the screen mask
constexpr u64 SunMaskPs = 0xa291c498;    // the final mask: AO and the sun's shadow
constexpr u64 SunLightPs = 0xd92266b9;   // the sun's light (reads the mask's z)
constexpr u64 GlobalLightPs = 0x3fe9e964; // the global pass, before it (reads the mask's y)

struct SunConstants {
    std::array<float, 12> view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> jitter;
    std::array<float, 3> sun;
    float spread;
    std::array<float, 4> region_a;
    std::array<float, 4> region_b;
    float erode;
    float tmax;
    u32 frame;
    u32 mode;
};
static_assert(sizeof(SunConstants) == 144);
} // namespace

void RayTracedAO::OnShaderConstants(u64 hash, u32 slot, VAddr address, u64 size) {
    const auto* f = reinterpret_cast<const float*>(address);
    // By size, not slot (the guest-buffer path numbers bindings across the stages): the resolve's
    // camera block is its only 864-byte buffer, the mask pass's coefficients its only 32-byte one.
    (void)slot;
    if (hash == SunResolvePs && size == 864) {
        const float x = f[80], y = f[81], z = f[82];
        const float len = std::sqrt(x * x + y * y + z * z);
        if (len > 0.5f && len < 1.5f) {
            sun_dir = {x / len, y / len, z / len};
            sun_dir_frame = frame;
        }
    } else if (hash == SunMaskPs && size == 32) {
        // Its two draws (two regions): y = f[1] s + f[5], z = f[2] s + f[6]; the channel that
        // carries the shadow is the one with a scale.
        if (sun_k_frame != frame) {
            sun_k_frame = frame;
            sun_regions = 0;
            sun_k = {};
        }
        if (sun_regions < 2) {
            const u32 channel = f[2] != 0.0f ? 2 : f[1] != 0.0f ? 1 : 0;
            if (channel) {
                sun_k[sun_regions++] = {float(channel), f[channel], f[channel + 4], 0.0f};
            }
        }
    }
}

bool RayTracedAO::EnsureSun(u32 w, u32 h) {
    if (sun_vis_view && sun_w == w && sun_h == h) {
        return true;
    }
    const auto device = instance.GetDevice();
    if (!sun_pipeline) {
        const auto binding = [](u32 index, vk::DescriptorType type) {
            return vk::DescriptorSetLayoutBinding{.binding = index,
                                                  .descriptorType = type,
                                                  .descriptorCount = 1,
                                                  .stageFlags = vk::ShaderStageFlagBits::eCompute};
        };
        const std::array bindings = {binding(0, vk::DescriptorType::eSampledImage),
                                     binding(1, vk::DescriptorType::eSampledImage),
                                     binding(2, vk::DescriptorType::eStorageImage),
                                     binding(3, vk::DescriptorType::eStorageImage),
                                     binding(4, vk::DescriptorType::eAccelerationStructureKHR)};
        sun_desc_layout = Check(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = u32(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                          .offset = 0,
                                          .size = sizeof(SunConstants)};
        sun_layout = Check(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &*sun_desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto module = CompileSPV(RT_SUN_SHADOW_COMP, device);
        sun_pipeline = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                    .layout = *sun_layout,
                }));
        device.destroyShaderModule(module);
    }
    scheduler.Finish(); // the old image may be in flight
    sun_vis_view.reset();
    sun_vis = VideoCore::HostImage(device, instance.GetAllocator());
    sun_vis.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR8Unorm,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    sun_vis_view = Check(device.createImageViewUnique({
        .image = vk::Image(sun_vis),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR8Unorm,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    sun_w = w;
    sun_h = h;
    std::printf("RTAO: sun shadow %ux%u (the sun's shadow by ray tracing, sun_shadow=1)\n", w, h);
    return true;
}

void RayTracedAO::OnSunDraw(VideoCore::ImageId color, u64 ps_hash) {
    if (ps_hash == SunMaskPs) {
        sun_mask = color;
        sun_mask_frame = frame;
        return;
    }
    if ((ps_hash != SunLightPs && ps_hash != GlobalLightPs) || sun_done_frame == frame) {
        return;
    }
    sun_done_frame = frame;
    // The report: per 600 frames, how often the sun draw found everything it needs.
    if (++sun_frames == 600) {
        std::printf("RTAO: sun shadow: %u of 600 sun draws ray traced (%u without the RT scene, the mask "
                    "or the sun's direction); sun (%.3f %.3f %.3f), mask regions: %u (channel %.0f, %.0f)\n",
                    sun_traced, sun_frames - sun_traced, sun_dir[0], sun_dir[1], sun_dir[2], sun_regions,
                    sun_k[0][0], sun_k[1][0]);
        sun_frames = sun_traced = 0;
    }
    const vk::AccelerationStructureKHR tlas = FrameTlas();
    if (!settings.sun_shadow || !tlas || sun_mask_frame != frame || sun_dir_frame != frame ||
        sun_k_frame != frame || sun_regions == 0 || !depth_id || !normal_id) {
        return;
    }
    auto& mask = texture_cache.GetImage(sun_mask);
    auto& depth = texture_cache.GetImage(depth_id);
    auto& normal = texture_cache.GetImage(normal_id);
    const u32 w = depth.info.size.width, h = depth.info.size.height;
    if (mask.info.pixel_format != vk::Format::eR8G8B8A8Unorm || mask.info.size.width != w ||
        mask.info.size.height != h || normal.info.size.width != w || !EnsureSun(w, h)) {
        return;
    }
    ++sun_traced;
    scheduler.EndRendering();
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&normal, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&mask, vk::ImageLayout::eGeneral, Compute, rw);
    runtime.FlushBarriers();
    const vk::ImageView depth_view = CachedView(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const vk::ImageView normal_view = CachedView(normal, normal.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView mask_view = CachedView(mask, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);

    const auto view = camera_motion.View();
    std::array<float, 3> sun_view{};
    for (int r = 0; r < 3; ++r) {
        sun_view[r] = view[r * 4] * sun_dir[0] + view[r * 4 + 1] * sun_dir[1] + view[r * 4 + 2] * sun_dir[2];
    }
    const SunConstants constants{
        .view = view,
        .proj = camera_motion.Projection(),
        .full_size = {float(w), float(h)},
        .jitter = camera_motion.Jitter(),
        .sun = sun_view,
        .spread = settings.sun_spread,
        .region_a = sun_k[0],
        .region_b = sun_k[1],
        .erode = float(settings.sun_erode),
        .tmax = settings.sun_far,
        .frame = frame,
        .mode = settings.sun_debug << 4,
    };
    const vk::ImageMemoryBarrier2 to_write{
        .srcStageMask = Compute,
        .srcAccessMask = rw,
        .dstStageMask = Compute,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined, // its last content is not needed
        .newLayout = vk::ImageLayout::eGeneral,
        .image = vk::Image(sun_vis),
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    scheduler.RecordCrumb(
        {.name = "rt sun shadow"},
        [constants, to_write, tlas, depth_view, normal_view, mask_view, vis = *sun_vis_view, w, h,
         pipeline = *sun_pipeline, layout = *sun_layout](vk::CommandBuffer cmdbuf) {
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_write});
            const auto info = [](vk::ImageView v) {
                return vk::DescriptorImageInfo{.imageView = v, .imageLayout = vk::ImageLayout::eGeneral};
            };
            const auto d = info(depth_view), n = info(normal_view), m = info(mask_view), v = info(vis);
            const vk::WriteDescriptorSetAccelerationStructureKHR as{.accelerationStructureCount = 1,
                                                                    .pAccelerationStructures = &tlas};
            const std::array writes = {
                vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &n},
                vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &m},
                vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &v},
                vk::WriteDescriptorSet{.pNext = &as, .dstBinding = 4, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eAccelerationStructureKHR}};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            // Mode 0: the visibility (reads the game's mask around each pixel), then mode 1: into
            // the mask's z.
            auto pc = constants;
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmdbuf.dispatch((w + 7) / 8, (h + 7) / 8, 1);
            const vk::MemoryBarrier2 b{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                       .srcAccessMask = vk::AccessFlagBits2::eShaderWrite |
                                                        vk::AccessFlagBits2::eShaderRead,
                                       .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                       .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                                        vk::AccessFlagBits2::eShaderWrite};
            cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
            pc.mode = 1 | (pc.mode & ~15u);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
            cmdbuf.dispatch((w + 7) / 8, (h + 7) / 8, 1);
        });
}

} // namespace Vulkan
