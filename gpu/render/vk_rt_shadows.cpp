// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_rt_shadows.h"

#include <cmath>
#include <cstdio>
#include <vk_mem_alloc.h>
#include "bbgpu/shaders/rt_spot_map_comp.h"
#include "bbgpu/shaders/rt_sun_shadow_comp.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_gpu_profiler.h"
#include "render/vk_rt_pipeline.h"
#include "render/vk_rt_lights.h"
#include "render/vk_rtao.h"
#include "textures/texture_cache.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "vulkan/shader_module.h"

namespace Vulkan {

namespace {
constexpr u64 SunLightPs = 0xd92266b9;   // the sun's light (reads the mask's z)
constexpr u64 GlobalLightPs = 0x3fe9e964; // the global pass, before it (reads the mask's y)
constexpr auto Compute = vk::PipelineStageFlagBits2::eComputeShader;

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

struct SpotMapConstants {
    std::array<float, 4> row_x;
    std::array<float, 4> row_y;
    std::array<float, 4> row_w;
    std::array<float, 3> light;
    float inv_range;
    float near;
    u32 size;
    u32 mode;
    s32 blur;
    float var_cap;
};
static_assert(sizeof(SpotMapConstants) == 84);

// a (4x4, rows) x the 3x4 rows of b extended by (0 0 0 1).
std::array<std::array<float, 4>, 4> Mul(const std::array<std::array<float, 4>, 4>& a,
                                        const std::array<float, 12>& b) {
    std::array<std::array<float, 4>, 4> out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float v = c == 3 ? a[r][3] : 0.0f;
            for (int k = 0; k < 3; ++k) {
                v += a[r][k] * b[k * 4 + c];
            }
            out[r][c] = v;
        }
    }
    return out;
}

} // namespace

RtShadows::RtShadows(const Instance& instance, Scheduler& scheduler, VideoCore::TextureCache& texture_cache,
                     Runtime& runtime, CameraMotion& camera_motion, const RayTracedAO& rtao, const RtLights& lights,
                     const RtLightingSettings::Settings& settings)
    : instance{instance}, scheduler{scheduler}, texture_cache{texture_cache}, runtime{runtime},
      camera_motion{camera_motion}, rtao{rtao}, lights{lights}, settings{settings}, views{instance, scheduler} {}

RtShadows::~RtShadows() = default;

void RtShadows::OnFrameStart(u32 next_frame) {
    frame = next_frame;
}

void RtShadows::OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal) {
    depth_id = depth;
    normal_id = normal;
}

void RtShadows::OnShaderConstants(u64 hash, VAddr address, u64 size) {
    const auto* f = reinterpret_cast<const float*>(address);
    // The resolve's camera block is its only 864-byte buffer, the mask pass's coefficients its
    // only 32-byte one.
    if (hash == ResolvePs && size == 864) {
        const float x = f[80], y = f[81], z = f[82];
        const float len = std::sqrt(x * x + y * y + z * z);
        if (len > 0.5f && len < 1.5f) {
            sun_dir = {x / len, y / len, z / len};
            sun_dir_frame = frame;
        }
    } else if (hash == MaskPs && size == 32) {
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

void RtShadows::OnDraw(VideoCore::ImageId color, u64 ps_hash) {
    if (ps_hash == MaskPs) {
        sun_mask = color;
        sun_mask_frame = frame;
        return;
    }
    if ((ps_hash != SunLightPs && ps_hash != GlobalLightPs) || done_frame == frame) {
        return;
    }
    done_frame = frame; // before the light passes: once a frame
    if (++report_frames == 600) {
        std::printf("RT shadows: sun ray traced in %u of 600 frames (sun %d: (%.3f %.3f %.3f), mask regions %u: "
                    "channel %.0f, %.0f); spot map in %u (spot %d)\n",
                    sun_traced, settings.sun_shadow ? 1 : 0, sun_dir[0], sun_dir[1], sun_dir[2], sun_regions,
                    sun_k[0][0], sun_k[1][0], spot_maps, settings.spot_shadow ? 1 : 0);
        report_frames = sun_traced = spot_maps = 0;
    }
    if (settings.spot_shadow) {
        RecordSpotMap();
    }
    if (settings.sun_shadow) {
        RecordSun();
    }
}

bool RtShadows::EnsureSun(u32 w, u32 h) {
    if (sun_vis_view && sun_w == w && sun_h == h) {
        return true;
    }
    const auto device = instance.GetDevice();
    if (!sun_pipeline) {
        const std::array bindings = {RtPipeline::Binding(0, vk::DescriptorType::eSampledImage),
                                     RtPipeline::Binding(1, vk::DescriptorType::eSampledImage),
                                     RtPipeline::Binding(2, vk::DescriptorType::eStorageImage),
                                     RtPipeline::Binding(3, vk::DescriptorType::eStorageImage),
                                     RtPipeline::Binding(4, vk::DescriptorType::eAccelerationStructureKHR)};
        sun_pipeline = RtPipeline::Make(device, RT_SUN_SHADOW_COMP, bindings, sizeof(SunConstants), sun_desc_layout,
                                    sun_layout);
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
    std::printf("RT shadows: sun %ux%u (the sun's shadow by ray tracing, sun_shadow=1)\n", w, h);
    return true;
}

void RtShadows::RecordSun() {
    const vk::AccelerationStructureKHR tlas = rtao.FrameTlas();
    if (!tlas || sun_mask_frame != frame || sun_dir_frame != frame || sun_k_frame != frame || sun_regions == 0 ||
        !depth_id || !normal_id) {
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
    const vk::ImageView depth_view = views.Get(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const vk::ImageView normal_view = views.Get(normal, normal.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView mask_view = views.Get(mask, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);

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
    const ProfileScope profile{0x55606ull, "rt sun shadow"};
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
            // the mask's channel.
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

void RtShadows::OnSpotMapImage(VAddr address, vk::Format format) {
    if (format == vk::Format::eR16G16Sfloat) {
        spot_target = address;
    }
}

vk::ImageView RtShadows::SpotMapView(VAddr address) const noexcept {
    return address && address == spot_target && spot_map_frame == frame && spot_map_view ? *spot_map_view
                                                                                         : vk::ImageView{};
}

void RtShadows::RecordSpotMap() {
    const vk::AccelerationStructureKHR tlas = rtao.FrameTlas();
    // Last frame's light (its record arrives in this frame's light passes, after this).
    const auto& spot = lights.FrameSpot();
    if (!tlas || !spot_target || !spot.valid || !(spot.range > 0.0f)) {
        return;
    }
    const u32 size = settings.spot_res;
    const auto device = instance.GetDevice();
    if (!spot_pipeline) {
        const std::array bindings = {RtPipeline::Binding(0, vk::DescriptorType::eStorageImage),
                                     RtPipeline::Binding(1, vk::DescriptorType::eAccelerationStructureKHR),
                                     RtPipeline::Binding(2, vk::DescriptorType::eStorageImage)};
        spot_pipeline = RtPipeline::Make(device, RT_SPOT_MAP_COMP, bindings, sizeof(SpotMapConstants), spot_desc_layout,
                                     spot_layout);
    }
    if (spot_map_size != size) {
        scheduler.Finish(); // the old map may be in flight
        spot_map_view.reset();
        spot_blur_view.reset();
        const auto make = [&](VideoCore::HostImage& image, vk::UniqueImageView& view) {
            image = VideoCore::HostImage(device, instance.GetAllocator());
            image.Create(vk::ImageCreateInfo{
                .imageType = vk::ImageType::e2D,
                .format = vk::Format::eR32G32Sfloat,
                .extent = {size, size, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .tiling = vk::ImageTiling::eOptimal,
                .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                .initialLayout = vk::ImageLayout::eUndefined,
            });
            view = Check(device.createImageViewUnique({
                .image = vk::Image(image),
                .viewType = vk::ImageViewType::e2D,
                .format = vk::Format::eR32G32Sfloat,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            }));
        };
        make(spot_map, spot_map_view);   // the map the light passes read
        make(spot_blur, spot_blur_view); // the blur's middle
        spot_map_size = size;
        std::printf("RT shadows: spot map %ux%u (the shadowed spot's, ray traced; spot_shadow=1)\n", size, size);
    }
    // World -> light clip x this frame's view -> world: the rows the rays are built from.
    const auto m = Mul(spot.world_to_clip, camera_motion.InvView());
    const auto view = camera_motion.View();
    std::array<float, 3> light{};
    for (int i = 0; i < 3; ++i) {
        light[i] = view[i * 4] * spot.light[0] + view[i * 4 + 1] * spot.light[1] + view[i * 4 + 2] * spot.light[2] +
                   view[i * 4 + 3];
    }
    const SpotMapConstants constants{
        .row_x = m[0],
        .row_y = m[1],
        .row_w = m[3],
        .light = light,
        .inv_range = 1.0f / spot.range,
        .near = settings.spot_near,
        .size = size,
        .blur = s32(settings.spot_blur),
        .var_cap = settings.spot_soft / spot.range,
    };
    scheduler.EndRendering();
    const vk::ImageMemoryBarrier2 to_write{
        .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = Compute,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined, // rewritten whole
        .newLayout = vk::ImageLayout::eGeneral,
        .image = vk::Image(spot_map),
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    auto to_write_blur = to_write;
    to_write_blur.image = vk::Image(spot_blur);
    const vk::ImageMemoryBarrier2 to_read{
        .srcStageMask = Compute,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = vk::Image(spot_map),
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    const ProfileScope profile{0x55607ull, "rt spot map"};
    scheduler.RecordCrumb(
        {.name = "rt spot map"},
        [constants, to_write, to_write_blur, to_read, tlas, map = *spot_map_view, mid = *spot_blur_view, size,
         pipeline = *spot_pipeline, layout = *spot_layout](vk::CommandBuffer cmdbuf) {
            const std::array to_writes = {to_write, to_write_blur};
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = to_writes.data()});
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            const vk::WriteDescriptorSetAccelerationStructureKHR as{.accelerationStructureCount = 1,
                                                                    .pAccelerationStructures = &tlas};
            // out <- in: one dispatch of `mode` (the trace reads no image; it is bound all the same).
            const auto run = [&](vk::ImageView out, vk::ImageView in, u32 mode) {
                const vk::DescriptorImageInfo o{.imageView = out, .imageLayout = vk::ImageLayout::eGeneral};
                const vk::DescriptorImageInfo i{.imageView = in, .imageLayout = vk::ImageLayout::eGeneral};
                const std::array writes = {
                    vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &o},
                    vk::WriteDescriptorSet{.pNext = &as, .dstBinding = 1, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
                    vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &i}};
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
                auto pc = constants;
                pc.mode = mode;
                cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
                cmdbuf.dispatch((size + 7) / 8, (size + 7) / 8, 1);
                const vk::MemoryBarrier2 b{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                           .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                           .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                           .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                                            vk::AccessFlagBits2::eShaderWrite};
                cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
            };
            if (constants.blur > 0) { // trace into the map, blur x into the middle, y back into the map
                run(map, mid, 0);
                run(mid, map, 1);
                run(map, mid, 2);
            } else {
                run(map, mid, 0);
            }
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
        });
    spot_map_frame = frame;
    ++spot_maps;
}

} // namespace Vulkan
