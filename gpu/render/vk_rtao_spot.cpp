// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the shadowed spot light's shadow map by ray tracing (experimental, spot_shadow= in
// rtao.txt; vk_rtao.h, rt_spot_map.comp). The game's shadowed spot (ps 82951fa5, ticket 59) reads
// a 512x512 variance shadow map, blocky up close. Its light record (the light list, the frame
// before) gives the light and its view -> light clip matrix; before the next frame's light passes
// a larger map is ray traced from the light, and the light passes sample it in the game's place
// (raster_textures.cpp).
#include "render/vk_rtao.h"
#include "render/vk_rtao_internal.h"

#include <cmath>
#include <cstdio>
#include <vk_mem_alloc.h>
#include "bbgpu/shaders/rt_spot_map_comp.h"
#include "render/scheduler.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

namespace Vulkan {
using namespace RtaoInternal;

namespace {
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
};
static_assert(sizeof(SpotMapConstants) == 80);

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

void RayTracedAO::OnSpotRecord(const float* r) {
    if (spot_frame == frame) {
        return; // the first shadowed spot of the frame
    }
    // World -> light clip: the record's view -> light clip x this frame's world -> view.
    std::array<std::array<float, 4>, 4> m{};
    for (int row = 0; row < 4; ++row) {
        for (int c = 0; c < 4; ++c) {
            m[row][c] = r[16 + row * 4 + c];
        }
    }
    spot_world = Mul(m, camera_motion.View());
    const auto inv_view = camera_motion.InvView();
    for (int i = 0; i < 3; ++i) {
        spot_light[i] = inv_view[i * 4] * r[0] + inv_view[i * 4 + 1] * r[1] + inv_view[i * 4 + 2] * r[2] +
                        inv_view[i * 4 + 3];
    }
    spot_range = r[3];
    spot_frame = frame;
}

void RayTracedAO::OnSpotMapImage(VAddr address, vk::Format format) {
    if (format == vk::Format::eR16G16Sfloat) {
        spot_target = address;
    }
}

vk::ImageView RayTracedAO::SpotMapView(VAddr address) const noexcept {
    return address && address == spot_target && spot_map_frame == frame && spot_map_view ? *spot_map_view
                                                                                         : vk::ImageView{};
}

void RayTracedAO::RecordSpotMap() {
    const vk::AccelerationStructureKHR tlas = FrameTlas();
    // Last frame's light (its record arrives in this frame's light passes, after this).
    if (!settings.spot_shadow || !tlas || !spot_target || spot_frame + 1 != frame || !(spot_range > 0.0f)) {
        return;
    }
    const u32 size = settings.spot_res;
    const auto device = instance.GetDevice();
    if (!spot_pipeline) {
        const std::array bindings = {
            vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eStorageImage,
                                           .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            vk::DescriptorSetLayoutBinding{.binding = 1,
                                           .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
                                           .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
            vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eStorageImage,
                                           .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};
        spot_desc_layout = Check(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = u32(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                          .offset = 0,
                                          .size = sizeof(SpotMapConstants)};
        spot_layout = Check(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &*spot_desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto module = CompileSPV(RT_SPOT_MAP_COMP, device);
        spot_pipeline = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                    .layout = *spot_layout,
                }));
        device.destroyShaderModule(module);
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
        std::printf("RTAO: spot shadow map %ux%u (the shadowed spot's, ray traced; spot_shadow=1)\n", size, size);
    }
    // World -> light clip x this frame's view -> world: the rows the rays are built from.
    const auto m = Mul(spot_world, camera_motion.InvView());
    const auto view = camera_motion.View();
    std::array<float, 3> light{};
    for (int i = 0; i < 3; ++i) {
        light[i] = view[i * 4] * spot_light[0] + view[i * 4 + 1] * spot_light[1] + view[i * 4 + 2] * spot_light[2] +
                   view[i * 4 + 3];
    }
    const SpotMapConstants constants{
        .row_x = m[0],
        .row_y = m[1],
        .row_w = m[3],
        .light = light,
        .inv_range = 1.0f / spot_range,
        .near = settings.spot_near,
        .size = size,
        .blur = s32(settings.spot_blur),
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
