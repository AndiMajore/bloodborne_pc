// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the ray-traced AO's surface cache (experimental, surface=1 in rtao.txt): radiance per
// static pool triangle, refreshed from the lit screen where a triangle is seen (a budgeted slice
// a frame), read by other ray-traced passes (the RT gather of SSGI, the RT reflections) where
// their hit is hidden from the camera. A screen-fed cache in the spirit of Lumen's surface cache,
// without cards: what the camera never saw stays dark (weight 0), view-dependent highlights are
// baked in, and lighting changes catch up as the slices come round again.
#include "render/vk_rtao.h"
#include "render/vk_rtao_internal.h"

#include <algorithm>
#include <cstdio>
#include <vk_mem_alloc.h>
#include "bbgpu/shaders/rtao_surface_comp.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_gpu_profiler.h"
#include "textures/texture_cache.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

namespace Vulkan {
using namespace RtaoInternal;

namespace {
struct SurfaceConstants {
    std::array<float, 12> view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> jitter;
    u64 pool_address;
    u64 parts_address;
    u64 cache_address;
    u32 part_count;
    u32 total;
    u32 cursor;
    u32 count;
};
static_assert(sizeof(SurfaceConstants) == 120);
} // namespace

bool RayTracedAO::CreateSurfaceCache() {
    if (surface_cache.buffer) {
        return true;
    }
    using BU = vk::BufferUsageFlagBits;
    if (!CreateBuffer(surface_cache, vk::DeviceSize(pool_triangles) * 8, BU::eStorageBuffer | BU::eTransferDst,
                      false) ||
        !CreateBuffer(surface_parts, vk::DeviceSize(MaxSurfaceParts) * 8, BU::eStorageBuffer | BU::eTransferDst,
                      false)) {
        DestroyBuffer(surface_cache);
        settings.surface = false;
        std::puts("RTAO: surface cache: buffers failed, off");
        return false;
    }
    const auto device = instance.GetDevice();
    const auto binding = [](u32 index) {
        return vk::DescriptorSetLayoutBinding{.binding = index,
                                              .descriptorType = vk::DescriptorType::eSampledImage,
                                              .descriptorCount = 1,
                                              .stageFlags = vk::ShaderStageFlagBits::eCompute};
    };
    const std::array bindings = {binding(0), binding(1)};
    surface_desc_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(SurfaceConstants)};
    surface_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*surface_desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(RTAO_SURFACE_COMP, device);
    surface_pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                .layout = *surface_layout,
            }));
    device.destroyShaderModule(module);
    surface_cleared = false;
    // The table is filled with the next static build; make one now.
    static_dirty = true;
    std::printf("RTAO: surface cache on (%llu MiB, %u static triangles a frame)\n",
                (unsigned long long)((surface_cache.size + surface_parts.size) >> 20), settings.surface_budget);
    return true;
}

void RayTracedAO::SurfaceTable(PassInputs& in) {
    in.surface_table.clear();
    if (!settings.surface || !surface_cache.buffer) {
        return;
    }
    if (in.static_parts.size() > MaxSurfaceParts) {
        surface_part_count = surface_total = 0; // too many to map: off until a smaller build
        return;
    }
    u32 first = 0;
    for (const auto& [offset, count] : in.static_parts) {
        in.surface_table.push_back({offset, first});
        first += count;
    }
    in.surface_parts_handle = surface_parts.buffer;
    surface_part_count = u32(in.static_parts.size());
    surface_total = first;
    surface_cursor = 0;
}

void RayTracedAO::OnOpaqueScene(VideoCore::ImageId color_id) {
    if (!enabled || !Active() || !color_id) {
        return;
    }
    if (settings.surface && !surface_cache.buffer && resources_ready) {
        CreateSurfaceCache();
    }
    if (!settings.surface || !surface_cache.buffer || tlas_frame != frame || !depth_id || surface_frame == frame) {
        return;
    }
    surface_frame = frame;
    if (!surface_cleared) { // fresh buffers hold garbage: never seen everywhere
        surface_cleared = true;
        scheduler.EndRendering();
        scheduler.Record([cache = surface_cache.buffer](vk::CommandBuffer cmdbuf) {
            cmdbuf.fillBuffer(cache, 0, VK_WHOLE_SIZE, 0);
            const vk::MemoryBarrier2 b{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                       .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                       .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                       .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                                        vk::AccessFlagBits2::eShaderWrite};
            cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
        });
    }
    if (!surface_part_count || !surface_total) {
        return;
    }
    auto& color = texture_cache.GetImage(color_id);
    auto& depth = texture_cache.GetImage(depth_id);
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat ||
        color.info.size.width != depth.info.size.width) {
        return;
    }
    scheduler.EndRendering();
    const auto compute = vk::PipelineStageFlagBits2::eComputeShader;
    runtime.Transit(&color, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const vk::ImageView color_view =
        CachedView(color, color.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView depth_view =
        CachedView(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const u32 count = std::min(settings.surface_budget, surface_total);
    const SurfaceConstants constants{
        .view = camera_motion.View(),
        .proj = camera_motion.Projection(),
        .full_size = {float(depth.info.size.width), float(depth.info.size.height)},
        .jitter = camera_motion.Jitter(),
        .pool_address = static_pool.address,
        .parts_address = surface_parts.address,
        .cache_address = surface_cache.address,
        .part_count = surface_part_count,
        .total = surface_total,
        .cursor = surface_cursor,
        .count = count,
    };
    surface_cursor = (surface_cursor + count) % surface_total;
    const ProfileScope profile{0x55605ull, "rtao surface"};
    scheduler.RecordCrumb({.name = "rtao surface"},
                          [constants, color_view, depth_view, count, pipeline = *surface_pipeline,
                           layout = *surface_layout](vk::CommandBuffer cmdbuf) {
                              const vk::DescriptorImageInfo c{.imageView = color_view,
                                                              .imageLayout = vk::ImageLayout::eGeneral};
                              const vk::DescriptorImageInfo d{.imageView = depth_view,
                                                              .imageLayout = vk::ImageLayout::eGeneral};
                              const std::array writes = {
                                  vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage,
                                                         .pImageInfo = &c},
                                  vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage,
                                                         .pImageInfo = &d}};
                              cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
                              cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
                              cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                                   sizeof(constants), &constants);
                              cmdbuf.dispatch((count + 63) / 64, 1, 1);
                              const vk::MemoryBarrier2 b{
                                  .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                  .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                  .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                                                  vk::PipelineStageFlagBits2::eTransfer,
                                  .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                                   vk::AccessFlagBits2::eTransferWrite};
                              cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
                          });
}

} // namespace Vulkan
