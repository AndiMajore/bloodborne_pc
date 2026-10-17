// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_rt_probes.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include "common/path_util.h"

#include "bbgpu/shaders/rt_probes_apply_comp.h"
#include "bbgpu/shaders/rt_probes_update_comp.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_rtao.h"
#include "textures/texture_cache.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

namespace Vulkan {

namespace {
struct UpdateConstants {
    std::array<float, 12> view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> jitter;
    std::array<s32, 4> grid_origin;
    std::array<s32, 4> dims;
    float spacing;
    float hysteresis;
    float max_dist;
    float sky;
    u32 frame;
    u32 surface_parts_count;
    u64 surface_cache;
    u64 surface_parts;
};
static_assert(sizeof(UpdateConstants) == 152);
struct ApplyConstants {
    std::array<float, 12> inv_view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> jitter;
    std::array<s32, 4> grid_origin;
    std::array<s32, 4> dims;
    float spacing;
    float strength;
    float normal_bias;
    u32 debug;
};
static_assert(sizeof(ApplyConstants) == 128);
constexpr u32 ProbeBytes = 64; // vec4 r, g, b + ivec4 cell
} // namespace

RtProbes::RtProbes(const Instance& instance_, Scheduler& scheduler_,
                   VideoCore::TextureCache& texture_cache_, Runtime& runtime_,
                   CameraMotion& camera_motion_, RayTracedAO& rtao_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, runtime{runtime_},
      camera_motion{camera_motion_}, rtao{rtao_} {
    const char* env = std::getenv("BB_RT_PROBES");
    enabled = env && env[0] == '1' && rtao.Enabled() && instance.IsRayQuerySupported() &&
              instance.GetPhysicalDevice().getProperties().limits.maxPushConstantsSize >=
                  sizeof(UpdateConstants);
    if (!enabled) {
        if (env && env[0] == '1') {
            std::printf("RT probes: off (needs the RTAO with ray queries)\n");
        }
        return;
    }
    if (const char* file = std::getenv("BB_RT_PROBES_FILE")) {
        settings_path = file;
    } else {
        settings_path =
            (Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "rtprobes.txt").string();
    }
    const u32 count = u32(Dims[0] * Dims[1] * Dims[2]);
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = VkDeviceSize(count) * ProbeBytes,
        .usage = VkBufferUsageFlags(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo alloc_ci{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkBuffer raw{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &raw, &probe_allocation, nullptr) !=
        VK_SUCCESS) {
        std::printf("RT probes: buffer failed, off\n");
        enabled = false;
        return;
    }
    probe_buffer = raw;
    std::printf("RT probes: on (experimental), %dx%dx%d probes, settings in %s\n", Dims[0], Dims[1],
                Dims[2], settings_path.c_str());

    const auto device = instance.GetDevice();
    const auto make = [&](std::span<const vk::DescriptorSetLayoutBinding> bindings, u32 push_size,
                          std::span<const u32> code, vk::UniqueDescriptorSetLayout& desc_layout,
                          vk::UniquePipelineLayout& layout, vk::UniquePipeline& pipeline) {
        desc_layout = Check(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = u32(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                          .offset = 0,
                                          .size = push_size};
        layout = Check(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &*desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto module = CompileSPV(code, device);
        pipeline = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                    .layout = *layout,
                }));
        device.destroyShaderModule(module);
    };
    const auto binding = [](u32 index, vk::DescriptorType type) {
        return vk::DescriptorSetLayoutBinding{.binding = index,
                                              .descriptorType = type,
                                              .descriptorCount = 1,
                                              .stageFlags = vk::ShaderStageFlagBits::eCompute};
    };
    using DT = vk::DescriptorType;
    const std::array update_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eSampledImage),
                                        binding(2, DT::eAccelerationStructureKHR),
                                        binding(3, DT::eStorageBuffer)};
    const std::array apply_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eSampledImage),
                                       binding(2, DT::eSampledImage), binding(3, DT::eStorageImage),
                                       binding(4, DT::eStorageBuffer)};
    make(update_bindings, sizeof(UpdateConstants), RT_PROBES_UPDATE_COMP, update_desc_layout,
         update_layout, update_pipeline);
    make(apply_bindings, sizeof(ApplyConstants), RT_PROBES_APPLY_COMP, apply_desc_layout,
         apply_layout, apply_pipeline);
}

RtProbes::~RtProbes() {
    const auto device = instance.GetDevice();
    for (auto& entry : view_cache) {
        if (entry.view) {
            device.destroyImageView(entry.view);
        }
    }
    if (probe_buffer) {
        vmaDestroyBuffer(instance.GetAllocator(), probe_buffer, probe_allocation);
    }
}

void RtProbes::OnFrameStart() {
    if (!enabled) {
        return;
    }
    done_this_frame = false;
    ++frame;
    PollSettings();
    if (frame % 600 == 0 && (passes || skipped)) {
        std::printf("RT probes: %u passes in 600 frames, %u skipped (no RTAO scene); spacing %.1f m, "
                    "%u rays, strength %.2f\n",
                    passes, skipped, spacing, rays, strength);
        passes = skipped = 0;
    }
}

void RtProbes::PollSettings() {
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
    const float old_spacing = spacing;
    std::ifstream file(settings_path);
    std::string line;
    while (std::getline(file, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos || line.empty() || line[0] == '#') {
            continue;
        }
        const std::string key = line.substr(0, eq);
        const float value = std::strtof(line.c_str() + eq + 1, nullptr);
        if (key == "enabled") active = value != 0.0f;
        else if (key == "strength") strength = std::clamp(value, 0.0f, 8.0f);
        else if (key == "spacing") spacing = std::clamp(value, 0.5f, 16.0f);
        else if (key == "rays") rays = std::clamp(u32(std::max(value, 1.0f)), 1u, 64u);
        else if (key == "hysteresis") hysteresis = std::clamp(value, 0.0f, 0.99f);
        else if (key == "sky") sky = std::max(value, 0.0f);
        else if (key == "bias") bias = std::clamp(value, 0.0f, 2.0f);
        else if (key == "debug") debug = std::clamp(u32(std::max(value, 0.0f)), 0u, 2u);
    }
    reset |= spacing != old_spacing;
    std::printf("RT probes: settings: enabled %d, strength %.2f, spacing %.1f m, %u rays, hysteresis "
                "%.2f, sky %.2f, bias %.2f, debug %u\n",
                active ? 1 : 0, strength, spacing, rays, hysteresis, sky, bias, debug);
}

void RtProbes::OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal,
                             VideoCore::ImageId albedo) {
    depth_id = depth;
    normal_id = normal;
    albedo_id = albedo;
}

vk::ImageView RtProbes::CachedView(const VideoCore::Image& image, vk::Format format,
                                   vk::ImageAspectFlags aspect) {
    const vk::Image handle = image.GetImage();
    ViewEntry* oldest = &view_cache[0];
    for (auto& entry : view_cache) {
        if (entry.view && entry.image == handle && entry.uid == image.image_uid &&
            entry.format == format && entry.aspect == aspect) {
            entry.last_use = ++view_uses;
            return entry.view;
        }
        if (entry.last_use < oldest->last_use) {
            oldest = &entry;
        }
    }
    const auto device = instance.GetDevice();
    if (oldest->view) {
        scheduler.DeferOperation([device, view = oldest->view] { device.destroyImageView(view); });
    }
    *oldest = {
        .image = handle,
        .uid = image.image_uid,
        .format = format,
        .aspect = aspect,
        .view = Check(device.createImageView({
            .image = handle,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {aspect, 0, 1, 0, 1},
        })),
        .last_use = ++view_uses,
    };
    return oldest->view;
}

void RtProbes::OnOpaqueScene(VideoCore::ImageId color_id) {
    if (!enabled || !active || done_this_frame || !color_id || !depth_id || !normal_id || !albedo_id) {
        return;
    }
    done_this_frame = true;
    const vk::AccelerationStructureKHR tlas = rtao.FrameTlas();
    if (!tlas) {
        ++skipped;
        return;
    }
    auto& color = texture_cache.GetImage(color_id);
    auto& depth = texture_cache.GetImage(depth_id);
    auto& normal = texture_cache.GetImage(normal_id);
    auto& albedo = texture_cache.GetImage(albedo_id);
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat ||
        color.info.size.width != depth.info.size.width || normal.info.size.width != depth.info.size.width ||
        albedo.info.size.width != depth.info.size.width) {
        ++skipped;
        return;
    }
    ++passes;
    scheduler.EndRendering();
    const auto compute = vk::PipelineStageFlagBits2::eComputeShader;
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&normal, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&albedo, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&color, vk::ImageLayout::eGeneral, compute, rw);
    runtime.FlushBarriers();
    const vk::ImageView depth_view = CachedView(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const vk::ImageView normal_view = CachedView(normal, normal.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView albedo_view = CachedView(albedo, albedo.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView color_view = CachedView(color, color.info.pixel_format, vk::ImageAspectFlagBits::eColor);

    // The grid follows the camera in whole cells.
    const auto inv_view = camera_motion.InvView();
    const std::array<float, 3> cam{inv_view[3], inv_view[7], inv_view[11]};
    std::array<s32, 4> origin{};
    for (int i = 0; i < 3; ++i) {
        origin[i] = s32(std::floor(cam[i] / spacing)) - Dims[i] / 2;
    }
    const auto surface = rtao.FrameSurface();
    const u32 w = depth.info.size.width, h = depth.info.size.height;
    const UpdateConstants update{
        .view = camera_motion.View(),
        .proj = camera_motion.Projection(),
        .full_size = {float(w), float(h)},
        .jitter = camera_motion.Jitter(),
        .grid_origin = origin,
        .dims = {Dims[0], Dims[1], Dims[2], s32(rays)},
        .spacing = spacing,
        .hysteresis = hysteresis,
        .max_dist = spacing * 10.0f,
        .sky = sky,
        .frame = frame,
        .surface_parts_count = surface.part_count,
        .surface_cache = surface.cache,
        .surface_parts = surface.parts,
    };
    const ApplyConstants apply{
        .inv_view = inv_view,
        .proj = update.proj,
        .full_size = update.full_size,
        .jitter = update.jitter,
        .grid_origin = origin,
        .dims = update.dims,
        .spacing = spacing,
        .strength = strength,
        .normal_bias = bias,
        .debug = debug,
    };
    const bool clear = std::exchange(reset, false);
    const u32 probes = u32(Dims[0] * Dims[1] * Dims[2]);
    scheduler.RecordCrumb(
        {.name = "rt probes"},
        [update, apply, clear, probes, tlas, depth_view, normal_view, albedo_view, color_view, w, h,
         buffer = probe_buffer, update_pipeline = *update_pipeline, update_layout = *update_layout,
         apply_pipeline = *apply_pipeline, apply_layout = *apply_layout](vk::CommandBuffer cmdbuf) {
            const auto barrier = [&](vk::PipelineStageFlags2 src, vk::AccessFlags2 src_access,
                                     vk::AccessFlags2 dst_access) {
                const vk::MemoryBarrier2 b{.srcStageMask = src,
                                           .srcAccessMask = src_access,
                                           .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                           .dstAccessMask = dst_access};
                cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
            };
            if (clear) { // cell -1 everywhere: every probe starts over
                cmdbuf.fillBuffer(buffer, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
                barrier(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
            }
            const auto image = [](vk::ImageView view) {
                return vk::DescriptorImageInfo{.imageView = view, .imageLayout = vk::ImageLayout::eGeneral};
            };
            const vk::DescriptorBufferInfo probes_info{buffer, 0, VK_WHOLE_SIZE};
            {
                const auto c = image(color_view), d = image(depth_view);
                const vk::WriteDescriptorSetAccelerationStructureKHR as{.accelerationStructureCount = 1,
                                                                        .pAccelerationStructures = &tlas};
                const std::array writes = {
                    vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &c},
                    vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                    vk::WriteDescriptorSet{.pNext = &as, .dstBinding = 2, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
                    vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &probes_info}};
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, update_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, update_layout, 0, writes);
                cmdbuf.pushConstants(update_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(update), &update);
                cmdbuf.dispatch(probes, 1, 1);
            }
            // The probes are written, the lit image is read: both before the apply writes it.
            barrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite |
                        vk::AccessFlagBits2::eShaderRead,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
            {
                const auto d = image(depth_view), n = image(normal_view), a = image(albedo_view),
                           c = image(color_view);
                const std::array writes = {
                    vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                    vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &n},
                    vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &a},
                    vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &c},
                    vk::WriteDescriptorSet{.dstBinding = 4, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &probes_info}};
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, apply_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, apply_layout, 0, writes);
                cmdbuf.pushConstants(apply_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(apply), &apply);
                cmdbuf.dispatch((w + 7) / 8, (h + 7) / 8, 1);
            }
        });
}

} // namespace Vulkan
