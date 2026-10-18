// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_rt_reflect.h"
#include "render/vk_rt_lighting.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>
#include "common/path_util.h"

#include "bbgpu/shaders/rt_reflect_comp.h"
#include "bbgpu/shaders/rt_reflect_copy_comp.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_rtao.h"
#include "textures/texture_cache.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

namespace Vulkan {

namespace {
struct TraceConstants {
    std::array<float, 12> view;
    std::array<float, 12> inv_view;
    std::array<float, 12> prev_view;
    std::array<float, 4> proj;
    std::array<float, 4> prev_proj;
    std::array<float, 2> full_size;
    std::array<float, 2> jitter;
    std::array<float, 2> ssr_size;
    std::array<float, 2> lit_size;
    float far;
    float strength;
    u32 mode;
    u32 debug;
    u32 surface_parts_count;
    float emission; // x the RTAO's emission image (the transparents' light); 0: none
    u64 surface_cache;
    u64 surface_parts;
    float soften;   // x the material's lobe (its gloss in the G-buffer): the ray's cone; 0: a mirror
    u32 frame;
};
static_assert(sizeof(TraceConstants) == 256); // push constants: 256 bytes on RADV and NVIDIA
struct CopyConstants {
    std::array<s32, 2> full_size;
    std::array<s32, 2> half_size;
};
} // namespace

RtReflections::RtReflections(const Instance& instance_, Scheduler& scheduler_,
                             VideoCore::TextureCache& texture_cache_, Runtime& runtime_,
                             CameraMotion& camera_motion_, const RayTracedAO& rtao_,
                             const RtLighting& lighting_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, runtime{runtime_},
      camera_motion{camera_motion_}, rtao{rtao_}, lighting{lighting_} {
    const char* env = std::getenv("BB_RT_REFLECT");
    enabled = env && env[0] == '1' && rtao.Enabled() && instance.IsRayQuerySupported() &&
              instance.GetPhysicalDevice().getProperties().limits.maxPushConstantsSize >=
                  sizeof(TraceConstants);
    if (!enabled) {
        if (env && env[0] == '1') {
            std::printf("RT reflections: off (needs the RTAO with ray queries and 256 bytes of push "
                        "constants)\n");
        }
        return;
    }
    if (const char* file = std::getenv("BB_RT_REFLECT_FILE")) {
        settings_path = file;
    } else {
        settings_path =
            (Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "rtreflect.txt").string();
    }
    std::printf("RT reflections: on (experimental), settings in %s\n", settings_path.c_str());

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
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
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
    const std::array trace_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eSampledImage),
                                       binding(2, DT::eSampledImage), binding(3, DT::eStorageImage),
                                       binding(4, DT::eAccelerationStructureKHR), binding(5, DT::eSampledImage),
                                       binding(6, DT::eSampledImage)};
    const std::array copy_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eStorageImage)};
    make_layouts(trace_bindings, sizeof(TraceConstants), trace_desc_layout, trace_layout);
    make_layouts(copy_bindings, sizeof(CopyConstants), copy_desc_layout, copy_layout);
    trace_pipeline = make_pipeline(RT_REFLECT_COMP, *trace_layout);
    copy_pipeline = make_pipeline(RT_REFLECT_COPY_COMP, *copy_layout);
}

RtReflections::~RtReflections() {
    const auto device = instance.GetDevice();
    for (auto& entry : view_cache) {
        if (entry.view) {
            device.destroyImageView(entry.view);
        }
    }
}

void RtReflections::OnFrameStart() {
    if (!enabled) {
        return;
    }
    done_this_frame = copied_this_frame = replaced_this_frame = false;
    ssr_target = last_color = {};
    ++frame;
    PollSettings();
    if (frame % 600 == 0 && (passes || skipped)) {
        std::printf("RT reflections: %u passes in 600 frames, %u skipped (no RTAO scene, no "
                    "history or a size mismatch); mode %u, far %.0f m; the game's SSR left out in %u\n",
                    passes, skipped, mode, far, replaced);
        passes = skipped = replaced = 0;
    }
}

void RtReflections::PollSettings() {
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
    std::string line;
    while (std::getline(file, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos || line.empty() || line[0] == '#') {
            continue;
        }
        const std::string key = line.substr(0, eq);
        const float value = std::strtof(line.c_str() + eq + 1, nullptr);
        if (key == "enabled") active = value != 0.0f;
        else if (key == "mode") mode = std::clamp(u32(std::max(value, 1.0f)), 1u, 2u);
        else if (key == "far") far = std::clamp(value, 1.0f, 500.0f);
        else if (key == "strength") strength = std::clamp(value, 0.0f, 1.0f);
        else if (key == "debug") debug = std::clamp(u32(std::max(value, 0.0f)), 0u, 2u);
        else if (key == "replace_ssr") replace_ssr = value != 0.0f;
        else if (key == "soften") soften = std::clamp(value, 0.0f, 4.0f);
    }
    std::printf("RT reflections: settings: enabled %d, mode %u, far %.0f m, strength %.2f, debug %u, "
                "replace_ssr %d, soften %.2f\n",
                active ? 1 : 0, mode, far, strength, debug, replace_ssr ? 1 : 0, soften);
}

void RtReflections::OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal,
                                  VideoCore::ImageId material) {
    depth_id = depth;
    normal_id = normal;
    material_id = material;
}

bool RtReflections::SkipGameSsr() {
    // Only when the trace will run (its conditions in Record): else the target keeps stale content.
    if (!enabled || !active || !replace_ssr || done_this_frame || !rtao.FrameTlas() || !copy_valid ||
        !camera_motion.HistoryValid() || !depth_id || !normal_id || !ssr_target) {
        return false;
    }
    const auto& ssr = texture_cache.GetImage(ssr_target);
    const auto& depth = texture_cache.GetImage(depth_id);
    const auto& normal = texture_cache.GetImage(normal_id);
    if (ssr.info.pixel_format != vk::Format::eR16G16B16A16Sfloat ||
        normal.info.size.width != depth.info.size.width) {
        return false;
    }
    replaced_this_frame = true;
    ++replaced;
    return true;
}

void RtReflections::OnColorTarget(VideoCore::ImageId color, u64 ps_hash) {
    if (!enabled) {
        return;
    }
    if (color != last_color) {
        // The previous pass ended. Was it the game's SSR?
        if (last_color && last_color == ssr_target && !done_this_frame && active) {
            done_this_frame = true;
            Record(ssr_target);
        }
        last_color = color;
    }
    if (ps_hash == SsrShader && color) {
        ssr_target = color;
    }
}

vk::ImageView RtReflections::CachedView(const VideoCore::Image& image, vk::Format format,
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

bool RtReflections::EnsureCopy(u32 w, u32 h) {
    if (copy_image && copy_w == w && copy_h == h) {
        return true;
    }
    const auto device = instance.GetDevice();
    scheduler.Finish(); // the old copy may be in flight
    copy_view.reset();
    copy_image = VideoCore::HostImage(device, instance.GetAllocator());
    copy_image.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    copy_view = Check(device.createImageViewUnique({
        .image = vk::Image(copy_image),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    copy_w = w;
    copy_h = h;
    copy_valid = false;
    return true;
}

namespace {
void ImageBarrier(Scheduler& scheduler, vk::Image image, vk::ImageLayout old_layout,
                  vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                  vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    scheduler.Record([b](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
    });
}
} // namespace

void RtReflections::OnOpaqueScene(VideoCore::ImageId color_id) {
    if (!enabled || !active || copied_this_frame || !color_id) {
        return;
    }
    copied_this_frame = true;
    auto& color = texture_cache.GetImage(color_id);
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat) {
        return;
    }
    const u32 w = color.info.size.width, h = color.info.size.height;
    const u32 hw = (w + 1) / 2, hh = (h + 1) / 2;
    EnsureCopy(hw, hh);
    scheduler.EndRendering();
    const auto compute = vk::PipelineStageFlagBits2::eComputeShader;
    runtime.Transit(&color, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const vk::ImageView color_view = CachedView(color, color.info.pixel_format,
                                                vk::ImageAspectFlagBits::eColor);
    // Written after this frame's reflections read it (same queue): the copy's last reader first.
    ImageBarrier(scheduler, vk::Image(copy_image),
                 copy_valid ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined, compute,
                 vk::AccessFlagBits2::eShaderRead, vk::AccessFlagBits2::eShaderWrite);
    const CopyConstants constants{.full_size = {s32(w), s32(h)}, .half_size = {s32(hw), s32(hh)}};
    scheduler.RecordCrumb({.name = "rt reflect copy"},
                          [constants, color_view, copy = *copy_view, hw, hh,
                           pipeline = *copy_pipeline, layout = *copy_layout](vk::CommandBuffer cmdbuf) {
                              const vk::DescriptorImageInfo in{.imageView = color_view,
                                                               .imageLayout = vk::ImageLayout::eGeneral};
                              const vk::DescriptorImageInfo out{.imageView = copy,
                                                                .imageLayout = vk::ImageLayout::eGeneral};
                              const std::array writes = {
                                  vk::WriteDescriptorSet{.dstBinding = 0,
                                                         .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage,
                                                         .pImageInfo = &in},
                                  vk::WriteDescriptorSet{.dstBinding = 1,
                                                         .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eStorageImage,
                                                         .pImageInfo = &out}};
                              cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
                              cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0,
                                                          writes);
                              cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                                   sizeof(constants), &constants);
                              cmdbuf.dispatch((hw + 7) / 8, (hh + 7) / 8, 1);
                          });
    ImageBarrier(scheduler, vk::Image(copy_image), vk::ImageLayout::eGeneral, compute,
                 vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead);
    copy_valid = true;
}

void RtReflections::Record(VideoCore::ImageId ssr_id) {
    const vk::AccelerationStructureKHR tlas = rtao.FrameTlas();
    if (!tlas || !copy_valid || !camera_motion.HistoryValid() || !depth_id || !normal_id) {
        ++skipped;
        return;
    }
    auto& ssr = texture_cache.GetImage(ssr_id);
    auto& depth = texture_cache.GetImage(depth_id);
    auto& normal = texture_cache.GetImage(normal_id);
    if (ssr.info.pixel_format != vk::Format::eR16G16B16A16Sfloat ||
        normal.info.size.width != depth.info.size.width) {
        ++skipped;
        return;
    }
    ++passes;
    static u32 first = 0;
    if (first++ == 0) {
        std::printf("RT reflections: first pass: SSR target %ux%u, depth %ux%u, copy %ux%u\n",
                    ssr.info.size.width, ssr.info.size.height, depth.info.size.width,
                    depth.info.size.height, copy_w, copy_h);
    }
    scheduler.EndRendering();
    const auto compute = vk::PipelineStageFlagBits2::eComputeShader;
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&normal, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&ssr, vk::ImageLayout::eGeneral, compute, rw);
    // The material target (its gloss), when it has the depth's size: the rays' cone (soften).
    VideoCore::Image* material = nullptr;
    if (material_id && soften > 0.0f) {
        auto& m = texture_cache.GetImage(material_id);
        if (m.info.pixel_format == vk::Format::eR8G8B8A8Unorm && m.info.size.width == depth.info.size.width &&
            m.info.size.height == depth.info.size.height) {
            material = &m;
            runtime.Transit(material, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
        }
    }
    runtime.FlushBarriers();
    const vk::ImageView depth_view =
        CachedView(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const vk::ImageView normal_view =
        CachedView(normal, normal.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView ssr_view =
        CachedView(ssr, ssr.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView material_view =
        material ? CachedView(*material, material->info.pixel_format, vk::ImageAspectFlagBits::eColor)
                 : vk::ImageView{};
    const u32 sw = ssr.info.size.width, sh = ssr.info.size.height;
    const auto surface = rtao.FrameSurface();
    // Last frame's, as the lit copy; without it the copy is bound in its place, x 0.
    const auto emission = lighting.FrameEmission();
    const vk::ImageView emission_view = emission.view ? emission.view : *copy_view;
    const TraceConstants constants{
        .view = camera_motion.View(),
        .inv_view = camera_motion.InvView(),
        .prev_view = camera_motion.PrevView(),
        .proj = camera_motion.Projection(),
        .prev_proj = camera_motion.PrevProjection(),
        .full_size = {float(depth.info.size.width), float(depth.info.size.height)},
        .jitter = camera_motion.Jitter(),
        .ssr_size = {float(sw), float(sh)},
        .lit_size = {float(copy_w), float(copy_h)},
        .far = far,
        .strength = strength,
        .mode = replaced_this_frame ? 3u : mode, // 3: the game's SSR left out, the whole target ours
        .debug = debug,
        .surface_parts_count = surface.part_count,
        .emission = emission.strength,
        .surface_cache = surface.cache,
        .surface_parts = surface.parts,
        .soften = material_view ? soften : 0.0f,
        .frame = frame,
    };
    scheduler.RecordCrumb(
        {.name = "rt reflect"},
        [constants, depth_view, normal_view, ssr_view, lit = *copy_view, emission_view, tlas, sw, sh,
         gloss = material_view ? material_view : normal_view,
         pipeline = *trace_pipeline, layout = *trace_layout](vk::CommandBuffer cmdbuf) {
            const auto info = [](vk::ImageView view) {
                return vk::DescriptorImageInfo{.imageView = view, .imageLayout = vk::ImageLayout::eGeneral};
            };
            const auto d = info(depth_view), n = info(normal_view), l = info(lit), s = info(ssr_view),
                       e = info(emission_view), g = info(gloss);
            const vk::WriteDescriptorSetAccelerationStructureKHR as_info{
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &tlas,
            };
            const std::array writes = {
                vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &n},
                vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &l},
                vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &s},
                vk::WriteDescriptorSet{.pNext = &as_info, .dstBinding = 4, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
                vk::WriteDescriptorSet{.dstBinding = 5, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &e},
                vk::WriteDescriptorSet{.dstBinding = 6, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &g}};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants),
                                 &constants);
            cmdbuf.dispatch((sw + 7) / 8, (sh + 3) / 4, 1);
        });
}

} // namespace Vulkan
