// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_ssgi.h"
#include "render/vk_rt_lighting.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include "common/path_util.h"

#include "bbgpu/shaders/ssgi_apply_comp.h"
#include "bbgpu/shaders/ssgi_gather_comp.h"
#include "bbgpu/shaders/ssgi_gather_rt_comp.h"
#include "render/vk_rtao.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "upscale/vk_scene_resolution.h"
#include "render/scheduler.h"
#include "vulkan/shader_module.h"
#include "textures/texture_cache.h"

namespace Vulkan {

namespace {
struct GatherConstants {
    std::array<float, 12> view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> half_size;
    float radius;
    float history;
    u32 frame;
    u32 directions;
    u32 steps;
    float emission; // x the RTAO's emission image (the transparents' light); 0: none
};
struct GatherRtConstants {
    std::array<float, 12> view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> half_size;
    std::array<float, 2> jitter;
    float radius;
    float history;
    u32 frame;
    u32 rays;
    u32 debug;
    u32 surface_parts_count;
    float clip;
    float max_radiance;
    float emission;
    float fire_light; // the fires as point lights (the RTAO's fire_light); 0: none
    u64 surface_cache;
    u64 surface_parts;
    float fire_shadow;
    float fire_range;
    u32 fire_count;
    float fire_size;
    float fire_height;
    float fire_wrap;
};
static_assert(sizeof(GatherRtConstants) == 168);
struct ApplyConstants {
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> half_size;
    float strength;
    u32 debug;
    float adaptive;
};
float EnvFloat(const char* name, float fallback) {
    const char* value = std::getenv(name);
    return value ? std::strtof(value, nullptr) : fallback;
}
u32 EnvU32(const char* name, u32 fallback) {
    const char* value = std::getenv(name);
    return value ? u32(std::strtoul(value, nullptr, 10)) : fallback;
}
} // namespace

ScreenSpaceGI::ScreenSpaceGI(const Instance& instance_, Scheduler& scheduler_,
                             VideoCore::TextureCache& texture_cache_, Runtime& runtime_,
                             CameraMotion& camera_motion_, SceneTargets& scene_targets_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_},
      runtime{runtime_}, camera_motion{camera_motion_}, scene_targets{scene_targets_} {
    const char* env = std::getenv("BB_SSGI");
    enabled = env && env[0] == '1' && camera_motion.Enabled();
    if (!enabled) {
        return;
    }
    debug = EnvU32("BB_SSGI_DEBUG", 0);
    rt = EnvU32("BB_SSGI_RT", 0) != 0;
    rays = std::clamp(EnvU32("BB_SSGI_RAYS", 2), 1u, 8u);
    strength = EnvFloat("BB_SSGI_STRENGTH", 1.0f);
    radius = EnvFloat("BB_SSGI_RADIUS", 2.0f);
    directions = std::clamp(EnvU32("BB_SSGI_DIRECTIONS", 4), 1u, 16u);
    steps = std::clamp(EnvU32("BB_SSGI_STEPS", 6), 1u, 32u);
    history = std::clamp(EnvFloat("BB_SSGI_HISTORY", 0.9f), 0.0f, 0.98f);
    adaptive = std::max(EnvFloat("BB_SSGI_ADAPTIVE", 4.0f), 0.0f);
    std::printf("SSGI: on, strength %.2f, radius %.2f, %u directions x %u steps, history %.2f%s\n",
                strength, radius, directions, steps, history, debug ? ", debug view" : "");
    if (const char* file = std::getenv("BB_SSGI_FILE")) {
        settings_path = file;
    } else {
        settings_path = (Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "ssgi.txt").string();
    }
    std::printf("SSGI: settings re-read from %s while the game runs\n", settings_path.c_str());

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
    const auto sampled = [](u32 binding) {
        return vk::DescriptorSetLayoutBinding{.binding = binding,
                                              .descriptorType = vk::DescriptorType::eSampledImage,
                                              .descriptorCount = 1,
                                              .stageFlags = vk::ShaderStageFlagBits::eCompute};
    };
    const auto storage = [](u32 binding) {
        return vk::DescriptorSetLayoutBinding{.binding = binding,
                                              .descriptorType = vk::DescriptorType::eStorageImage,
                                              .descriptorCount = 1,
                                              .stageFlags = vk::ShaderStageFlagBits::eCompute};
    };
    const std::array gather_bindings = {sampled(0), sampled(1), sampled(2),
                                        sampled(3), sampled(4), storage(5), sampled(6)};
    const std::array apply_bindings = {sampled(0), sampled(1), sampled(2), storage(3)};
    make_layouts(gather_bindings, sizeof(GatherConstants), gather_desc_layout, gather_layout);
    make_layouts(apply_bindings, sizeof(ApplyConstants), apply_desc_layout, apply_layout);
    gather_pipeline = make_pipeline(SSGI_GATHER_COMP, *gather_layout);
    apply_pipeline = make_pipeline(SSGI_APPLY_COMP, *apply_layout);
    // The ray-traced gather (rt=1): the same bindings and the RTAO's TLAS. Only with ray queries
    // on the device (BB_RTAO=1 on a GPU that has them).
    if (instance.IsRayQuerySupported()) {
        const std::array rt_bindings = {
            sampled(0), sampled(1), sampled(2), sampled(3), sampled(4), storage(5),
            vk::DescriptorSetLayoutBinding{.binding = 6,
                                           .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
                                           .descriptorCount = 1,
                                           .stageFlags = vk::ShaderStageFlagBits::eCompute},
            sampled(7), sampled(8)};
        make_layouts(rt_bindings, sizeof(GatherRtConstants), gather_rt_desc_layout, gather_rt_layout);
        gather_rt_pipeline = make_pipeline(SSGI_GATHER_RT_COMP, *gather_rt_layout);
        std::printf("SSGI: ray-traced gather available (rt=1 in ssgi.txt, experimental)\n");
    }
}

ScreenSpaceGI::~ScreenSpaceGI() {
    const auto device = instance.GetDevice();
    for (auto& entry : view_cache) {
        if (entry.view) {
            device.destroyImageView(entry.view);
        }
    }
}

void ScreenSpaceGI::OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal,
                                  VideoCore::ImageId albedo) {
    depth_id = depth;
    normal_id = normal;
    albedo_id = albedo;
}

void ScreenSpaceGI::OnFrameStart() {
    done_this_frame = false;
    ++frame;
    PollSettings();
}

void ScreenSpaceGI::PollSettings() {
    if (!enabled && settings_path.empty()) {
        return;
    }
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
        if (eq == std::string::npos || line[0] == '#') {
            continue;
        }
        const std::string key = line.substr(0, eq);
        const float value = std::strtof(line.c_str() + eq + 1, nullptr);
        if (key == "strength") strength = value;
        else if (key == "radius") radius = std::max(value, 0.05f);
        else if (key == "history") history = std::clamp(value, 0.0f, 0.98f);
        else if (key == "directions") directions = std::clamp(u32(value), 1u, 16u);
        else if (key == "steps") steps = std::clamp(u32(value), 1u, 32u);
        else if (key == "debug") debug = u32(std::max(value, 0.0f));
        else if (key == "rt") rt = value != 0.0f;
        else if (key == "rays") rays = std::clamp(u32(std::max(value, 1.0f)), 1u, 8u);
        else if (key == "rt_clip") rt_clip = std::max(value, 0.0f);
        else if (key == "rt_max") rt_max = std::max(value, 0.01f);
        else if (key == "enabled") enabled = value != 0.0f;
        else if (key == "adaptive") adaptive = std::max(value, 0.0f);
    }
    std::printf("SSGI: settings: enabled %d, strength %.2f, radius %.2f, %u x %u, history %.2f, "
                "adaptive %.2f, debug %u, rt %d (%u rays, clip %.2f, max %.2f)\n",
                enabled ? 1 : 0, strength, radius, directions, steps, history, adaptive, debug,
                rt ? 1 : 0, rays, rt_clip, rt_max);
}

void ScreenSpaceGI::OnOpaqueScene(VideoCore::ImageId color) {
    if (!enabled || done_this_frame) {
        return;
    }
    if (!depth_id || !normal_id || !albedo_id) {
        static u32 reported = 0;
        if (reported++ < 3) {
            std::printf("SSGI: opaque scene without G-buffer targets (depth %u normal %u albedo %u)\n",
                        depth_id.index, normal_id.index, albedo_id.index);
        }
        return;
    }
    done_this_frame = true;
    Record(color);
}

bool ScreenSpaceGI::EnsureResources(u32 w, u32 h) {
    if (width == w && height == h && gi_images[0]) {
        return true;
    }
    const auto device = instance.GetDevice();
    const auto allocator = instance.GetAllocator();
    const auto make_image = [&](VideoCore::HostImage& image, vk::UniqueImageView& view,
                                vk::Format format, u32 iw, u32 ih) {
        view.reset();
        image = VideoCore::HostImage(device, allocator);
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {iw, ih, 1},
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
            .format = format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
    };
    scheduler.Finish(); // the old images may be in flight
    for (u32 i = 0; i < 2; ++i) {
        make_image(gi_images[i], gi_views[i], vk::Format::eR16G16B16A16Sfloat, (w + 1) / 2,
                   (h + 1) / 2);
    }
    make_image(motion_image, motion_view, vk::Format::eR16G16Sfloat, w, h);
    width = w;
    height = h;
    history_valid = false;
    gi_index = 0;
    return true;
}

vk::ImageView ScreenSpaceGI::CachedView(const VideoCore::Image& image, vk::Format format,
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

void ScreenSpaceGI::Record(VideoCore::ImageId color_id) {
    auto& color = texture_cache.GetImage(color_id);
    auto& depth = texture_cache.GetImage(depth_id);
    auto& normal = texture_cache.GetImage(normal_id);
    auto& albedo = texture_cache.GetImage(albedo_id);
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat) {
        return;
    }
    // With live resolution the scene is drawn into the port's proxy targets (SceneTargets), the
    // texture-cache images hold nothing current: read and write the proxies, at their size.
    const bool reduced = scene_targets.Reduced() && scene_targets.EligibleScene(color);
    const u32 w = reduced ? scene_targets.Size().width : color.info.size.width;
    const u32 h = reduced ? scene_targets.Size().height : color.info.size.height;
    if (!reduced && (depth.info.size.width != w || depth.info.size.height != h ||
                     normal.info.size.width != w || albedo.info.size.width != w)) {
        static u32 reported = 0;
        if (reported++ < 3) {
            std::printf("SSGI: skipped: color %ux%u, depth %ux%u, normal %ux%u, albedo %ux%u\n", w,
                        h, depth.info.size.width, depth.info.size.height, normal.info.size.width,
                        normal.info.size.height, albedo.info.size.width, albedo.info.size.height);
        }
        return;
    }
    if (!EnsureResources(w, h)) {
        return;
    }
    static u32 passes = 0;
    if (passes++ % 600 == 0) {
        std::printf("SSGI: pass %u at %ux%u (%s, frame %u)\n", passes, w, h,
                    reduced ? "proxies" : "native images", frame);
    }

    scheduler.EndRendering();
    const auto compute = vk::PipelineStageFlagBits2::eComputeShader;
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    // Each input as (image, view) in General layout: the proxy or the native image.
    const auto resolve = [&](VideoCore::ImageId id, VideoCore::Image& image,
                             vk::ImageAspectFlags aspect, vk::AccessFlags2 access) {
        if (reduced && scene_targets.Eligible(image)) {
            VideoCore::ImageViewInfo ci;
            ci.format = image.info.pixel_format;
            const auto target = scene_targets.Read(id, ci, compute, access);
            return std::pair{target.image, target.view};
        }
        runtime.Transit(&image, vk::ImageLayout::eGeneral, compute, access);
        return std::pair{image.GetImage(), CachedView(image, image.info.pixel_format, aspect)};
    };
    const auto [depth_image, depth_view] =
        resolve(depth_id, depth, vk::ImageAspectFlagBits::eDepth, vk::AccessFlagBits2::eShaderRead);
    const auto [normal_image, normal_view] = resolve(normal_id, normal, vk::ImageAspectFlagBits::eColor,
                                                     vk::AccessFlagBits2::eShaderRead);
    const auto [albedo_image, albedo_view] = resolve(albedo_id, albedo, vk::ImageAspectFlagBits::eColor,
                                                     vk::AccessFlagBits2::eShaderRead);
    const auto [color_image, color_view] =
        resolve(color_id, color, vk::ImageAspectFlagBits::eColor, rw);
    runtime.FlushBarriers();

    // Our own images: the motion image is rewritten each frame, the gi images ping-pong.
    const u32 read = gi_index, write = gi_index ^ 1;
    gi_index = write;
    const auto barrier = [&](vk::Image image, vk::ImageLayout old_layout,
                             vk::AccessFlags2 src_access, vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 b{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = src_access,
            .dstStageMask = compute,
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
    barrier(vk::Image(motion_image), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
            vk::AccessFlagBits2::eShaderWrite);
    barrier(vk::Image(gi_images[write]), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
            vk::AccessFlagBits2::eShaderWrite);
    barrier(vk::Image(gi_images[read]),
            history_valid ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined, rw,
            vk::AccessFlagBits2::eShaderRead);

    // Motion vectors for the history (also commits the frame's camera).
    camera_motion.RecordMotion(depth_view, *motion_view, w, h);
    barrier(vk::Image(motion_image), vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderRead);

    // The transparents' light (fires) from the RTAO, last frame's; without it the colour is bound
    // in its place and weighted 0.
    const auto emission = lighting ? lighting->FrameEmission() : RtLighting::Emission{};
    const vk::ImageView emission_view = emission.view ? emission.view : color_view;
    const vk::ImageView lights_view = emission.lights ? emission.lights : emission_view;
    const GatherConstants gather{
        .view = camera_motion.View(),
        .proj = camera_motion.Projection(),
        .full_size = {float(w), float(h)},
        .half_size = {float((w + 1) / 2), float((h + 1) / 2)},
        .radius = radius,
        .history = history_valid ? history : 0.0f,
        .frame = frame,
        .directions = directions,
        .steps = steps,
        .emission = emission.strength,
    };
    const ApplyConstants apply{
        .proj = gather.proj,
        .full_size = gather.full_size,
        .half_size = gather.half_size,
        .strength = strength,
        .debug = debug != 0 ? 1u : 0u,
        .adaptive = adaptive,
    };
    history_valid = true;
    // The ray-traced gather: when asked for and the RTAO built its scene this frame; otherwise
    // the screen march (counted: a frame without the RTAO's pass falls back).
    const vk::AccelerationStructureKHR tlas =
        rt && gather_rt_pipeline && rtao ? rtao->FrameTlas() : vk::AccelerationStructureKHR{};
    if (rt) {
        ++(tlas ? rt_passes : rt_fallbacks);
        if ((rt_passes + rt_fallbacks) % 600 == 0) {
            std::printf("SSGI: ray-traced gather in %u of 600 passes (%u fell back to the march)\n",
                        rt_passes, rt_fallbacks);
            rt_passes = rt_fallbacks = 0;
        }
    }
    const auto surface = tlas ? rtao->FrameSurface() : RayTracedAO::SurfaceCache{};
    const GatherRtConstants gather_rt{
        .view = gather.view,
        .proj = gather.proj,
        .full_size = gather.full_size,
        .half_size = gather.half_size,
        .jitter = camera_motion.Jitter(),
        .radius = radius,
        .history = gather.history,
        .frame = frame,
        .rays = rays,
        .debug = debug,
        .surface_parts_count = surface.part_count,
        .clip = rt_clip,
        .max_radiance = rt_max,
        .emission = emission.strength,
        .fire_light = emission.light_strength,
        .surface_cache = surface.cache,
        .surface_parts = surface.parts,
        .fire_shadow = emission.light_shadow,
        .fire_range = emission.light_range,
        .fire_count = emission.lights ? emission.light_count : 0u,
        .fire_size = emission.light_size,
        .fire_height = emission.light_height,
        .fire_wrap = emission.light_wrap,
    };

    const vk::ImageView gi_read = *gi_views[read], gi_write = *gi_views[write];
    const vk::ImageView motion = *motion_view;
    scheduler.RecordCrumb(
        {.name = "ssgi"},
        [gather, gather_rt, tlas, apply, depth_view, normal_view, color_view, albedo_view, motion, emission_view,
         lights_view, gi_read, gi_write, gi_image = vk::Image(gi_images[write]), w, h,
         gather_pipeline = *gather_pipeline, gather_layout = *gather_layout,
         gather_rt_pipeline = *gather_rt_pipeline, gather_rt_layout = *gather_rt_layout,
         apply_pipeline = *apply_pipeline, apply_layout = *apply_layout,
         compute](vk::CommandBuffer cmdbuf) {
            const auto info = [](vk::ImageView view) {
                return vk::DescriptorImageInfo{.imageView = view,
                                               .imageLayout = vk::ImageLayout::eGeneral};
            };
            const auto write_of = [](u32 binding, vk::DescriptorType type,
                                     const vk::DescriptorImageInfo* image) {
                return vk::WriteDescriptorSet{.dstBinding = binding,
                                              .descriptorCount = 1,
                                              .descriptorType = type,
                                              .pImageInfo = image};
            };
            const auto sampled = vk::DescriptorType::eSampledImage;
            const auto storage = vk::DescriptorType::eStorageImage;
            {
                const auto d = info(depth_view), n = info(normal_view), c = info(color_view),
                           m = info(motion), p = info(gi_read), o = info(gi_write),
                           e = info(emission_view), l = info(lights_view);
                const std::array writes = {write_of(0, sampled, &d), write_of(1, sampled, &n),
                                           write_of(2, sampled, &c), write_of(3, sampled, &m),
                                           write_of(4, sampled, &p), write_of(5, storage, &o),
                                           write_of(6, sampled, &e)};
                if (tlas) {
                    const vk::WriteDescriptorSetAccelerationStructureKHR as_info{
                        .accelerationStructureCount = 1,
                        .pAccelerationStructures = &tlas,
                    };
                    std::array<vk::WriteDescriptorSet, 9> rt_writes{};
                    std::copy(writes.begin(), writes.begin() + 6, rt_writes.begin());
                    rt_writes[7] = write_of(7, sampled, &e);
                    rt_writes[8] = write_of(8, sampled, &l);
                    rt_writes[6] = vk::WriteDescriptorSet{
                        .pNext = &as_info,
                        .dstBinding = 6,
                        .descriptorCount = 1,
                        .descriptorType = vk::DescriptorType::eAccelerationStructureKHR};
                    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, gather_rt_pipeline);
                    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, gather_rt_layout, 0,
                                                rt_writes);
                    cmdbuf.pushConstants(gather_rt_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                         sizeof(gather_rt), &gather_rt);
                    cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 7) / 8, 1);
                } else {
                    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, gather_pipeline);
                    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, gather_layout, 0,
                                                writes);
                    cmdbuf.pushConstants(gather_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                         sizeof(gather), &gather);
                    cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 7) / 8, 1);
                }
            }
            const vk::ImageMemoryBarrier2 b{
                .srcStageMask = compute,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = compute,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = gi_image,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
            {
                const auto g = info(gi_write), d = info(depth_view), a = info(albedo_view),
                           c = info(color_view);
                const std::array writes = {write_of(0, sampled, &g), write_of(1, sampled, &d),
                                           write_of(2, sampled, &a), write_of(3, storage, &c)};
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, apply_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, apply_layout, 0,
                                            writes);
                cmdbuf.pushConstants(apply_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                     sizeof(apply), &apply);
                cmdbuf.dispatch((w + 7) / 8, (h + 7) / 8, 1);
            }
        });
}

} // namespace Vulkan
