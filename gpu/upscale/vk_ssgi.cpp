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
#include "bbgpu/shaders/ssgi_denoise_atrous_comp.h"
#include "bbgpu/shaders/ssgi_denoise_temporal_comp.h"
#include "bbgpu/shaders/ssgi_gather_comp.h"
#include "bbgpu/shaders/ssgi_gather_rt_comp.h"
#include "render/vk_gpu_profiler.h"
#include "render/vk_rt_governor.h"
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
    float fire_light; // the fires as lights (rtlighting.txt fire_light); 0: none
    u64 surface_cache;
    u64 surface_parts;
    float fire_range;
    u32 fire_count; // 0: no fire lights bound
    float fire_wrap;
    float base_size;
    float base_height;
    float spark_size;
    u32 fire_rays;
    float large_base_size, large_base_height, large_spark_size;
    u32 record_rays; // > 0: shared rays (69), the RTAO trace's records shaded instead of traced
    float pad;
};
static_assert(sizeof(GatherRtConstants) == 192);
struct ApplyConstants {
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> half_size;
    float strength;
    u32 debug;
    float adaptive;
};
struct TemporalConstants {
    std::array<float, 12> to_prev;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> half_size;
    std::array<float, 2> jitter;
    float max_frames;
    float clip;
    u32 history_ok;
};
static_assert(sizeof(TemporalConstants) == 100);
struct AtrousConstants {
    std::array<float, 12> view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> half_size;
    std::array<float, 2> jitter;
    s32 step;
    u32 mode; // 1: the first pass, 2: the last
    float phi_luma;
    float phi_plane;
    float phi_normal;
};
static_assert(sizeof(AtrousConstants) == 108);
// a * b for 3x4 affine rows (b's fourth row 0 0 0 1).
std::array<float, 12> Compose(const std::array<float, 12>& a, const std::array<float, 12>& b) {
    std::array<float, 12> m{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            float v = 0.0f;
            for (int k = 0; k < 3; ++k) {
                v += a[r * 4 + k] * b[k * 4 + c];
            }
            m[r * 4 + c] = v + (c == 3 ? a[r * 4 + 3] : 0.0f);
        }
    }
    return m;
}
void MakeImage(const Instance& instance, VideoCore::HostImage& image, vk::UniqueImageView& view,
               vk::Format format, u32 width, u32 height) {
    const auto device = instance.GetDevice();
    view.reset();
    image = VideoCore::HostImage(device, instance.GetAllocator());
    image.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
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
}
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
            sampled(7), sampled(8), storage(9)};
        make_layouts(rt_bindings, sizeof(GatherRtConstants), gather_rt_desc_layout, gather_rt_layout);
        gather_rt_pipeline = make_pipeline(SSGI_GATHER_RT_COMP, *gather_rt_layout);
        // The colour denoiser (rt_denoise=1), for the ray-traced gather only.
        const std::array temporal_bindings = {sampled(0), sampled(1), sampled(2), sampled(3),
                                              sampled(4), sampled(5), storage(6), storage(7)};
        const std::array atrous_bindings = {sampled(0), sampled(1), storage(2)};
        make_layouts(temporal_bindings, sizeof(TemporalConstants), temporal_desc_layout, temporal_layout);
        make_layouts(atrous_bindings, sizeof(AtrousConstants), atrous_desc_layout, atrous_layout);
        temporal_pipeline = make_pipeline(SSGI_DENOISE_TEMPORAL_COMP, *temporal_layout);
        atrous_pipeline = make_pipeline(SSGI_DENOISE_ATROUS_COMP, *atrous_layout);
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

bool ScreenSpaceGI::RtOn() const noexcept {
    return rt && RtGovernor::Current().gi_rt;
}

void ScreenSpaceGI::OnFrameStart() {
    done_this_frame = false;
    ++frame;
    PollSettings();
    // The fire lights are made only for the ray-traced gather (the march does not read them).
    if (lighting) {
        lighting->SetFireReader(enabled && RtOn() && gather_rt_pipeline && rtao);
    }
    // Shared rays (69): the RTAO's trace records this frame's rays for the gather.
    if (enabled && shared_rays && RtOn() && gather_rt_pipeline && rtao) {
        rtao->RequestRecords(std::min(rays, RtGovernor::Current().gi_rays), radius);
    }
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
    std::string line, unknown;
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
        else if (key == "rt_denoise") rt_denoise = value != 0.0f;
        else if (key == "denoise_frames") denoise_frames = std::clamp(value, 1.0f, 255.0f);
        else if (key == "denoise_clip") denoise_clip = std::max(value, 0.0f);
        else if (key == "denoise_passes") denoise_passes = std::clamp(u32(std::max(value, 1.0f)), 1u, 5u);
        else if (key == "denoise_luma") denoise_luma = std::max(value, 0.01f);
        else if (key == "denoise_plane") denoise_plane = std::max(value, 0.001f);
        else if (key == "denoise_normal") denoise_normal = std::max(value, 0.0f);
        else if (key == "shared_rays") shared_rays = value != 0.0f;
        else unknown += key + ' ';
    }
    std::printf("SSGI: settings: enabled %d, strength %.2f, radius %.2f, %u x %u, history %.2f, "
                "adaptive %.2f, debug %u, rt %d (%u rays, clip %.2f, max %.2f)\n",
                enabled ? 1 : 0, strength, radius, directions, steps, history, adaptive, debug,
                rt ? 1 : 0, rays, rt_clip, rt_max);
    if (!unknown.empty()) {
        std::printf("SSGI: keys in %s skipped (unknown, or no number): %s\n", settings_path.c_str(), unknown.c_str());
    }
    if (rt_denoise) {
        std::printf("SSGI: rt denoiser on: %.0f frames, clip %.2f, %u passes (luma %.2f, plane %.3f, normal %.0f)\n",
                    denoise_frames, denoise_clip, denoise_passes, denoise_luma, denoise_plane, denoise_normal);
    }
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
    const auto make_image = [&](VideoCore::HostImage& image, vk::UniqueImageView& view,
                                vk::Format format, u32 iw, u32 ih) {
        MakeImage(instance, image, view, format, iw, ih);
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

void ScreenSpaceGI::EnsureDenoiseResources() {
    if (denoise_width == width && denoise_height == height && acc_images[0]) {
        return;
    }
    scheduler.Finish();
    const u32 hw = (width + 1) / 2, hh = (height + 1) / 2;
    for (u32 i = 0; i < 2; ++i) {
        MakeImage(instance, acc_images[i], acc_views[i], vk::Format::eR16G16B16A16Sfloat, hw, hh);
        MakeImage(instance, mom_images[i], mom_views[i], vk::Format::eR32G32B32A32Uint, hw, hh);
        MakeImage(instance, atrous_images[i], atrous_views[i], vk::Format::eR16G16B16A16Sfloat, hw, hh);
    }
    denoise_width = width;
    denoise_height = height;
    denoise_valid = false;
    denoise_index = 0;
    std::printf("SSGI: rt denoiser images %ux%u\n", hw, hh);
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
    if (RtOn() && rt_denoise && temporal_pipeline) {
        EnsureDenoiseResources();
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
        RtOn() && gather_rt_pipeline && rtao ? rtao->FrameTlas() : vk::AccelerationStructureKHR{};
    if (RtOn()) {
        ++(tlas ? rt_passes : rt_fallbacks);
        if ((rt_passes + rt_fallbacks) % 600 == 0) {
            std::printf("SSGI: ray-traced gather in %u of 600 passes (%u fell back to the march)\n",
                        rt_passes, rt_fallbacks);
            rt_passes = rt_fallbacks = 0;
        }
    }
    const auto surface = tlas ? rtao->FrameSurface() : RayTracedAO::SurfaceCache{};
    // The colour denoiser: the gather writes this frame's rays alone, the denoiser accumulates
    // and filters them into the same image. A frame without it ends its history.
    struct Denoise {
        bool on = false;
        u32 passes = 0;
        TemporalConstants temporal{};
        AtrousConstants atrous{};
        vk::ImageView acc_read, acc_write, mom_read, mom_write;
        std::array<vk::ImageView, 2> scratch{};
        vk::Pipeline temporal_pipeline, atrous_pipeline;
        vk::PipelineLayout temporal_layout, atrous_layout;
    } denoise;
    denoise.on = tlas && rt_denoise && temporal_pipeline && acc_images[0] && denoise_width == w &&
                 denoise_height == h;
    if (denoise.on) {
        const u32 dr = denoise_index, dw = denoise_index ^ 1;
        denoise_index = dw;
        for (auto* image : {&acc_images[dw], &mom_images[dw], &atrous_images[0], &atrous_images[1]}) {
            barrier(vk::Image(*image), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone, rw);
        }
        for (auto* image : {&acc_images[dr], &mom_images[dr]}) {
            barrier(vk::Image(*image), denoise_valid ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
                    rw, vk::AccessFlagBits2::eShaderRead);
        }
        denoise.passes = denoise_passes;
        denoise.temporal = {
            .to_prev = Compose(camera_motion.PrevView(), camera_motion.InvView()),
            .proj = gather.proj,
            .full_size = gather.full_size,
            .half_size = gather.half_size,
            .jitter = camera_motion.Jitter(),
            .max_frames = denoise_frames,
            .clip = denoise_clip,
            .history_ok = denoise_valid && camera_motion.HistoryValid() ? 1u : 0u,
        };
        denoise.atrous = {
            .view = gather.view,
            .proj = gather.proj,
            .full_size = gather.full_size,
            .half_size = gather.half_size,
            .jitter = camera_motion.Jitter(),
            .step = 1,
            .mode = 0,
            .phi_luma = denoise_luma,
            .phi_plane = denoise_plane,
            .phi_normal = denoise_normal,
        };
        denoise.acc_read = *acc_views[dr];
        denoise.acc_write = *acc_views[dw];
        denoise.mom_read = *mom_views[dr];
        denoise.mom_write = *mom_views[dw];
        denoise.scratch = {*atrous_views[0], *atrous_views[1]};
        denoise.temporal_pipeline = *temporal_pipeline;
        denoise.atrous_pipeline = *atrous_pipeline;
        denoise.temporal_layout = *temporal_layout;
        denoise.atrous_layout = *atrous_layout;
    }
    denoise_valid = denoise.on;
    // Shared rays (69): this frame's records of the RTAO's trace, when they cover this gather.
    const auto records = rtao && shared_rays && tlas ? rtao->FrameRecords() : RayTracedAO::Records{};
    const bool use_records = records.rays > 0 && records.width == u32(gather.half_size[0]) &&
                             records.height == u32(gather.half_size[1]);
    const vk::ImageView records_view = use_records ? records.view : rtao && tlas ? rtao->StandInRecords() : vk::ImageView{};
    if (shared_rays && tlas) {
        ++(use_records ? shared_passes : shared_misses);
        if ((shared_passes + shared_misses) % 600 == 0) {
            std::printf("SSGI: shared rays: the RTAO's records in %u of 600 gathers (%u traced its own)\n",
                        shared_passes, shared_misses);
            shared_passes = shared_misses = 0;
        }
    }
    const GatherRtConstants gather_rt{
        .view = gather.view,
        .proj = gather.proj,
        .full_size = gather.full_size,
        .half_size = gather.half_size,
        .jitter = camera_motion.Jitter(),
        .radius = radius,
        .history = denoise.on ? 0.0f : gather.history,
        .frame = frame,
        .rays = std::min(rays, RtGovernor::Current().gi_rays), // the governor's cap (71)
        .debug = debug,
        .surface_parts_count = surface.part_count,
        .clip = rt_clip,
        .max_radiance = rt_max,
        .emission = emission.strength,
        .fire_light = emission.light_strength,
        .surface_cache = surface.cache,
        .surface_parts = surface.parts,
        .fire_range = emission.light_range,
        .fire_count = emission.lights ? 1u : 0u,
        .fire_wrap = emission.light_wrap,
        .base_size = emission.base_size,
        .base_height = emission.base_height,
        .spark_size = emission.spark_size,
        .fire_rays = emission.light_rays,
        .large_base_size = emission.large_base_size,
        .large_base_height = emission.large_base_height,
        .large_spark_size = emission.large_spark_size,
        .record_rays = use_records ? records.rays : 0u,
    };

    const vk::ImageView gi_read = *gi_views[read], gi_write = *gi_views[write];
    const vk::ImageView motion = *motion_view;
    const ProfileScope profile{0x55601ull, "ssgi"};
    scheduler.RecordCrumb(
        {.name = "ssgi"},
        [gather, gather_rt, tlas, apply, denoise, depth_view, normal_view, color_view, albedo_view, motion, emission_view,
         lights_view, records_view, gi_read, gi_write, gi_image = vk::Image(gi_images[write]), w, h,
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
                    const auto rec = info(records_view);
                    std::array<vk::WriteDescriptorSet, 10> rt_writes{};
                    std::copy(writes.begin(), writes.begin() + 6, rt_writes.begin());
                    rt_writes[7] = write_of(7, sampled, &e);
                    rt_writes[8] = write_of(8, sampled, &l);
                    rt_writes[9] = write_of(9, storage, &rec);
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
                    if (denoise.on) {
                        const vk::MemoryBarrier2 mb{
                            .srcStageMask = compute,
                            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                            .dstStageMask = compute,
                            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        };
                        const auto settle = [&] {
                            cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &mb});
                        };
                        settle();
                        const auto r = info(gi_write), pa = info(denoise.acc_read), pm = info(denoise.mom_read),
                                   wa = info(denoise.acc_write), wm = info(denoise.mom_write);
                        const std::array t_writes = {write_of(0, sampled, &r), write_of(1, sampled, &d),
                                                     write_of(2, sampled, &n), write_of(3, sampled, &m),
                                                     write_of(4, sampled, &pa), write_of(5, sampled, &pm),
                                                     write_of(6, storage, &wa), write_of(7, storage, &wm)};
                        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, denoise.temporal_pipeline);
                        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, denoise.temporal_layout, 0,
                                                    t_writes);
                        cmdbuf.pushConstants(denoise.temporal_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                             sizeof(denoise.temporal), &denoise.temporal);
                        cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 7) / 8, 1);
                        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, denoise.atrous_pipeline);
                        for (u32 i = 0; i < denoise.passes; ++i) {
                            settle();
                            const bool last = i + 1 == denoise.passes;
                            const auto in = info(i == 0 ? denoise.acc_write : denoise.scratch[(i - 1) & 1]);
                            const auto out = info(last ? gi_write : denoise.scratch[i & 1]);
                            const std::array a_writes = {write_of(0, sampled, &in), write_of(1, sampled, &wm),
                                                         write_of(2, storage, &out)};
                            auto constants = denoise.atrous;
                            constants.step = 1 << i;
                            constants.mode = (i == 0 ? 1u : 0u) | (last ? 2u : 0u);
                            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, denoise.atrous_layout, 0,
                                                        a_writes);
                            cmdbuf.pushConstants(denoise.atrous_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                                 sizeof(constants), &constants);
                            cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 7) / 8, 1);
                        }
                    }
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
