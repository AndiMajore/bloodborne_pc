// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_rt_emission.h"

#include <algorithm>
#include <cstdio>
#include <vk_mem_alloc.h>
#include "bbgpu/shaders/rt_fire_debug_comp.h"
#include "bbgpu/shaders/rtao_emission_comp.h"
#include "bbgpu/shaders/rtao_fire_lights_comp.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_rt_pipeline.h"
#include "render/vk_rt_lights.h"
#include "render/vk_rtao.h"
#include "textures/texture_cache.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"

namespace Vulkan {

namespace {
struct EmissionConstants {
    std::array<s32, 2> full_size;
    std::array<s32, 2> quarter_size;
    u32 mode;
    float min_saturation;
    s32 radius;
};
static_assert(sizeof(EmissionConstants) == 28);
struct FireLightConstants {
    std::array<float, 12> inv_view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> jitter;
    std::array<s32, 2> quarter_size;
    s32 tiles_x;
    float min_peak;
    float pull;
    float smoothing;
    float pos_smoothing;
    u32 mode;
    u32 count;
    u32 bases;
    float base_smooth;
    float base_fade;
    float base_merge;
    float switch_ratio;
    float hold;
    float merge;
    u32 depth_mode;
    u32 anchor_count; // the game's fire lights below (bases snap to them)
    std::array<u32, 2> pad;
    std::array<std::array<float, 4>, 6> anchors; // world position, w its radius
};
static_assert(sizeof(FireLightConstants) == 256);
constexpr u32 MaxAnchors = 6;
struct FireDebugConstants {
    std::array<float, 12> view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> jitter;
    std::array<float, 4> up_radius;
    float height;
    u32 count;
    u32 base_count;
    float spark_radius; // 0: the lights drawn as cylinders
    float base_radius;
    float base_height;
    std::array<float, 2> pad;
};
static_assert(sizeof(FireDebugConstants) == 128);
constexpr u32 FireLightTile = 64; // quarter-size texels per light (rtao_fire_lights.comp)
constexpr u32 FireBases = 32;      // base slots after the lights (rtao_fire_lights.comp MaxBases)
constexpr auto Compute = vk::PipelineStageFlagBits2::eComputeShader;

void Barrier(Scheduler& scheduler, vk::Image image, vk::ImageLayout old_layout, vk::AccessFlags2 src_access,
             vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = Compute,
        .srcAccessMask = src_access,
        .dstStageMask = Compute,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS},
    };
    scheduler.Record([b](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
    });
}

u64 Key(VideoCore::ImageId id) {
    return id ? u64(id.index) + 1 : 0;
}
} // namespace

RtEmission::RtEmission(const Instance& instance, Scheduler& scheduler, VideoCore::TextureCache& texture_cache,
                       Runtime& runtime, CameraMotion& camera_motion, const RayTracedAO& rtao,
                       const RtLights& game_lights, const RtLightingSettings::Settings& settings)
    : instance{instance}, scheduler{scheduler}, texture_cache{texture_cache}, runtime{runtime},
      camera_motion{camera_motion}, rtao{rtao}, game_lights{game_lights}, settings{settings},
      views{instance, scheduler} {}

RtEmission::~RtEmission() = default;

RtEmission::Frame RtEmission::FrameEmission() const noexcept {
    if (!rtao.Active() || !settings.EmissionWanted() || !emit_view || emit_frame + 1 != frame) {
        return {};
    }
    if (settings.fire_light <= 0.0f || !lights_views[lights_out] || lights_frame + 1 != frame) {
        return {*emit_view, settings.emission};
    }
    Frame out{*emit_view, settings.emission, *lights_views[lights_out], lights_count, settings.fire_light,
              settings.fire_shadow, settings.fire_range, settings.fire_size, settings.fire_height, settings.fire_wrap};
    if (settings.fire_base > 0.0f && bases_frame == lights_frame) {
        out.base_count = FireBases;
        out.base_strength = settings.fire_base;
        out.spark_strength = settings.fire_spark;
        out.base_size = settings.fire_base_size;
        out.base_height = settings.fire_base_height;
        out.spark_size = settings.fire_spark_size;
    }
    return out;
}

void RtEmission::OnFrameStart(u32 next_frame) {
    frame = next_frame;
    trigger.Taken(); // a frame whose transparents never ended does not wait into the next
}

void RtEmission::OnGBufferPass(VideoCore::ImageId depth) {
    depth_id = depth;
}

bool RtEmission::Ensure(u32 w, u32 h) {
    if (emit_image && emit_w == w && emit_h == h) {
        return true;
    }
    const auto device = instance.GetDevice();
    if (!emit_pipeline) {
        const std::array bindings = {RtPipeline::Binding(0, vk::DescriptorType::eSampledImage),
                                     RtPipeline::Binding(1, vk::DescriptorType::eStorageImage),
                                     RtPipeline::Binding(2, vk::DescriptorType::eStorageImage)};
        emit_pipeline = RtPipeline::Make(device, RTAO_EMISSION_COMP, bindings, sizeof(EmissionConstants),
                                         emit_desc_layout, emit_layout);
        const std::array light_bindings = {RtPipeline::Binding(0, vk::DescriptorType::eSampledImage),
                                           RtPipeline::Binding(1, vk::DescriptorType::eSampledImage),
                                           RtPipeline::Binding(2, vk::DescriptorType::eStorageImage),
                                           RtPipeline::Binding(3, vk::DescriptorType::eStorageImage),
                                           RtPipeline::Binding(4, vk::DescriptorType::eStorageImage)};
        lights_pipeline = RtPipeline::Make(device, RTAO_FIRE_LIGHTS_COMP, light_bindings, sizeof(FireLightConstants),
                                           lights_desc_layout, lights_layout);
    }
    scheduler.Finish(); // the old images may be in flight
    opaque_view.reset();
    emit_view.reset();
    for (auto& view : lights_views) {
        view.reset();
    }
    const auto make = [&](VideoCore::HostImage& image, vk::UniqueImageView* image_views, u32 iw, u32 ih,
                          vk::Format format, u32 layers) {
        image = VideoCore::HostImage(device, instance.GetAllocator());
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {iw, ih, 1},
            .mipLevels = 1,
            .arrayLayers = layers,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        for (u32 layer = 0; layer < layers; ++layer) { // a 2D view per layer
            image_views[layer] = Check(device.createImageViewUnique({
                .image = vk::Image(image),
                .viewType = vk::ImageViewType::e2D,
                .format = format,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, layer, 1},
            }));
        }
    };
    make(opaque_image, &opaque_view, w, h, vk::Format::eR16G16B16A16Sfloat, 1);
    make(emit_image, &emit_view, w, h, vk::Format::eR16G16B16A16Sfloat, 1);
    tiles_x = (w + FireLightTile - 1) / FireLightTile;
    tiles_y = (h + FireLightTile - 1) / FireLightTile;
    lights_count = tiles_x * tiles_y;
    make(lights_image, lights_views.data(), lights_count + FireBases, 3, vk::Format::eR32G32B32A32Sfloat, 3);
    emit_w = w;
    emit_h = h;
    emit_written = false;
    emit_frame = ~0u - 1;
    lights_written = false;
    lights_frame = ~0u - 1;
    bases_frame = ~0u - 1;
    std::printf("RT emission: capture %ux%u (the transparents' light for the RT lighting)\n", w, h);
    return true;
}

void RtEmission::Record(VideoCore::ImageId color_id, u32 mode) {
    auto& color = texture_cache.GetImage(color_id);
    const u32 w = color.info.size.width, h = color.info.size.height;
    const u32 qw = (w + 3) / 4, qh = (h + 3) / 4;
    if (mode == 0 ? !Ensure(qw, qh) : (emit_w != qw || emit_h != qh)) {
        return;
    }
    scheduler.EndRendering();
    runtime.Transit(&color, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const vk::ImageView color_view = views.Get(color, color.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    if (mode == 0) {
        Barrier(scheduler, vk::Image(opaque_image), vk::ImageLayout::eUndefined, rw, vk::AccessFlagBits2::eShaderWrite);
    } else {
        Barrier(scheduler, vk::Image(opaque_image), vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
                vk::AccessFlagBits2::eShaderRead);
        // Last frame's emission was read by this frame's effects (same queue) before this.
        Barrier(scheduler, vk::Image(emit_image), emit_written ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
                rw, vk::AccessFlagBits2::eShaderWrite);
    }
    const EmissionConstants constants{.full_size = {s32(w), s32(h)},
                                      .quarter_size = {s32(qw), s32(qh)},
                                      .mode = mode,
                                      .min_saturation = settings.emission_sat,
                                      .radius = s32(settings.emission_blur)};
    // mode 1 then the blur's two halves (2: emit -> opaque, 3: opaque -> emit) when it has a radius.
    const u32 last_mode = mode == 1 && settings.emission_blur > 0 ? 3 : mode;
    scheduler.RecordCrumb({.name = mode == 0 ? "rt emission opaque" : "rt emission"},
                          [constants, last_mode, color_view, opaque = *opaque_view, emit = *emit_view, qw, qh,
                           pipeline = *emit_pipeline, layout = *emit_layout](vk::CommandBuffer cmdbuf) {
                              const vk::DescriptorImageInfo c{.imageView = color_view,
                                                              .imageLayout = vk::ImageLayout::eGeneral};
                              const vk::DescriptorImageInfo o{.imageView = opaque,
                                                              .imageLayout = vk::ImageLayout::eGeneral};
                              const vk::DescriptorImageInfo e{.imageView = emit,
                                                              .imageLayout = vk::ImageLayout::eGeneral};
                              const std::array writes = {
                                  vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage,
                                                         .pImageInfo = &c},
                                  vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eStorageImage,
                                                         .pImageInfo = &o},
                                  vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eStorageImage,
                                                         .pImageInfo = &e}};
                              cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
                              cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
                              auto pc = constants;
                              for (;;) {
                                  cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
                                  cmdbuf.dispatch((qw + 7) / 8, (qh + 7) / 8, 1);
                                  if (pc.mode >= last_mode) {
                                      break;
                                  }
                                  ++pc.mode;
                                  const vk::MemoryBarrier2 b{
                                      .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .srcAccessMask =
                                          vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
                                      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .dstAccessMask =
                                          vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
                                  cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
                              }
                          });
    if (mode == 1) {
        Barrier(scheduler, vk::Image(emit_image), vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
                vk::AccessFlagBits2::eShaderRead);
        emit_written = true;
        emit_frame = frame;
        if (settings.fire_light > 0.0f) {
            RecordFireLights();
        }
    }
}

void RtEmission::RecordFireLights() {
    if (!depth_id || !lights_views[0]) {
        return;
    }
    auto& depth = texture_cache.GetImage(depth_id);
    const auto proj = camera_motion.Projection();
    if (proj[0] == 0.0f || proj[1] == 0.0f || proj[3] == 0.0f) {
        return;
    }
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const vk::ImageView depth_view = views.Get(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    // Last frame's lights were read by this frame's SSGI (same queue) before this; they are blended
    // in when they are last frame's and no cut came between. This frame's go to the other layer.
    const bool continued = lights_written && lights_frame + 1 == frame && camera_motion.HistoryValid();
    const u32 prev = lights_out;
    lights_out = prev == 1 ? 2 : 1;
    Barrier(scheduler, vk::Image(lights_image), lights_written ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined,
            vk::AccessFlagBits2::eShaderRead, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    FireLightConstants constants{
        .inv_view = camera_motion.InvView(),
        .proj = proj,
        .full_size = {float(depth.info.size.width), float(depth.info.size.height)},
        .jitter = camera_motion.Jitter(),
        .quarter_size = {s32(emit_w), s32(emit_h)},
        .tiles_x = s32(tiles_x),
        .min_peak = settings.fire_min,
        .pull = settings.fire_pull,
        .smoothing = continued ? settings.fire_smooth : 0.0f,
        .pos_smoothing = continued ? settings.fire_smooth_pos : 0.0f,
        .count = lights_count,
        .bases = settings.fire_base > 0.0f ? FireBases : 0u,
        .base_smooth = settings.fire_base_smooth,
        .base_fade = settings.fire_base_fade,
        .base_merge = settings.fire_base_merge,
        .switch_ratio = settings.fire_switch,
        .hold = float(settings.fire_hold),
        .merge = settings.fire_merge,
        .depth_mode = settings.fire_depth,
    };
    if (constants.bases && settings.fire_base_anchor) {
        // The game's warm point lights nearest the camera (its own fire lights: orange, not the
        // grey fill lights), last frame's list.
        const auto inv_view = camera_motion.InvView();
        const std::array<float, 3> camera{inv_view[3], inv_view[7], inv_view[11]};
        std::array<std::pair<float, const RtLightRecords::SceneLight*>, 64> warm{};
        u32 n = 0;
        for (const auto& light : game_lights.FrameLights()) {
            const auto& c = light.color;
            if (light.spot || !(c[0] > 1.3f * c[2] && c[0] >= c[1] && c[1] >= c[2]) || n == warm.size()) {
                continue;
            }
            float d2 = 0.0f;
            for (int i = 0; i < 3; ++i) {
                d2 += (light.position[i] - camera[i]) * (light.position[i] - camera[i]);
            }
            warm[n++] = {d2, &light};
        }
        std::sort(warm.begin(), warm.begin() + n, [](const auto& a, const auto& b) { return a.first < b.first; });
        for (u32 i = 0; i < std::min(n, MaxAnchors); ++i) {
            const auto& l = *warm[i].second;
            constants.anchors[i] = {l.position[0], l.position[1], l.position[2], l.radius};
        }
        constants.anchor_count = std::min(n, MaxAnchors);
    }
    // The bases carry on from last frame's when there were any and no cut came between.
    const bool bases_continued = continued && bases_frame + 1 == frame;
    if (constants.bases) {
        bases_frame = frame;
    }
    scheduler.RecordCrumb({.name = "rt fire lights"},
                          [constants, depth_view, emit = *emit_view, raw = *lights_views[0],
                           prev_lights = *lights_views[prev], lights = *lights_views[lights_out], tx = tiles_x,
                           ty = tiles_y, bases_continued, pipeline = *lights_pipeline,
                           layout = *lights_layout](vk::CommandBuffer cmdbuf) {
                              const auto info = [](vk::ImageView v) {
                                  return vk::DescriptorImageInfo{.imageView = v, .imageLayout = vk::ImageLayout::eGeneral};
                              };
                              const auto e = info(emit), d = info(depth_view), r = info(raw), p = info(prev_lights),
                                         l = info(lights);
                              const std::array writes = {
                                  vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage,
                                                         .pImageInfo = &e},
                                  vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage,
                                                         .pImageInfo = &d},
                                  vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eStorageImage,
                                                         .pImageInfo = &r},
                                  vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eStorageImage,
                                                         .pImageInfo = &p},
                                  vk::WriteDescriptorSet{.dstBinding = 4, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eStorageImage,
                                                         .pImageInfo = &l}};
                              cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
                              cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
                              // Mode 0 per tile into layer 0, then mode 1 per light: matched with the
                              // frame before's into this frame's layer.
                              auto pc = constants;
                              cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
                              cmdbuf.dispatch(tx, ty, 1);
                              const vk::MemoryBarrier2 b{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                                         .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                                         .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                                         .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
                              cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
                              pc.mode = 1;
                              cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
                              cmdbuf.dispatch((pc.count + 255) / 256, 1, 1);
                              if (pc.bases) { // mode 2: the bases from this frame's lights
                                  const vk::MemoryBarrier2 rw{
                                      .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
                                  cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &rw});
                                  pc.mode = 2;
                                  pc.smoothing = bases_continued ? 1.0f : 0.0f;
                                  cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
                                  cmdbuf.dispatch(1, 1, 1);
                              }
                          });
    Barrier(scheduler, vk::Image(lights_image), vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
            vk::AccessFlagBits2::eShaderRead);
    lights_written = true;
    lights_frame = frame;
}

void RtEmission::RecordFireDebug(VideoCore::ImageId color_id) {
    // Last frame's lights (this frame's are made after the transparents).
    if (!depth_id || !lights_views[lights_out] || lights_frame + 1 != frame) {
        return;
    }
    auto& color = texture_cache.GetImage(color_id);
    auto& depth = texture_cache.GetImage(depth_id);
    const u32 w = color.info.size.width, h = color.info.size.height;
    if (depth.info.size.width != w || depth.info.size.height != h) {
        return;
    }
    if (!debug_pipeline) {
        const std::array bindings = {RtPipeline::Binding(0, vk::DescriptorType::eSampledImage),
                                     RtPipeline::Binding(1, vk::DescriptorType::eSampledImage),
                                     RtPipeline::Binding(2, vk::DescriptorType::eStorageImage)};
        debug_pipeline = RtPipeline::Make(instance.GetDevice(), RT_FIRE_DEBUG_COMP, bindings,
                                          sizeof(FireDebugConstants), debug_desc_layout, debug_layout);
    }
    scheduler.EndRendering();
    runtime.Transit(&color, vk::ImageLayout::eGeneral, Compute,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const vk::ImageView color_view = views.Get(color, color.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const vk::ImageView depth_view = views.Get(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const auto view = camera_motion.View();
    // The world's up (y) in view space: the view rows' y column.
    const FireDebugConstants constants{
        .view = view,
        .proj = camera_motion.Projection(),
        .full_size = {float(w), float(h)},
        .jitter = camera_motion.Jitter(),
        .up_radius = {view[1], view[5], view[9], settings.fire_size},
        .height = settings.fire_height,
        .count = lights_count,
        .base_count = settings.fire_base > 0.0f && bases_frame == lights_frame ? FireBases : 0u,
        .spark_radius = settings.fire_base > 0.0f ? settings.fire_spark_size : 0.0f,
        .base_radius = settings.fire_base_size,
        .base_height = settings.fire_base_height,
    };
    scheduler.RecordCrumb({.name = "rt fire debug"},
                          [constants, color_view, depth_view, lights = *lights_views[lights_out], w, h,
                           pipeline = *debug_pipeline, layout = *debug_layout](vk::CommandBuffer cmdbuf) {
                              const auto info = [](vk::ImageView v) {
                                  return vk::DescriptorImageInfo{.imageView = v, .imageLayout = vk::ImageLayout::eGeneral};
                              };
                              const auto d = info(depth_view), l = info(lights), c = info(color_view);
                              const std::array writes = {
                                  vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                                  vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &l},
                                  vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                                         .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &c}};
                              cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
                              cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
                              cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants), &constants);
                              cmdbuf.dispatch((w + 7) / 8, (h + 7) / 8, 1);
                              const vk::MemoryBarrier2 b{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                                         .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                                                         .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                                         .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
                              cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
                          });
}

void RtEmission::OnDrawAfterOpaque(VideoCore::ImageId color, bool scene_depth, bool blended, bool depth_write,
                                   u32 indices, u32 instances, u64 ps_hash) {
    ++trace_draws;
    if (trace_lines && trace_lines < 400) {
        ++trace_lines;
        std::printf("  draw %u: target %s, scene depth %d, blended %d, depth write %d, %u indices x %u, ps %016llx\n",
                    trace_draws, !color ? "none" : color == scene ? "scene" : "other", scene_depth ? 1 : 0,
                    blended ? 1 : 0, depth_write ? 1 : 0, indices, instances, (unsigned long long)ps_hash);
    }
    if (settings.emission_at == 1 && trigger.OnDraw(Key(color), scene_depth)) {
        Record(scene, 1);
        if (trace_lines) {
            std::printf("  -> emission taken before draw %u\n", trace_draws);
        }
    }
}

void RtEmission::OnOpaqueSceneLit(VideoCore::ImageId color_id) {
    if (!rtao.Active() || !color_id || !settings.EmissionWanted() || snapshot_frame == frame) {
        return;
    }
    auto& color = texture_cache.GetImage(color_id);
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat) {
        return;
    }
    // The last frame's outcome (per 600 frames in the log).
    if (++report_frames == 600) {
        std::printf("RT emission: made in %u of 600 frames (%u without; at %s)\n", taken, missed,
                    settings.emission_at == 1 ? "the end of the transparents" : "the first full-screen pass");
        report_frames = taken = missed = 0;
    }
    if (snapshot_frame + 1 == frame) {
        ++(emit_frame + 1 == frame ? taken : missed);
    }
    if (trace_lines) {
        std::printf("  (end of the traced frame: %u draws)\n", trace_draws);
        trace_lines = 0;
    }
    if (settings.emission_trace != trace_seen) { // a new value: trace this frame's draws
        trace_seen = settings.emission_trace;
        trace_lines = 1;
        std::printf("RT emission: trace, frame %u (draws after the opaque scene; at %u):\n", frame, settings.emission_at);
    }
    snapshot_frame = frame;
    scene = color_id;
    trace_draws = 0;
    trigger.Begin(Key(color_id));
    if (settings.fire_debug) {
        RecordFireDebug(color_id); // before the opaque base is kept: not counted as light
    }
    Record(color_id, 0);
}

void RtEmission::OnSceneComposite(VideoCore::ImageId color_id) {
    if (settings.emission_at != 0 || !color_id || snapshot_frame != frame || emit_frame == frame || !emit_image) {
        return;
    }
    auto& color = texture_cache.GetImage(color_id);
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat) {
        return;
    }
    trigger.Taken();
    Record(color_id, 1);
}

} // namespace Vulkan
