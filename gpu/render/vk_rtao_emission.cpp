// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the transparents' light for the ray-traced lighting effects (experimental, emission= in
// rtao.txt, default 1; vk_rtao.h). Fires, sparks and glows are blended over the scene after the
// opaque pass, so SSGI, the probes and the reflections, which read the lit opaque scene, never saw
// them as light. The opaque scene (after SSGI and the probes added their bounce) is kept at a
// quarter of its size at the first transparent draw; at the first full-screen pass after the
// transparents (the fog composite) the scene minus that is the emission image. The effects read it
// the frame after, at a hit's screen position: one frame late and a little blurred, which suits
// flames; the light seems to come from the surface behind the flame.
#include "render/vk_rtao.h"
#include "render/vk_rtao_internal.h"

#include <cstdio>
#include <vk_mem_alloc.h>
#include "bbgpu/shaders/rtao_emission_comp.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

namespace Vulkan {
using namespace RtaoInternal;

namespace {
struct EmissionConstants {
    std::array<s32, 2> full_size;
    std::array<s32, 2> quarter_size;
    u32 mode;
    float min_saturation;
    s32 radius;
};
static_assert(sizeof(EmissionConstants) == 28);

void Barrier(Scheduler& scheduler, vk::Image image, vk::ImageLayout old_layout,
             vk::AccessFlags2 src_access, vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
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

bool RayTracedAO::EnsureEmission(u32 w, u32 h) {
    if (emit_image && emit_w == w && emit_h == h) {
        return true;
    }
    const auto device = instance.GetDevice();
    if (!emit_pipeline) {
        const auto binding = [](u32 index, vk::DescriptorType type) {
            return vk::DescriptorSetLayoutBinding{.binding = index,
                                                  .descriptorType = type,
                                                  .descriptorCount = 1,
                                                  .stageFlags = vk::ShaderStageFlagBits::eCompute};
        };
        const std::array bindings = {binding(0, vk::DescriptorType::eSampledImage),
                                     binding(1, vk::DescriptorType::eStorageImage),
                                     binding(2, vk::DescriptorType::eStorageImage)};
        emit_desc_layout = Check(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = u32(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                          .offset = 0,
                                          .size = sizeof(EmissionConstants)};
        emit_layout = Check(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &*emit_desc_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto module = CompileSPV(RTAO_EMISSION_COMP, device);
        emit_pipeline = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                    .layout = *emit_layout,
                }));
        device.destroyShaderModule(module);
    }
    scheduler.Finish(); // the old images may be in flight
    emit_opaque_view.reset();
    emit_view.reset();
    const auto make = [&](VideoCore::HostImage& image, vk::UniqueImageView& view) {
        image = VideoCore::HostImage(device, instance.GetAllocator());
        image.Create(vk::ImageCreateInfo{
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
        view = Check(device.createImageViewUnique({
            .image = vk::Image(image),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
    };
    make(emit_opaque, emit_opaque_view);
    make(emit_image, emit_view);
    emit_w = w;
    emit_h = h;
    emit_written = false;
    emit_frame = ~0u - 1;
    std::printf("RTAO: emission capture %ux%u (the transparents' light for the RT lighting)\n", w, h);
    return true;
}

void RayTracedAO::RecordEmission(VideoCore::ImageId color_id, u32 mode) {
    auto& color = texture_cache.GetImage(color_id);
    const u32 w = color.info.size.width, h = color.info.size.height;
    const u32 qw = (w + 3) / 4, qh = (h + 3) / 4;
    if (mode == 0 ? !EnsureEmission(qw, qh) : (emit_w != qw || emit_h != qh)) {
        return;
    }
    scheduler.EndRendering();
    const auto compute = vk::PipelineStageFlagBits2::eComputeShader;
    runtime.Transit(&color, vk::ImageLayout::eGeneral, compute, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const vk::ImageView color_view =
        CachedView(color, color.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    if (mode == 0) {
        Barrier(scheduler, vk::Image(emit_opaque), vk::ImageLayout::eUndefined, rw,
                vk::AccessFlagBits2::eShaderWrite);
    } else {
        Barrier(scheduler, vk::Image(emit_opaque), vk::ImageLayout::eGeneral,
                vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead);
        // Last frame's emission was read by this frame's effects (same queue) before this.
        Barrier(scheduler, vk::Image(emit_image),
                emit_written ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined, rw,
                vk::AccessFlagBits2::eShaderWrite);
    }
    const EmissionConstants constants{.full_size = {s32(w), s32(h)},
                                      .quarter_size = {s32(qw), s32(qh)},
                                      .mode = mode,
                                      .min_saturation = settings.emission_sat,
                                      .radius = s32(settings.emission_blur)};
    // mode 1 then the blur's two halves (2: emit -> opaque, 3: opaque -> emit) when it has a radius.
    const u32 last_mode = mode == 1 && settings.emission_blur > 0 ? 3 : mode;
    scheduler.RecordCrumb({.name = mode == 0 ? "rt emission opaque" : "rt emission"},
                          [constants, last_mode, color_view, opaque = *emit_opaque_view, emit = *emit_view, qw, qh,
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
                                  cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                                       sizeof(pc), &pc);
                                  cmdbuf.dispatch((qw + 7) / 8, (qh + 7) / 8, 1);
                                  if (pc.mode >= last_mode) {
                                      break;
                                  }
                                  ++pc.mode;
                                  const vk::MemoryBarrier2 b{
                                      .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .srcAccessMask = vk::AccessFlagBits2::eShaderWrite |
                                                       vk::AccessFlagBits2::eShaderRead,
                                      .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                      .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                                       vk::AccessFlagBits2::eShaderWrite};
                                  cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
                              }
                          });
    if (mode == 1) {
        Barrier(scheduler, vk::Image(emit_image), vk::ImageLayout::eGeneral,
                vk::AccessFlagBits2::eShaderWrite, vk::AccessFlagBits2::eShaderRead);
        emit_written = true;
        emit_frame = frame;
    }
}

void RayTracedAO::OnDrawAfterOpaque(VideoCore::ImageId color, bool scene_depth, bool blended,
                                    bool depth_write, u32 indices, u32 instances, u64 ps_hash) {
    ++emit_draws;
    if (emit_trace_lines && emit_trace_lines < 400) {
        ++emit_trace_lines;
        std::printf("  draw %u: target %s, scene depth %d, blended %d, depth write %d, %u indices x %u, ps %016llx\n",
                    emit_draws, !color ? "none" : color == emit_scene ? "scene" : "other", scene_depth ? 1 : 0,
                    blended ? 1 : 0, depth_write ? 1 : 0, indices, instances, (unsigned long long)ps_hash);
    }
    // The transparents are drawn with the scene depth; the first draw without it (the full-screen
    // fog composite, no depth target) or into another image ends them. With the fog the whole
    // scene would glow grey.
    if (settings.emission_at == 1 && color && (color != emit_scene || !scene_depth) && emit_frame != frame) {
        RecordEmission(emit_scene, 1);
        if (emit_trace_lines) {
            std::printf("  -> emission taken before draw %u\n", emit_draws);
        }
    }
}

void RayTracedAO::OnOpaqueSceneLit(VideoCore::ImageId color_id) {
    if (!enabled || !Active() || !color_id || settings.emission <= 0.0f || emit_snapshot_frame == frame) {
        return;
    }
    auto& color = texture_cache.GetImage(color_id);
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat || !resources_ready) {
        return;
    }
    // The last frame's outcome (per 600 frames in the log).
    if (++emit_frames == 600) {
        std::printf("RTAO: emission: made in %u of 600 frames (%u without; at %s)\n", emit_taken, emit_missed,
                    settings.emission_at == 1 ? "the end of the transparents" : "the first full-screen pass");
        emit_frames = emit_taken = emit_missed = 0;
    }
    if (emit_snapshot_frame + 1 == frame) {
        ++(emit_frame + 1 == frame ? emit_taken : emit_missed);
    }
    if (emit_trace_lines) {
        std::printf("  (end of the traced frame: %u draws)\n", emit_draws);
        emit_trace_lines = 0;
    }
    if (settings.emission_trace != emit_trace_seen) { // a new value: trace this frame's draws
        emit_trace_seen = settings.emission_trace;
        emit_trace_lines = 1;
        std::printf("RTAO: emission trace, frame %u (draws after the opaque scene; at %u):\n", frame,
                    settings.emission_at);
    }
    emit_snapshot_frame = frame;
    emit_scene = color_id;
    emit_draws = 0;
    RecordEmission(color_id, 0);
}

void RayTracedAO::OnSceneComposite(VideoCore::ImageId color_id) {
    if (settings.emission_at != 0 || !enabled || !color_id || emit_snapshot_frame != frame || emit_frame == frame || !emit_image) {
        return;
    }
    auto& color = texture_cache.GetImage(color_id);
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat) {
        return;
    }
    RecordEmission(color_id, 1);
}

} // namespace Vulkan
