// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the RT lighting passes' compute pipelines: push-descriptor layouts and one module each.

#pragma once

#include <span>
#include "common/types.h"
#include "vulkan/platform.h"
#include "vulkan/shader_module.h"
#include "vulkan/vk_common.h"

namespace Vulkan::RtPipeline {

inline vk::DescriptorSetLayoutBinding Binding(u32 index, vk::DescriptorType type) {
    return {.binding = index, .descriptorType = type, .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
}

/// A compute pipeline over `bindings` (push descriptors) and `push_size` bytes of push constants.
inline vk::UniquePipeline Make(vk::Device device, std::span<const u32> code,
                               std::span<const vk::DescriptorSetLayoutBinding> bindings, u32 push_size,
                               vk::UniqueDescriptorSetLayout& desc_layout, vk::UniquePipelineLayout& layout) {
    desc_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = push_size};
    layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(code, device);
    auto pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                .layout = *layout,
            }));
    device.destroyShaderModule(module);
    return pipeline;
}

} // namespace Vulkan::RtPipeline
