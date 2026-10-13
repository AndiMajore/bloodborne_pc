// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_rtao.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include "common/path_util.h"

#include <vk_mem_alloc.h>

#include "video_core/host_shaders/rtao_apply_comp.h"
#include "video_core/host_shaders/rtao_blur_comp.h"
#include "video_core/host_shaders/rtao_expand_comp.h"
#include "video_core/host_shaders/rtao_trace_comp.h"
#include "video_core/renderer_vulkan/vk_camera_motion.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_object_motion.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {
struct ExpandConstants {
    u64 positions_address;
    u64 pad; // std430: the vec4 below is 16-byte aligned in the shader's push block
    std::array<float, 4> proj;
    u32 num_geometries;
    u32 total_triangles;
    float far_cut;
    u32 pad1;
};
struct TraceConstants {
    std::array<float, 12> view;
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> half_size;
    std::array<float, 2> jitter;
    float radius;
    float history;
    u32 rays;
    u32 frame;
    u32 mode; // 0 ao, 2 primary-ray test
    float snap_far;
    float ao_far;
};
struct ApplyConstants {
    std::array<float, 4> proj;
    std::array<float, 2> full_size;
    std::array<float, 2> half_size;
    float strength;
    u32 debug;
    float power;
    u32 blur;
    u32 replace;
    u32 pad1;
};
static_assert(sizeof(ExpandConstants) == 48 && offsetof(ExpandConstants, far_cut) == 40);
static_assert(sizeof(TraceConstants) == 116 && offsetof(TraceConstants, ao_far) == 112);
struct BlurConstants {
    std::array<float, 2> half_size;
    std::array<s32, 2> dir;
    float sigma;
    u32 pad;
};
static_assert(sizeof(ApplyConstants) == 56);
static_assert(sizeof(BlurConstants) == 24);
float EnvFloat(const char* name, float fallback) {
    const char* value = std::getenv(name);
    return value ? std::strtof(value, nullptr) : fallback;
}
u32 EnvU32(const char* name, u32 fallback) {
    const char* value = std::getenv(name);
    return value ? u32(std::strtoul(value, nullptr, 10)) : fallback;
}
constexpr auto AsBuild = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;
constexpr auto Compute = vk::PipelineStageFlagBits2::eComputeShader;
} // namespace

RayTracedAO::RayTracedAO(const Instance& instance_, Scheduler& scheduler_,
                         VideoCore::TextureCache& texture_cache_, Runtime& runtime_,
                         CameraMotion& camera_motion_, ObjectMotion& object_motion_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_},
      runtime{runtime_}, camera_motion{camera_motion_}, object_motion{object_motion_} {
    const char* env = std::getenv("BB_RTAO");
    if (!env || env[0] != '1') {
        return;
    }
    if (!instance.IsRayQuerySupported()) {
        std::puts("RTAO: ray queries unsupported by the driver, off");
        return;
    }
    if (!instance.IsRayTracingPositionFetchSupported()) {
        std::puts("RTAO: VK_KHR_ray_tracing_position_fetch unsupported by the driver, off");
        return;
    }
    if (!object_motion.Enabled() || !camera_motion.Enabled()) {
        std::puts("RTAO: needs the object motion vectors (menu: object motion, with an upscaler), off");
        return;
    }
    debug = std::min(EnvU32("BB_RTAO_DEBUG", 0), 4u);
    strength = std::clamp(EnvFloat("BB_RTAO_STRENGTH", 1.0f), 0.0f, 1.0f);
    radius = std::max(EnvFloat("BB_RTAO_RADIUS", 1.5f), 0.05f);
    power = std::clamp(EnvFloat("BB_RTAO_POWER", 2.0f), 0.1f, 8.0f);
    history = std::clamp(EnvFloat("BB_RTAO_HISTORY", 0.5f), 0.0f, 0.98f);
    sigma = std::clamp(EnvFloat("BB_RTAO_SIGMA", 1.5f), 0.0f, 4.0f);
    rays = std::clamp(EnvU32("BB_RTAO_RAYS", 6), 1u, 32u);
    max_triangles = std::clamp(EnvU32("BB_RTAO_MAX_TRIANGLES", 3000000), 10000u, 16000000u);
    stage = std::clamp(EnvU32("BB_RTAO_STAGE", 5), 0u, 5u);

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
    const std::array expand_bindings = {binding(0, DT::eStorageBuffer), binding(1, DT::eStorageBuffer)};
    const std::array trace_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eSampledImage),
                                       binding(2, DT::eSampledImage), binding(3, DT::eSampledImage),
                                       binding(4, DT::eStorageImage),
                                       binding(5, DT::eAccelerationStructureKHR)};
    const std::array apply_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eSampledImage),
                                       binding(2, DT::eStorageImage)};
    make_layouts(expand_bindings, sizeof(ExpandConstants), expand_desc_layout, expand_layout);
    make_layouts(trace_bindings, sizeof(TraceConstants), trace_desc_layout, trace_layout);
    const std::array blur_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eStorageImage)};
    make_layouts(apply_bindings, sizeof(ApplyConstants), apply_desc_layout, apply_layout);
    make_layouts(blur_bindings, sizeof(BlurConstants), blur_desc_layout, blur_layout);
    blur_pipeline = make_pipeline(RTAO_BLUR_COMP, *blur_layout);
    expand_pipeline = make_pipeline(RTAO_EXPAND_COMP, *expand_layout);
    trace_pipeline = make_pipeline(RTAO_TRACE_COMP, *trace_layout);
    apply_pipeline = make_pipeline(RTAO_APPLY_COMP, *apply_layout);

    if (!CreateAccelerationStructures()) {
        std::puts("RTAO: acceleration structure setup failed, off");
        return;
    }
    enabled = true;
    std::printf("RTAO: on, %u rays, radius %.2f, strength %.2f, history %.2f, up to %u triangles%s\n",
                rays, radius, strength, history, max_triangles, debug ? ", debug view" : "");
    if (const char* file = std::getenv("BB_RTAO_FILE")) {
        settings_path = file;
    } else {
        settings_path = (Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "rtao.txt").string();
    }
    std::printf("RTAO: settings re-read from %s while the game runs\n", settings_path.c_str());
}

void RayTracedAO::PollSettings() {
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
        if (key == "strength") strength = std::clamp(value, 0.0f, 1.0f);
        else if (key == "radius") radius = std::max(value, 0.05f);
        else if (key == "history") history = std::clamp(value, 0.0f, 0.98f);
        else if (key == "rays") rays = std::clamp(u32(value), 1u, 32u);
        else if (key == "debug") debug = std::clamp(u32(value), 0u, 4u);
        else if (key == "enabled") active = value != 0.0f;
        else if (key == "stage") stage = std::clamp(u32(value), 0u, 5u);
        else if (key == "dump") dump_request = value;
        else if (key == "power") power = std::clamp(value, 0.1f, 8.0f);
        else if (key == "blur") blur = value != 0.0f;
        else if (key == "sigma") sigma = std::clamp(value, 0.0f, 4.0f);
        else if (key == "replace") replace_mask = value != 0.0f;
        else if (key == "snap") snap_far = std::max(value, 0.0f);
        else if (key == "build") build_fast_trace = value != 0.0f;
        else if (key == "far") ao_far = std::max(value, 0.0f);
        else if (key == "alpha") include_alpha = value != 0.0f;
        else if (key == "decals") include_nowrite = value != 0.0f;
    }
    std::printf("RTAO: settings: enabled %d, %u rays, radius %.2f, strength %.2f, power %.2f, history %.2f, "
                "sigma %.2f, blur %d, replace %d, snap %.0f m, far %.0f m, build %s, debug %u, stage %u, alpha %d, decals %d\n",
                active ? 1 : 0, rays, radius, strength, power, history, sigma, blur ? 1 : 0, replace_mask ? 1 : 0,
                snap_far, ao_far, build_fast_trace ? "fast trace" : "fast build",
                debug, stage, include_alpha ? 1 : 0, include_nowrite ? 1 : 0);
}

RayTracedAO::~RayTracedAO() {
    const auto device = instance.GetDevice();
    for (auto& entry : view_cache) {
        if (entry.view) {
            device.destroyImageView(entry.view);
        }
    }
    if (tlas) {
        device.destroyAccelerationStructureKHR(tlas);
    }
    if (blas) {
        device.destroyAccelerationStructureKHR(blas);
    }
    for (auto* buffer : {&table_buffer, &triangle_buffer, &blas_buffer, &blas_scratch, &tlas_buffer,
                         &tlas_scratch, &instance_buffer, &readback_buffer, &dump_buffer}) {
        DestroyBuffer(*buffer);
    }
}

bool RayTracedAO::CreateBuffer(Buffer& out, vk::DeviceSize size, vk::BufferUsageFlags usage,
                               bool host, vk::DeviceSize alignment) {
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VkBufferUsageFlags(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo alloc_ci{
        .flags = host ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT
                      : VmaAllocationCreateFlags{},
        .usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = host ? VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                              : VkMemoryPropertyFlags{},
    };
    VkBuffer raw{};
    VmaAllocationInfo info{};
    const VkResult result =
        alignment ? vmaCreateBufferWithAlignment(instance.GetAllocator(), &buffer_ci, &alloc_ci,
                                                 alignment, &raw, &out.allocation, &info)
                  : vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &raw,
                                    &out.allocation, &info);
    if (result != VK_SUCCESS) {
        std::printf("RTAO: buffer of %llu MiB failed (%d)\n", (unsigned long long)(size >> 20), result);
        return false;
    }
    out.buffer = raw;
    out.size = size;
    out.mapped = info.pMappedData;
    out.address = instance.GetDevice().getBufferAddress({.buffer = out.buffer});
    return true;
}

void RayTracedAO::DestroyBuffer(Buffer& buffer) {
    if (buffer.buffer) {
        vmaDestroyBuffer(instance.GetAllocator(), buffer.buffer, buffer.allocation);
        buffer = {};
    }
}

bool RayTracedAO::CreateAccelerationStructures() {
    const auto device = instance.GetDevice();
    const auto as_props = instance.GetPhysicalDevice()
                              .getProperties2<vk::PhysicalDeviceProperties2,
                                              vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
                              .get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
    scratch_alignment = std::max<vk::DeviceSize>(as_props.minAccelerationStructureScratchOffsetAlignment, 1);

    using BU = vk::BufferUsageFlagBits;
    if (!CreateBuffer(table_buffer, vk::DeviceSize(TableSlots) * max_geometries * sizeof(GpuGeometry),
                      BU::eStorageBuffer, true) ||
        !CreateBuffer(triangle_buffer, vk::DeviceSize(max_triangles) * 9 * sizeof(float),
                      BU::eStorageBuffer | BU::eAccelerationStructureBuildInputReadOnlyKHR, false) ||
        !CreateBuffer(instance_buffer, sizeof(vk::AccelerationStructureInstanceKHR),
                      BU::eAccelerationStructureBuildInputReadOnlyKHR, true) ||
        !CreateBuffer(readback_buffer, vk::DeviceSize(ReadbackTriangles) * 9 * sizeof(float),
                      BU::eTransferDst, true)) {
        return false;
    }

    // BLAS: one triangle geometry over the whole list, sized for the budget.
    vk::AccelerationStructureGeometryKHR blas_geometry{
        .geometryType = vk::GeometryTypeKHR::eTriangles,
        .geometry = {.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                         .vertexFormat = vk::Format::eR32G32B32Sfloat,
                         .vertexData = {.deviceAddress = triangle_buffer.address},
                         .vertexStride = 3 * sizeof(float),
                         .maxVertex = max_triangles * 3 - 1,
                         .indexType = vk::IndexType::eNoneKHR,
                     }},
        .flags = vk::GeometryFlagBitsKHR::eOpaque,
    };
    vk::AccelerationStructureBuildGeometryInfoKHR blas_info{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
                 vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &blas_geometry,
    };
    // Sized for both build preferences ("build=" in the settings switches them live).
    auto blas_sizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, blas_info, max_triangles);
    blas_info.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
                      vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess;
    const auto trace_sizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, blas_info, max_triangles);
    blas_sizes.accelerationStructureSize =
        std::max(blas_sizes.accelerationStructureSize, trace_sizes.accelerationStructureSize);
    blas_sizes.buildScratchSize = std::max(blas_sizes.buildScratchSize, trace_sizes.buildScratchSize);
    if (!CreateBuffer(blas_buffer, blas_sizes.accelerationStructureSize,
                      BU::eAccelerationStructureStorageKHR, false) ||
        !CreateBuffer(blas_scratch, blas_sizes.buildScratchSize, BU::eStorageBuffer, false,
                      scratch_alignment)) {
        return false;
    }
    blas = Check(device.createAccelerationStructureKHR({
        .buffer = blas_buffer.buffer,
        .size = blas_sizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
    }));

    // TLAS: one instance, identity (the triangles are already in view space).
    vk::AccelerationStructureGeometryKHR tlas_geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = {.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                         .arrayOfPointers = false,
                         .data = {.deviceAddress = instance_buffer.address},
                     }},
    };
    vk::AccelerationStructureBuildGeometryInfoKHR tlas_info{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &tlas_geometry,
    };
    const auto tlas_sizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, tlas_info, 1u);
    if (!CreateBuffer(tlas_buffer, tlas_sizes.accelerationStructureSize,
                      BU::eAccelerationStructureStorageKHR, false) ||
        !CreateBuffer(tlas_scratch, tlas_sizes.buildScratchSize, BU::eStorageBuffer, false,
                      scratch_alignment)) {
        return false;
    }
    tlas = Check(device.createAccelerationStructureKHR({
        .buffer = tlas_buffer.buffer,
        .size = tlas_sizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
    }));
    // The instance never changes: identity transform, the BLAS.
    vk::AccelerationStructureInstanceKHR as_instance{
        .instanceCustomIndex = 0,
        .mask = 0xFF,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = VkGeometryInstanceFlagsKHR(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable),
        .accelerationStructureReference =
            device.getAccelerationStructureAddressKHR({.accelerationStructure = blas}),
    };
    as_instance.transform.matrix[0][0] = 1.0f;
    as_instance.transform.matrix[1][1] = 1.0f;
    as_instance.transform.matrix[2][2] = 1.0f;
    std::memcpy(instance_buffer.mapped, &as_instance, sizeof(as_instance));
    vmaFlushAllocation(instance.GetAllocator(), instance_buffer.allocation, 0, VK_WHOLE_SIZE);
    std::printf("RTAO: acceleration structures: BLAS %llu MiB (+%llu scratch), triangles %llu MiB\n",
                (unsigned long long)(blas_sizes.accelerationStructureSize >> 20),
                (unsigned long long)(blas_sizes.buildScratchSize >> 20),
                (unsigned long long)(triangle_buffer.size >> 20));
    return true;
}

void RayTracedAO::OnFrameStart() {
    if (!enabled) {
        return;
    }
    static u32 reports = 0;
    if (dump_pending && frame >= dump_frame + 120 && dump_buffer.mapped) {
        dump_pending = false;
        vmaInvalidateAllocation(instance.GetAllocator(), dump_buffer.allocation, 0, VK_WHOLE_SIZE);
        const u32 dw = (width + 1) / 2, dh = (height + 1) / 2;
        const auto* texels = static_cast<const u16*>(dump_buffer.mapped);
        const auto half_to_float = [](u16 hf) {
            const u32 sign = (hf >> 15) & 1, exp = (hf >> 10) & 0x1f, mant = hf & 0x3ff;
            float value;
            if (exp == 0) value = std::ldexp(float(mant), -24);
            else if (exp == 31) value = mant ? NAN : INFINITY;
            else value = std::ldexp(float(mant | 0x400), int(exp) - 25);
            return sign ? -value : value;
        };
        const std::string path = "out/rtao_dump_" + std::to_string(dump_frame) + ".pgm";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fprintf(f, "P5\n%u %u\n255\n", dw, dh);
            std::vector<u8> row(dw);
            double sum = 0.0;
            u32 zeros = 0, ones = 0;
            for (u32 y = 0; y < dh; ++y) {
                for (u32 x = 0; x < dw; ++x) {
                    const float r = half_to_float(texels[(size_t(y) * dw + x) * 2]);
                    const float c = std::isfinite(r) ? std::clamp(r, 0.0f, 2.0f) : 0.0f;
                    row[x] = u8(c * 127.5f + 0.5f); // 1.0 -> 128
                    sum += c;
                    zeros += c == 0.0f;
                    ones += std::abs(c - 1.0f) < 0.1f;
                }
                std::fwrite(row.data(), 1, dw, f);
            }
            std::fclose(f);
            std::printf("RTAO: dumped %s (%ux%u, 1.0 = 128): mean %.3f, %.1f%% zero, %.1f%% within 0.1 of 1\n",
                        path.c_str(), dw, dh, sum / (double(dw) * dh), 100.0 * zeros / (double(dw) * dh),
                        100.0 * ones / (double(dw) * dh));
        }
    }
    if (readback_pending && frame >= readback_frame + 120) {
        // Copied 120 frames ago: long finished. How do the expanded triangles look?
        readback_pending = false;
        vmaInvalidateAllocation(instance.GetAllocator(), readback_buffer.allocation, 0, VK_WHOLE_SIZE);
        const float* v = static_cast<const float*>(readback_buffer.mapped);
        u32 valid = 0, finite = 0;
        const u32 count = std::min<u32>(ReadbackTriangles, readback_triangles);
        float zmin = 1e30f, zmax = -1e30f, xymax = 0.0f;
        for (u32 i = 0; i < count; ++i) {
            const float* tri = v + i * 9;
            bool nonzero = false, all_finite = true;
            for (u32 k = 0; k < 9; ++k) {
                nonzero |= tri[k] != 0.0f;
                all_finite &= std::isfinite(tri[k]);
            }
            if (!nonzero) continue;
            ++valid;
            if (!all_finite) continue;
            ++finite;
            for (u32 k = 0; k < 3; ++k) {
                zmin = std::min(zmin, tri[k * 3 + 2]);
                zmax = std::max(zmax, tri[k * 3 + 2]);
                xymax = std::max({xymax, std::abs(tri[k * 3]), std::abs(tri[k * 3 + 1])});
            }
        }
        std::printf("RTAO: readback (frame %u): of the first %u triangles %u non-zero, %u valid (finite), "
                    "%u invalid (NaN); z %.2f..%.2f, |xy| up to %.2f; first: (%.2f %.2f %.2f) (%.2f %.2f %.2f) "
                    "(%.2f %.2f %.2f)\n",
                    readback_frame, count, valid, finite, valid - finite, zmin, zmax, xymax, v[0], v[1], v[2],
                    v[3], v[4], v[5], v[6], v[7], v[8]);
    }
    if (frame % 600 == 0 && frame) {
        std::printf("RTAO: frame %u: %zu draws, %u triangles (%u draws, %u triangles dropped: budget %u; "
                    "%u strips/other, %u alpha-tested, %u without depth write, %u without index address "
                    "skipped), %u passes recorded in 600 frames\n",
                    frame, geometries.size(), frame_triangles, dropped_draws, dropped_triangles,
                    max_triangles, skipped_topology, skipped_alpha, skipped_nowrite, skipped_no_address,
                    recorded);
        if (recorded == 0 && !geometries.empty() && active) {
            std::puts("RTAO: geometry captured but no pass recorded: the game's AO pass (its R8 mask) was not "
                      "seen. Is the \"Disable SSAO\" game patch on? RTAO writes into that pass's mask.");
        }
        recorded = 0;
        ++reports;
    }
    geometries.clear();
    frame_triangles = 0;
    dropped_triangles = 0;
    dropped_draws = 0;
    skipped_topology = 0;
    skipped_no_address = 0;
    skipped_alpha = 0;
    skipped_nowrite = 0;
    done_this_frame = false;
    last_color = {};
    mask_candidate = {};
    ++frame;
    PollSettings();
}

void RayTracedAO::OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal) {
    depth_id = depth;
    normal_id = normal;
}

void RayTracedAO::AddGeometry(const Geometry& g) {
    if (!enabled || g.index_count < 3 || !g.vertices || !g.instances) {
        return;
    }
    if (!g.triangle_list) { // strips and the rest: not expanded (yet), counted
        ++skipped_topology;
        return;
    }
    if (g.alpha_test && !include_alpha) {
        ++skipped_alpha;
        return;
    }
    if (!g.depth_write && !include_nowrite) {
        ++skipped_nowrite;
        return;
    }
    if (g.index_address == 0 && !g.non_indexed) {
        ++skipped_no_address; // an index buffer without a device address
        return;
    }
    const u32 tris = (g.index_count / 3) * g.instances;
    if (g.non_indexed && g.index_count > g.vertices) {
        ++dropped_draws; // the capture holds fewer vertices than the draw
        dropped_triangles += tris;
        return;
    }
    // The positions must lie inside the capture (two halves of the object-motion buffer).
    const u64 last = u64(g.store) + u64(g.vertices) * g.instances;
    if (last > 1 + 2ull * object_motion.PositionsCapacity()) {
        ++dropped_draws;
        dropped_triangles += tris;
        return;
    }
    if (geometries.size() >= max_geometries || frame_triangles + tris > max_triangles) {
        ++dropped_draws;
        dropped_triangles += tris;
        return;
    }
    geometries.push_back({.store = g.store,
                          .vertices = g.vertices,
                          .instances = g.instances,
                          .first_vertex = g.first_vertex,
                          // 16-bit index buffers may start 2 bytes into a word: read words
                          // from the aligned address and skip the first half-word.
                          .index_address = g.index16 ? (g.index_address & ~u64(3)) : g.index_address,
                          .index_count = g.index_count,
                          .index16 = g.index16 ? (1u | u32((g.index_address & 2) ? 2u : 0u)) : 0u,
                          .base_vertex = g.base_vertex,
                          .restart = g.restart ? 1u : 0u,
                          .triangle_offset = frame_triangles,
                          .triangles = tris});
    frame_triangles += tris;
}

void RayTracedAO::OnColorTarget(VideoCore::ImageId color) {
    if (!enabled || color == last_color) {
        return;
    }
    // The previous pass ended. Was it the game's AO upsample (into the R8 mask)?
    if (last_color && last_color == mask_candidate && !done_this_frame && depth_id && active &&
        !geometries.empty()) {
        done_this_frame = true;
        Record(mask_candidate);
    }
    last_color = color;
    mask_candidate = {};
    if (!color || !depth_id) {
        return;
    }
    const auto& image = texture_cache.GetImage(color);
    const auto& depth = texture_cache.GetImage(depth_id);
    if (image.info.pixel_format == vk::Format::eR8Unorm &&
        image.info.size.width == depth.info.size.width &&
        image.info.size.height == depth.info.size.height) {
        mask_candidate = color;
    }
}

bool RayTracedAO::EnsureResources(u32 w, u32 h) {
    if (width == w && height == h && ao_images[0]) {
        return true;
    }
    const auto device = instance.GetDevice();
    const auto allocator = instance.GetAllocator();
    const auto make_image = [&](VideoCore::UniqueImage& image, vk::UniqueImageView& view,
                                vk::Format format, u32 iw, u32 ih) {
        view.reset();
        image = VideoCore::UniqueImage(device, allocator);
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
    scheduler.Finish();
    for (u32 i = 0; i < 2; ++i) {
        make_image(ao_images[i], ao_views[i], vk::Format::eR16G16Sfloat, (w + 1) / 2, (h + 1) / 2);
    }
    make_image(motion_image, motion_view, vk::Format::eR16G16Sfloat, w, h);
    make_image(blur_image, blur_view, vk::Format::eR16G16Sfloat, (w + 1) / 2, (h + 1) / 2);
    if (dump_buffer.buffer) {
        vmaDestroyBuffer(allocator, dump_buffer.buffer, dump_buffer.allocation);
        dump_buffer = {};
    }
    CreateBuffer(dump_buffer, vk::DeviceSize((w + 1) / 2) * ((h + 1) / 2) * 4, vk::BufferUsageFlagBits::eTransferDst,
                 true);
    dump_pending = false;
    width = w;
    height = h;
    history_valid = false;
    ao_index = 0;
    return true;
}

vk::ImageView RayTracedAO::CachedView(const VideoCore::Image& image, vk::Format format,
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

void RayTracedAO::Record(VideoCore::ImageId mask_id) {
    auto& mask = texture_cache.GetImage(mask_id);
    auto& depth = texture_cache.GetImage(depth_id);
    auto& normal = texture_cache.GetImage(normal_id);
    const u32 w = depth.info.size.width, h = depth.info.size.height;
    if (normal.info.size.width != w || normal.info.size.height != h || !EnsureResources(w, h)) {
        return;
    }
    // The geometry table of this frame (a host ring slot; the GPU reads it this submission).
    const u32 slot = frame % TableSlots;
    auto* table = static_cast<GpuGeometry*>(table_buffer.mapped) + size_t(slot) * max_geometries;
    std::memcpy(table, geometries.data(), geometries.size() * sizeof(GpuGeometry));
    vmaFlushAllocation(instance.GetAllocator(), table_buffer.allocation,
                       vk::DeviceSize(slot) * max_geometries * sizeof(GpuGeometry),
                       std::max<vk::DeviceSize>(geometries.size(), 1) * sizeof(GpuGeometry));
    const u32 num_geometries = u32(geometries.size());
    const u32 total_triangles = frame_triangles;

    const auto depth_view = CachedView(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const auto normal_view = CachedView(normal, normal.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const auto mask_view = CachedView(mask, vk::Format::eR8Unorm, vk::ImageAspectFlagBits::eColor);

    scheduler.EndRendering();
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&normal, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&mask, vk::ImageLayout::eGeneral, Compute, rw);
    runtime.FlushBarriers();

    const u32 read = ao_index, write = ao_index ^ 1;
    ao_index = write;
    const auto image_barrier = [&](vk::Image image, vk::ImageLayout old_layout,
                                   vk::AccessFlags2 src_access, vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 b{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = src_access,
            .dstStageMask = Compute,
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
    image_barrier(vk::Image(motion_image), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
                  vk::AccessFlagBits2::eShaderWrite);
    image_barrier(vk::Image(ao_images[write]), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
                  vk::AccessFlagBits2::eShaderWrite);
    image_barrier(vk::Image(blur_image), vk::ImageLayout::eUndefined, vk::AccessFlagBits2::eNone,
                  vk::AccessFlagBits2::eShaderWrite);
    image_barrier(vk::Image(ao_images[read]),
                  history_valid ? vk::ImageLayout::eGeneral : vk::ImageLayout::eUndefined, rw,
                  vk::AccessFlagBits2::eShaderRead);
    camera_motion.RecordMotion(depth_view, *motion_view, w, h);
    image_barrier(vk::Image(motion_image), vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
                  vk::AccessFlagBits2::eShaderRead);

    const auto cam_proj = camera_motion.Projection();
    if (cam_proj[0] == 0.0f || cam_proj[1] == 0.0f || cam_proj[3] == 0.0f) {
        return; // no camera this frame (menus): nothing sensible to build
    }
    ++recorded;
    ++recorded_total;
    const bool do_readback = !readback_pending && total_triangles > 0 &&
                             (recorded_total == 1 || frame % 600 == 300);
    if (do_readback) {
        readback_pending = true;
        readback_frame = frame;
        readback_triangles = total_triangles;
    }
    const bool do_dump = dump_request != dump_done && !dump_pending && stage >= 4 && dump_buffer.mapped;
    if (do_dump) {
        dump_pending = true;
        dump_frame = frame;
        dump_done = dump_request;
    }
    const vk::Buffer dump_handle = dump_buffer.buffer;
    if (recorded_total <= 2) {
        std::printf("RTAO: pass %u: proj %g %g %g %g, %u geometries, %u triangles, %ux%u\n",
                    recorded_total, cam_proj[0], cam_proj[1], cam_proj[2], cam_proj[3], num_geometries,
                    total_triangles, w, h);
        for (u32 i = 0; i < std::min<u32>(num_geometries, 4); ++i) {
            const auto& g = geometries[i];
            std::printf("RTAO:   geometry %u: store %u, %u vertices x %u instances, first %u, indices 0x%llx "
                        "count %u %s base %d restart %u, triangles %u at %u\n",
                        i, g.store, g.vertices, g.instances, g.first_vertex,
                        (unsigned long long)g.index_address, g.index_count,
                        (g.index16 & 1) ? "16-bit" : "32-bit", g.base_vertex, g.restart, g.triangles,
                        g.triangle_offset);
        }
    }
    const vk::Buffer readback_handle = readback_buffer.buffer;
    const ExpandConstants expand{
        .positions_address = instance.GetDevice().getBufferAddress({.buffer = object_motion.PositionsBuffer()}),
        .proj = camera_motion.Projection(),
        .num_geometries = num_geometries,
        .total_triangles = total_triangles,
        .far_cut = ao_far > 0.0f ? ao_far + radius : 0.0f,
    };
    const auto jitter = camera_motion.Jitter();
    const TraceConstants trace{
        .view = camera_motion.View(),
        .proj = expand.proj,
        .full_size = {float(w), float(h)},
        .half_size = {float((w + 1) / 2), float((h + 1) / 2)},
        .jitter = jitter,
        .radius = radius,
        .history = history_valid ? history : 0.0f,
        .rays = rays,
        .frame = frame,
        .mode = debug,
        .snap_far = snap_far,
        .ao_far = ao_far,
    };
    const ApplyConstants apply{
        .proj = expand.proj,
        .full_size = trace.full_size,
        .half_size = trace.half_size,
        .strength = strength,
        .debug = debug ? 1u : 0u,
        .power = power,
        .blur = blur ? 1u : 0u,
        .replace = replace_mask ? 1u : 0u,
    };
    history_valid = true;

    const vk::Buffer table_handle = table_buffer.buffer;
    const vk::DeviceSize table_offset = vk::DeviceSize(slot) * max_geometries * sizeof(GpuGeometry);
    const vk::DeviceSize table_bytes = vk::DeviceSize(std::max<u32>(num_geometries, 1)) * sizeof(GpuGeometry);
    const vk::Buffer triangles_handle = triangle_buffer.buffer;
    const vk::DeviceSize triangles_bytes = triangle_buffer.size;
    const vk::DeviceAddress triangles_address = triangle_buffer.address;
    const vk::DeviceAddress blas_scratch_address = blas_scratch.address;
    const vk::DeviceAddress tlas_scratch_address = tlas_scratch.address;
    const vk::DeviceAddress instances_address = instance_buffer.address;
    const vk::ImageView ao_read = *ao_views[read], ao_write = *ao_views[write];
    const vk::ImageView blur_rw = *blur_view;
    const BlurConstants blur_h{.half_size = {float((w + 1) / 2), float((h + 1) / 2)}, .dir = {1, 0}, .sigma = sigma};
    const BlurConstants blur_v{.half_size = blur_h.half_size, .dir = {0, 1}, .sigma = sigma};
    const vk::ImageView motion = *motion_view;

    // Five crumbs, so a GPU hang names the stage; "stage" in the settings stops after that stage.
    const u32 last_stage = stage;
    const bool fast_trace = build_fast_trace;
    const auto memory_barrier = [](vk::CommandBuffer cmdbuf, vk::PipelineStageFlags2 src_stage,
                                   vk::AccessFlags2 src, vk::PipelineStageFlags2 dst_stage,
                                   vk::AccessFlags2 dst) {
        const vk::MemoryBarrier2 b{.srcStageMask = src_stage,
                                   .srcAccessMask = src,
                                   .dstStageMask = dst_stage,
                                   .dstAccessMask = dst};
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
    };
    const auto info = [](vk::ImageView view) {
        return vk::DescriptorImageInfo{.imageView = view, .imageLayout = vk::ImageLayout::eGeneral};
    };
    // BB_GPU_PROFILE=1: one segment per stage.
    const auto profile = [](u64 key, const char* name) {
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(key, [name] { return std::string{name}; });
        }
    };

    // 1. Expand: the vertex shaders wrote the positions (vertex stage), the table was written by
    // the host, the index buffers by uploads: wait for all of it.
    if (last_stage < 1) {
        return;
    }
    profile(0x47A01ull, "rtao expand");
    scheduler.RecordCrumb(
        {.name = "rtao expand"},
        [=, expand_pipeline = *expand_pipeline, expand_layout = *expand_layout](vk::CommandBuffer cmdbuf) {
            memory_barrier(cmdbuf, vk::PipelineStageFlagBits2::eAllCommands,
                           vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eHostWrite, Compute,
                           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
            if (total_triangles) {
                const vk::DescriptorBufferInfo table_info{table_handle, table_offset, table_bytes};
                const vk::DescriptorBufferInfo tri_info{triangles_handle, 0, triangles_bytes};
                const std::array writes = {
                    vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &table_info},
                    vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &tri_info}};
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, expand_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, expand_layout, 0, writes);
                cmdbuf.pushConstants(expand_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(expand),
                                     &expand);
                cmdbuf.dispatch((total_triangles + 63) / 64, 1, 1);
            }
            memory_barrier(cmdbuf, Compute, vk::AccessFlagBits2::eShaderWrite,
                           AsBuild | vk::PipelineStageFlagBits2::eTransfer,
                           vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);
            if (do_readback) {
                const vk::BufferCopy region{0, 0, vk::DeviceSize(std::min(ReadbackTriangles, total_triangles)) * 9 * sizeof(float)};
                cmdbuf.copyBuffer(triangles_handle, readback_handle, region);
            }
        });

    // 2. BLAS over the triangle list.
    if (last_stage < 2) {
        return;
    }
    profile(0x47A02ull, "rtao blas");
    scheduler.RecordCrumb({.name = "rtao blas"}, [=, blas = blas](vk::CommandBuffer cmdbuf) {
        vk::AccelerationStructureGeometryKHR blas_geometry{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = {.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                             .vertexFormat = vk::Format::eR32G32B32Sfloat,
                             .vertexData = {.deviceAddress = triangles_address},
                             .vertexStride = 3 * sizeof(float),
                             .maxVertex = std::max<u32>(total_triangles * 3, 1) - 1,
                             .indexType = vk::IndexType::eNoneKHR,
                         }},
            .flags = vk::GeometryFlagBitsKHR::eOpaque,
        };
        const vk::AccelerationStructureBuildGeometryInfoKHR blas_info{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags = (fast_trace ? vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                                 : vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild) |
                     vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .dstAccelerationStructure = blas,
            .geometryCount = 1,
            .pGeometries = &blas_geometry,
            .scratchData = {.deviceAddress = blas_scratch_address},
        };
        const vk::AccelerationStructureBuildRangeInfoKHR blas_range{.primitiveCount = total_triangles};
        cmdbuf.buildAccelerationStructuresKHR(blas_info, &blas_range);
        memory_barrier(cmdbuf, AsBuild, vk::AccessFlagBits2::eAccelerationStructureWriteKHR, AsBuild,
                       vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    });

    // 3. TLAS: the one instance.
    if (last_stage < 3) {
        return;
    }
    profile(0x47A03ull, "rtao tlas");
    scheduler.RecordCrumb({.name = "rtao tlas"}, [=, tlas = tlas](vk::CommandBuffer cmdbuf) {
        vk::AccelerationStructureGeometryKHR tlas_geometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = {.instances = vk::AccelerationStructureGeometryInstancesDataKHR{
                             .arrayOfPointers = false,
                             .data = {.deviceAddress = instances_address},
                         }},
        };
        const vk::AccelerationStructureBuildGeometryInfoKHR tlas_info{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .dstAccelerationStructure = tlas,
            .geometryCount = 1,
            .pGeometries = &tlas_geometry,
            .scratchData = {.deviceAddress = tlas_scratch_address},
        };
        const vk::AccelerationStructureBuildRangeInfoKHR tlas_range{.primitiveCount = 1};
        cmdbuf.buildAccelerationStructuresKHR(tlas_info, &tlas_range);
        memory_barrier(cmdbuf, AsBuild, vk::AccessFlagBits2::eAccelerationStructureWriteKHR, Compute,
                       vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    });

    // 4. Trace at half resolution.
    if (last_stage < 4) {
        return;
    }
    profile(0x47A04ull, "rtao trace");
    scheduler.RecordCrumb(
        {.name = "rtao trace"},
        [=, tlas = tlas, trace_pipeline = *trace_pipeline, trace_layout = *trace_layout,
         ao_image = vk::Image(ao_images[write])](vk::CommandBuffer cmdbuf) {
            const auto d = info(depth_view), n = info(normal_view), m = info(motion), p = info(ao_read),
                       o = info(ao_write);
            const vk::WriteDescriptorSetAccelerationStructureKHR as_write{
                .accelerationStructureCount = 1, .pAccelerationStructures = &tlas};
            const std::array writes = {
                vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &n},
                vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &m},
                vk::WriteDescriptorSet{.dstBinding = 3, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &p},
                vk::WriteDescriptorSet{.dstBinding = 4, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &o},
                vk::WriteDescriptorSet{.pNext = &as_write, .dstBinding = 5, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eAccelerationStructureKHR}};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, trace_pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, trace_layout, 0, writes);
            cmdbuf.pushConstants(trace_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(trace), &trace);
            cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 7) / 8, 1);
            const vk::ImageMemoryBarrier2 ao_barrier{
                .srcStageMask = Compute,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = Compute | vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = ao_image,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ao_barrier});
            if (do_dump) {
                const vk::BufferImageCopy region{
                    .bufferOffset = 0,
                    .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                    .imageExtent = {(w + 1) / 2, (h + 1) / 2, 1},
                };
                cmdbuf.copyImageToBuffer(ao_image, vk::ImageLayout::eGeneral, dump_handle, region);
            }
        });

    // 4b. Separable depth-aware blur of the AO image (in place through blur_image).
    if (sigma > 0.0f) {
        profile(0x47A06ull, "rtao blur");
        scheduler.RecordCrumb(
            {.name = "rtao blur"},
            [=, blur_pipeline = *blur_pipeline, blur_layout = *blur_layout, ao_image = vk::Image(ao_images[write]),
             blur_img = vk::Image(blur_image)](vk::CommandBuffer cmdbuf) {
                const auto barrier = [&](vk::Image image) {
                    const vk::ImageMemoryBarrier2 b{
                        .srcStageMask = Compute,
                        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                        .dstStageMask = Compute,
                        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                        .oldLayout = vk::ImageLayout::eGeneral,
                        .newLayout = vk::ImageLayout::eGeneral,
                        .image = image,
                        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                    };
                    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
                };
                const auto pass = [&](vk::ImageView src, vk::ImageView dst, const BlurConstants& constants) {
                    const auto si = info(src), di = info(dst);
                    const std::array writes = {
                        vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &si},
                        vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                               .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &di}};
                    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, blur_pipeline);
                    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, blur_layout, 0, writes);
                    cmdbuf.pushConstants(blur_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants),
                                         &constants);
                    cmdbuf.dispatch(((w + 1) / 2 + 7) / 8, ((h + 1) / 2 + 7) / 8, 1);
                };
                pass(ao_write, blur_rw, blur_h);
                barrier(blur_img);
                pass(blur_rw, ao_write, blur_v);
                barrier(ao_image);
            });
    }

    // 5. Into the game's mask.
    if (last_stage < 5) {
        return;
    }
    profile(0x47A05ull, "rtao apply");
    scheduler.RecordCrumb(
        {.name = "rtao apply"},
        [=, apply_pipeline = *apply_pipeline, apply_layout = *apply_layout](vk::CommandBuffer cmdbuf) {
            const auto a = info(ao_write), d = info(depth_view), mk = info(mask_view);
            const std::array writes = {
                vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &a},
                vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &d},
                vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &mk}};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, apply_pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, apply_layout, 0, writes);
            cmdbuf.pushConstants(apply_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(apply), &apply);
            cmdbuf.dispatch((w + 7) / 8, (h + 7) / 8, 1);
        });
}

} // namespace Vulkan
