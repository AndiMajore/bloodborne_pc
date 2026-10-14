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
#include "video_core/host_shaders/rtao_probe_comp.h"
#include "video_core/host_shaders/rtao_trace_comp.h"
#include "video_core/renderer_vulkan/vk_camera_motion.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_object_motion.h"
#include "video_core/renderer_vulkan/vk_platform.h"
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
    std::array<float, 12> inv_view; // view -> world rows (world-space entries)
};
struct ProbeConstants {
    u64 positions_address;
    u32 num_geometries;
    u32 id;
    std::array<float, 4> proj;
    std::array<float, 12> inv_view;
};
static_assert(sizeof(ProbeConstants) == 80 && offsetof(ProbeConstants, proj) == 16);
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
    u32 normals;
    float offset;
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
static_assert(sizeof(ExpandConstants) == 96 && offsetof(ExpandConstants, inv_view) == 48);
static_assert(sizeof(TraceConstants) == 124 && offsetof(TraceConstants, offset) == 120);
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
    cache_enabled = EnvU32("BB_RTAO_CACHE", 1) != 0;
    // ALLOW_DATA_ACCESS on the per-frame BLAS although nothing reads positions back: on RADV the
    // flag turns RDNA4's compressed triangle-pair leaves off, and their build cost more than
    // their traversal saved (Doll, three sessions: build 1.65 vs 2.41 ms, trace 1.79 vs 1.73 ms).
    // Cached BLASes are built once and traced for many frames: compressed (no flag).
    data_access = EnvU32("BB_RTAO_DATA_ACCESS", 1) != 0;

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
            .pushConstantRangeCount = push_size ? 1u : 0u,
            .pPushConstantRanges = push_size ? &range : nullptr,
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
                                       binding(5, DT::eAccelerationStructureKHR),
                                       binding(6, DT::eUniformBuffer)};
    const std::array apply_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eSampledImage),
                                       binding(2, DT::eStorageImage)};
    make_layouts(expand_bindings, sizeof(ExpandConstants), expand_desc_layout, expand_layout);
    make_layouts(trace_bindings, 0, trace_desc_layout, trace_layout); // constants: a uniform buffer
    const std::array blur_bindings = {binding(0, DT::eSampledImage), binding(1, DT::eStorageImage)};
    make_layouts(apply_bindings, sizeof(ApplyConstants), apply_desc_layout, apply_layout);
    make_layouts(blur_bindings, sizeof(BlurConstants), blur_desc_layout, blur_layout);
    blur_pipeline = make_pipeline(RTAO_BLUR_COMP, *blur_layout);
    const std::array probe_bindings = {binding(0, DT::eStorageBuffer), binding(1, DT::eStorageBuffer)};
    make_layouts(probe_bindings, sizeof(ProbeConstants), probe_desc_layout, probe_layout);
    probe_pipeline = make_pipeline(RTAO_PROBE_COMP, *probe_layout);
    expand_pipeline = make_pipeline(RTAO_EXPAND_COMP, *expand_layout);
    trace_pipeline = make_pipeline(RTAO_TRACE_COMP, *trace_layout);
    apply_pipeline = make_pipeline(RTAO_APPLY_COMP, *apply_layout);
    // Names for validation messages and captures.
    SetObjectName(device, *expand_pipeline, "rtao expand");
    SetObjectName(device, *probe_pipeline, "rtao probe");
    SetObjectName(device, *trace_pipeline, "rtao trace");
    SetObjectName(device, *blur_pipeline, "rtao blur");
    SetObjectName(device, *apply_pipeline, "rtao apply");

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
        else if (key == "cache") cache_enabled = value != 0.0f;
        else if (key == "promote") promote_after = std::clamp(u32(value), 2u, 600u);
        else if (key == "maxsize") cache_max_size = std::max(value, 0.0f);
        else if (key == "ringtest") ring_test = value != 0.0f;
        else if (key == "normals") depth_normals = value != 0.0f;
        else if (key == "offset") origin_offset = std::clamp(value, 0.0f, 0.1f);
        else if (key == "alpha") include_alpha = value != 0.0f;
        else if (key == "decals") include_nowrite = value != 0.0f;
    }
    std::printf("RTAO: settings: enabled %d, %u rays, radius %.2f, strength %.2f, power %.2f, history %.2f, "
                "sigma %.2f, blur %d, replace %d, snap %.0f m, far %.0f m, build %s, debug %u, stage %u, alpha %d, decals %d, "
                "cache %d (after %u frames)\n",
                active ? 1 : 0, rays, radius, strength, power, history, sigma, blur ? 1 : 0, replace_mask ? 1 : 0,
                snap_far, ao_far, build_fast_trace ? "fast trace" : "fast build",
                debug, stage, include_alpha ? 1 : 0, include_nowrite ? 1 : 0, cache_enabled ? 1 : 0,
                promote_after);
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
    for (auto& [key, entry] : cache) {
        if (entry.blas) {
            device.destroyAccelerationStructureKHR(entry.blas);
            vmaDestroyBuffer(instance.GetAllocator(), entry.buffer, entry.allocation);
        }
    }
    for (auto* buffer : {&table_buffer, &triangle_buffer, &blas_buffer, &blas_scratch, &tlas_buffer,
                         &tlas_scratch, &instance_buffer, &readback_buffer, &dump_buffer, &probe_buffer,
                         &promo_scratch, &constants_buffer}) {
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
        !CreateBuffer(triangle_buffer, vk::DeviceSize(max_triangles + PromoTriangles) * 9 * sizeof(float),
                      BU::eStorageBuffer | BU::eAccelerationStructureBuildInputReadOnlyKHR |
                          BU::eTransferSrc, // the debug readback copies from it
                      false) ||
        !CreateBuffer(instance_buffer,
                      vk::DeviceSize(TableSlots) * MaxInstances * sizeof(vk::AccelerationStructureInstanceKHR),
                      BU::eAccelerationStructureBuildInputReadOnlyKHR, true) ||
        !CreateBuffer(probe_buffer, vk::DeviceSize(ProbeSlots) * max_geometries * sizeof(ProbeStats),
                      BU::eStorageBuffer, true) ||
        !CreateBuffer(constants_buffer, vk::DeviceSize(TableSlots) * ConstantsStride, BU::eUniformBuffer, true) ||
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
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild | (data_access ? vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess : vk::BuildAccelerationStructureFlagsKHR{}),
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &blas_geometry,
    };
    // Sized for both build preferences ("build=" in the settings switches them live).
    auto blas_sizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, blas_info, max_triangles);
    blas_info.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | (data_access ? vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess : vk::BuildAccelerationStructureFlagsKHR{});
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
    blas_address = device.getAccelerationStructureAddressKHR({.accelerationStructure = blas});
    // The static cache's builds of a frame share one scratch (sized for PromoTriangles, built for
    // trace speed: a cached BLAS is built once and traced for many frames).
    {
        blas_info.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace; // cache: compressed
        blas_geometry.geometry.triangles.maxVertex = PromoTriangles * 3 - 1;
        const auto promo_sizes = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, blas_info, PromoTriangles);
        // Room for the alignment of many small builds.
        if (!CreateBuffer(promo_scratch, promo_sizes.buildScratchSize + 512 * scratch_alignment,
                          BU::eStorageBuffer, false, scratch_alignment)) {
            return false;
        }
    }

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
        vk::AccelerationStructureBuildTypeKHR::eDevice, tlas_info, MaxInstances);
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
    for (const auto& [buffer, name] : std::initializer_list<std::pair<const Buffer*, const char*>>{
             {&table_buffer, "rtao table"}, {&triangle_buffer, "rtao triangles"},
             {&instance_buffer, "rtao instances"}, {&probe_buffer, "rtao probe results"},
             {&constants_buffer, "rtao constants"}, {&blas_buffer, "rtao per-frame BLAS"},
             {&blas_scratch, "rtao BLAS scratch"}, {&promo_scratch, "rtao cache scratch"},
             {&tlas_buffer, "rtao TLAS"}, {&tlas_scratch, "rtao TLAS scratch"},
             {&readback_buffer, "rtao readback"}}) {
        SetObjectName(device, buffer->buffer, name);
    }
    SetObjectName(device, blas, "rtao per-frame BLAS");
    SetObjectName(device, tlas, "rtao TLAS");
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
    if (dump_pending && dump_buffer.mapped && scheduler.IsFree(dump_tick)) {
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
    if (readback_pending && scheduler.IsFree(readback_tick)) {
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
                    "skipped), %u passes recorded in 600 frames, %u skipped (ring slot still in use)\n",
                    frame, geometries.size(), frame_triangles, dropped_draws, dropped_triangles,
                    max_triangles, skipped_topology, skipped_alpha, skipped_nowrite, skipped_no_address,
                    recorded, skipped_busy);
        if (history_resets) {
            std::printf("RTAO: history reset %u times (gaps or camera jumps) in 600 frames\n", history_resets);
            history_resets = 0;
        }
        skipped_busy = 0;
        std::printf("RTAO: cache: %zu draws known, %u instanced (%u triangles), %llu MiB of BLASes; "
                    "%u built, %u evicted in 600 frames; probes %u read, %u not written, %u same, %u reordered, "
                    "%u changed; camera (%.1f %.1f %.1f)\n",
                    cache.size(), cache_instances, cache_instance_tris,
                    (unsigned long long)(cache_bytes >> 20), promotions, evictions_total, probes_read,
                    probes_stale, probes_matched, probes_reordered, probes_changed, camera_world[0],
                    camera_world[1], camera_world[2]);
        probes_read = probes_stale = probes_matched = probes_changed = probes_reordered = 0;
        promotions = 0;
        evictions_total = 0;
        if (recorded == 0 && !draws.empty() && active && !missing_pass_reported) {
            missing_pass_reported = true;
            std::puts("RTAO: OFF in practice: the game's AO pass never ran, so there is no AO mask to "
                      "write into. The \"Disable SSAO\" game patch (launcher: Game effects, SSAO "
                      "unchecked) removes it; RTAO needs that pass. (Reported once.)");
        }
        recorded = 0;
        ++reports;
    }
    TrimCache();
    draws.clear();
    occurrences.clear();
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
    if (draws.size() >= max_geometries) {
        ++dropped_draws;
        dropped_triangles += tris;
        return;
    }
    // The same buffers may be drawn more than once a frame: the occurrence keeps them apart.
    const u32 occurrence = occurrences[g.key]++;
    u32& copies = max_copies[g.key];
    copies = std::max(copies, occurrence + 1);
    Draw& draw = draws.emplace_back();
    draw.key = KeyOf(g.key, occurrence);
    draw.base = g.key;
    draw.occurrence = occurrence;
    draw.skinned = g.skinned;
    draw.gpu = GpuGeometry{.store = g.store,
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
                          .triangle_offset = 0,
                          .triangles = tris,
                          .out_base = 0,
                          .flags = 0};
}

bool RayTracedAO::StatsMatch(const ProbeStats& a, const ProbeStats& b) const {
    if (a.count != b.count || a.count == 0) {
        return false;
    }
    // World positions are rebuilt from clip space each frame: allow for the rounding, which grows
    // with the distance from the camera.
    const float dx = a.centroid[0] - camera_world[0], dy = a.centroid[1] - camera_world[1],
                dz = a.centroid[2] - camera_world[2];
    const float tol = 0.02f + 2e-4f * std::sqrt(dx * dx + dy * dy + dz * dz);
    for (int i = 0; i < 3; ++i) {
        if (std::abs(a.centroid[i] - b.centroid[i]) > tol || std::abs(a.min[i] - b.min[i]) > tol ||
            std::abs(a.max[i] - b.max[i]) > tol || std::abs(a.first[i] - b.first[i]) > tol) {
            return false;
        }
    }
    return true;
}

void RayTracedAO::Evict(CacheEntry& entry) {
    if (entry.blas) {
        const auto device = instance.GetDevice();
        scheduler.DeferOperation([device, allocator = instance.GetAllocator(), blas = entry.blas,
                                  buffer = entry.buffer, allocation = entry.allocation] {
            device.destroyAccelerationStructureKHR(blas);
            vmaDestroyBuffer(allocator, buffer, allocation);
        });
        cache_bytes -= entry.bytes;
    }
    entry.blas = vk::AccelerationStructureKHR{};
    entry.buffer = vk::Buffer{};
    entry.allocation = VmaAllocation{};
    entry.address = 0;
    entry.bytes = 0;
    entry.triangles = 0;
}

void RayTracedAO::ReadProbes(u32 ring) {
    const u32 id = probe_ids[ring];
    if (!id) {
        return;
    }
    probe_ids[ring] = 0;
    vmaInvalidateAllocation(instance.GetAllocator(), probe_buffer.allocation,
                            vk::DeviceSize(ring) * max_geometries * sizeof(ProbeStats),
                            vk::DeviceSize(max_geometries) * sizeof(ProbeStats));
    const auto* stats = static_cast<const ProbeStats*>(probe_buffer.mapped) + size_t(ring) * max_geometries;
    const auto& keys = probe_keys[ring];
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!keys[i]) {
            continue;
        }
        if (stats[i].id != id) {
            ++probes_stale; // the GPU has not written it (yet)
            continue;
        }
        ++probes_read;
        ProbeStats s;
        std::memcpy(&s, &stats[i], sizeof(s));
        if (s.id != id) {
            continue;
        }
        const auto it = cache.find(keys[i]);
        if (it == cache.end()) {
            continue;
        }
        CacheEntry& e = it->second;
        if (e.have_stats && StatsMatch(e.stats, s)) {
            ++e.stable; // the reference stays the first observation: slow drift is not static
            ++probes_matched;
            continue;
        }
        if (e.have_stats) {
            // Another copy of the same mesh in this slot (the draw order changed): not a change.
            bool sibling = false;
            const auto copies = max_copies.find(e.base);
            for (u32 o = 0; copies != max_copies.end() && o < copies->second && !sibling; ++o) {
                const auto other = cache.find(KeyOf(e.base, o));
                sibling = other != cache.end() && &other->second != &e && other->second.have_stats &&
                          StatsMatch(other->second.stats, s);
            }
            if (sibling) {
                ++probes_reordered;
                e.stable = 0; // the order is not stable: no promotion from this slot for now
                continue;
            }
            ++probes_changed;
            if (mismatch_prints < 6 || (frame % 600 < 3 && mismatch_prints < 600)) {
                ++mismatch_prints;
                float dmax = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    dmax = std::max({dmax, std::abs(e.stats.min[c] - s.min[c]), std::abs(e.stats.max[c] - s.max[c])});
                }
                const float dist = std::sqrt((s.centroid[0] - camera_world[0]) * (s.centroid[0] - camera_world[0]) +
                                             (s.centroid[1] - camera_world[1]) * (s.centroid[1] - camera_world[1]) +
                                             (s.centroid[2] - camera_world[2]) * (s.centroid[2] - camera_world[2]));
                std::printf("RTAO: probe change at %.1f m: bounds moved up to %.3f m, ", dist, dmax);
                std::printf("RTAO: probe change: count %u -> %u, centroid (%.3f %.3f %.3f) -> (%.3f %.3f %.3f), "
                            "first (%.3f %.3f %.3f) -> (%.3f %.3f %.3f), camera (%.1f %.1f %.1f)\n",
                            e.stats.count, s.count, e.stats.centroid[0], e.stats.centroid[1], e.stats.centroid[2],
                            s.centroid[0], s.centroid[1], s.centroid[2], e.stats.first[0], e.stats.first[1],
                            e.stats.first[2], s.first[0], s.first[1], s.first[2], camera_world[0],
                            camera_world[1], camera_world[2]);
            }
        }
        if (e.blas) { // cached, then it moved (a door, a lift): back to per frame
            Evict(e);
            ++e.evictions;
            ++evictions_total;
        }
        e.stats = s;
        e.have_stats = true;
        e.stable = 1;
    }
}

void RayTracedAO::TrimCache() {
    if (cache_bytes > cache_budget) {
        std::vector<std::pair<u32, u64>> order; // last seen, key
        for (const auto& [key, e] : cache) {
            if (e.blas && e.last_seen != frame) {
                order.emplace_back(e.last_seen, key);
            }
        }
        std::sort(order.begin(), order.end());
        for (const auto& [seen, key] : order) {
            if (cache_bytes <= cache_budget * 9 / 10) {
                break;
            }
            Evict(cache[key]);
        }
    }
    if (frame % 600 == 0) { // forget draws not seen for a minute
        for (auto it = cache.begin(); it != cache.end();) {
            if (frame - it->second.last_seen > 3600) {
                if (it->second.blas) {
                    Evict(it->second);
                }
                it = cache.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void RayTracedAO::OnColorTarget(VideoCore::ImageId color, u64 ps_hash) {
    if (!enabled) {
        return;
    }
    if (color != last_color) {
        // The previous pass ended. Was it the game's AO upsample (into the R8 mask)?
        if (last_color && last_color == mask_candidate && !done_this_frame && depth_id && active &&
            !draws.empty()) {
            done_this_frame = true;
            static bool reported_heuristic = false;
            if (mask_by_heuristic && !reported_heuristic) {
                reported_heuristic = true;
                std::puts("RTAO: the AO mask was found by its format (R8, scene size): the game's "
                          "upsample shader was not seen yet");
            }
            Record(mask_candidate);
        }
        last_color = color;
        mask_candidate = {};
        mask_by_heuristic = false;
    }
    if (!color || !depth_id || mask_candidate) {
        return;
    }
    if (ps_hash == AoUpsampleShader) {
        if (!upsample_seen) {
            upsample_seen = true;
            std::printf("RTAO: the game's AO pass found by its upsample shader 0x%llx\n",
                        (unsigned long long)AoUpsampleShader);
        }
        mask_candidate = color;
        return;
    }
    if (upsample_seen) {
        return; // the shader identifies the pass; the format heuristic is not needed
    }
    const auto& image = texture_cache.GetImage(color);
    const auto& depth = texture_cache.GetImage(depth_id);
    if (image.info.pixel_format == vk::Format::eR8Unorm &&
        image.info.size.width == depth.info.size.width &&
        image.info.size.height == depth.info.size.height) {
        mask_candidate = color;
        mask_by_heuristic = true;
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
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eTransferSrc, // the dump= copy
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
    SetObjectName(device, vk::Image(ao_images[0]), "rtao ao 0");
    SetObjectName(device, vk::Image(ao_images[1]), "rtao ao 1");
    SetObjectName(device, vk::Image(motion_image), "rtao motion");
    SetObjectName(device, vk::Image(blur_image), "rtao blur");
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
    // The host ring slot (geometry table, instances, constants): only once the GPU is done with it.
    const u32 slot = ring_test ? 0u : frame % TableSlots;
    if (slot_ticks[slot] && !scheduler.IsFree(slot_ticks[slot])) {
        ++skipped_busy; // the game's own AO stays this frame
        return;
    }

    const auto depth_view = CachedView(depth, depth.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const auto normal_view = CachedView(normal, normal.info.pixel_format, vk::ImageAspectFlagBits::eColor);
    const auto mask_view = CachedView(mask, vk::Format::eR8Unorm, vk::ImageAspectFlagBits::eColor);

    scheduler.EndRendering();
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&normal, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&mask, vk::ImageLayout::eGeneral, Compute, rw);
    runtime.FlushBarriers();

    // The history pair swaps once the trace is recorded (the stage limiter may stop before it).
    const u32 read = ao_index, write = ao_index ^ 1;
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

    // Static cache. The probes of the pass that last used this ring slot are read back first
    // (four passes ago: long finished), then the frame's draws are sorted: cached ones become
    // instances of their world-space BLAS, the rest is per-frame geometry, and draws that stayed
    // the same for promote_after observations get their BLAS built this frame.
    const u32 probe_id = recorded_total;
    const u32 ring = probe_id % ProbeSlots;
    if (probe_ids[ring] && probe_ticks[ring] && !scheduler.IsFree(probe_ticks[ring])) {
        probe_ids[ring] = 0; // not finished: those results are dropped, not read half-written
    }
    ReadProbes(ring);
    const auto view = camera_motion.View();
    const auto inv_view = camera_motion.InvView();
    camera_world = {inv_view[3], inv_view[7], inv_view[11]};
    // Cuts the upscaler does not report: frames without this pass (loading screens, menus) or a
    // camera jump no walk or sprint makes in one frame (warps, scripted cuts).
    {
        const float dx = camera_world[0] - last_camera_world[0], dy = camera_world[1] - last_camera_world[1],
                    dz = camera_world[2] - last_camera_world[2];
        if (history_valid && (frame != last_recorded_frame + 1 || dx * dx + dy * dy + dz * dz > 1.0f)) {
            history_valid = false;
            ++history_resets;
        }
        last_recorded_frame = frame;
        last_camera_world = camera_world;
    }
    const bool use_cache = cache_enabled && stage >= 3;
    const float cull = ao_far > 0.0f ? ao_far + radius : 0.0f;
    const auto device = instance.GetDevice();
    struct Promotion {
        vk::AccelerationStructureKHR blas;
        u32 out_base, triangles;
        vk::DeviceAddress scratch;
    };
    std::vector<GpuGeometry> dynamic_list, verify_list, promo_list;
    std::vector<u64> dynamic_keys, verify_keys;
    std::vector<Promotion> promos;
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    u32 promo_used = 0;
    vk::DeviceSize promo_scratch_used = 0;
    frame_triangles = 0;
    cache_instances = 0;
    cache_instance_tris = 0;
    const auto make_instance = [&](vk::DeviceAddress address, bool world) {
        vk::AccelerationStructureInstanceKHR as_instance{
            .instanceCustomIndex = world ? 1u : 0u,
            .mask = 0xFF,
            .instanceShaderBindingTableRecordOffset = 0,
            .flags = VkGeometryInstanceFlagsKHR(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable),
            .accelerationStructureReference = address,
        };
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                as_instance.transform.matrix[r][c] = world ? view[r * 4 + c] : (r == c ? 1.0f : 0.0f);
            }
        }
        return as_instance;
    };
    instances.push_back({}); // slot 0: the per-frame BLAS (filled once its size is known)
    // A mesh leaves the per-frame BVH only when every copy drawn this frame is cached: which draw
    // is which copy changes with the draw order, but the set of cached copies covers them all.
    std::unordered_map<u64, bool> mesh_cached;
    // Huge draws (terrain, building blocks) stay per frame: whole in a cached BLAS, their far
    // triangles came back into the BVH (the far cut works per frame) and rays slowed down.
    const auto too_big = [&](const CacheEntry& e) {
        if (cache_max_size <= 0.0f || !e.have_stats) {
            return false;
        }
        const float dx = e.stats.max[0] - e.stats.min[0], dy = e.stats.max[1] - e.stats.min[1],
                    dz = e.stats.max[2] - e.stats.min[2];
        return dx * dx + dy * dy + dz * dz > cache_max_size * cache_max_size;
    };
    for (const Draw& d : draws) {
        CacheEntry& e = cache[d.key];
        e.base = d.base;
        e.last_seen = frame;
        if (e.blas && too_big(e)) {
            Evict(e);
        }
        auto [it, inserted] = mesh_cached.try_emplace(d.base, true);
        it->second = it->second && use_cache && e.blas && frame >= e.ready_from;
    }
    // Probing costs a pass over the draw's indices: cached draws are rechecked every 8th pass,
    // draws waiting for a build slot every 4th; characters and draws that kept changing never.
    const auto probe_now = [&](const Draw& d, const CacheEntry& e) {
        if (d.skinned || e.evictions >= 3) {
            return false;
        }
        const u32 phase = u32(d.key >> 7) + probe_id;
        if (e.blas) {
            return phase % 8 == 0;
        }
        if (too_big(e)) {
            return phase % 64 == 0; // it will stay per frame; an occasional look is enough
        }
        if (e.stable >= promote_after) {
            return phase % 4 == 0;
        }
        return true;
    };
    for (const Draw& d : draws) {
        CacheEntry& e = cache[d.key];
        GpuGeometry gpu = d.gpu;
        const u32 probe_flag = probe_now(d, e) ? GeoProbe : 0u;
        if (mesh_cached[d.base]) {
            bool far = false;
            if (cull > 0.0f) { // distance from the camera to the draw's box
                float d2 = 0.0f;
                for (int i = 0; i < 3; ++i) {
                    const float v = std::max({e.stats.min[i] - camera_world[i], 0.0f,
                                              camera_world[i] - e.stats.max[i]});
                    d2 += v * v;
                }
                far = d2 > cull * cull;
            }
            if (far || instances.size() < MaxInstances) {
                if (!far) {
                    instances.push_back(make_instance(e.address, true));
                    ++cache_instances;
                    cache_instance_tris += e.triangles;
                }
                if (probe_flag) { // still checked: a door that starts moving leaves the cache
                    gpu.triangles = 0;
                    gpu.flags = probe_flag;
                    verify_list.push_back(gpu);
                    verify_keys.push_back(d.key);
                }
                continue;
            }
        }
        if (frame_triangles + gpu.triangles > max_triangles) {
            ++dropped_draws;
            dropped_triangles += gpu.triangles;
            if (probe_flag) { // measured all the same: once cached it no longer needs budget
                gpu.triangles = 0;
                gpu.flags = probe_flag;
                verify_list.push_back(gpu);
                verify_keys.push_back(d.key);
            }
            continue;
        }
        gpu.triangle_offset = frame_triangles;
        gpu.out_base = frame_triangles;
        gpu.flags = probe_flag;
        frame_triangles += gpu.triangles;
        dynamic_list.push_back(gpu);
        dynamic_keys.push_back(d.key);
        if (!use_cache || d.skinned || e.blas || e.evictions >= 3 || e.stable < promote_after || too_big(e) ||
            gpu.triangles == 0 || promo_used + gpu.triangles > PromoTriangles || promos.size() >= 256) {
            continue;
        }
        { // the same copy cached under another slot (before a reorder): not twice
            bool duplicate = false;
            const auto copies = max_copies.find(d.base);
            for (u32 o = 0; copies != max_copies.end() && o < copies->second && !duplicate; ++o) {
                const auto other = cache.find(KeyOf(d.base, o));
                duplicate = other != cache.end() && &other->second != &e && other->second.blas &&
                            StatsMatch(other->second.stats, e.stats);
            }
            if (duplicate) {
                continue;
            }
        }
        // Promotion: a world-space copy of the triangles, built into the draw's own BLAS.
        vk::AccelerationStructureGeometryKHR geometry{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = {.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                             .vertexFormat = vk::Format::eR32G32B32Sfloat,
                             .vertexStride = 3 * sizeof(float),
                             .maxVertex = gpu.triangles * 3 - 1,
                             .indexType = vk::IndexType::eNoneKHR,
                         }},
            .flags = vk::GeometryFlagBitsKHR::eOpaque,
        };
        const vk::AccelerationStructureBuildGeometryInfoKHR info{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &geometry,
        };
        const auto sizes = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, info, gpu.triangles);
        const vk::DeviceSize scratch_at =
            (promo_scratch_used + scratch_alignment - 1) / scratch_alignment * scratch_alignment;
        if (scratch_at + sizes.buildScratchSize > promo_scratch.size) {
            continue;
        }
        Buffer storage;
        if (!CreateBuffer(storage, sizes.accelerationStructureSize,
                          vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR, false)) {
            continue;
        }
        const auto as = device.createAccelerationStructureKHR({
            .buffer = storage.buffer,
            .size = sizes.accelerationStructureSize,
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        });
        if (as.result != vk::Result::eSuccess) {
            DestroyBuffer(storage);
            continue;
        }
        e.blas = as.value;
        SetObjectName(device, e.blas, "rtao cached BLAS");
        e.buffer = storage.buffer;
        e.allocation = storage.allocation;
        e.address = device.getAccelerationStructureAddressKHR({.accelerationStructure = e.blas});
        e.bytes = sizes.accelerationStructureSize;
        e.triangles = gpu.triangles;
        e.ready_from = frame + 1; // built in this pass, instanced from the next
        cache_bytes += e.bytes;
        ++promotions;
        promo_scratch_used = scratch_at + sizes.buildScratchSize;
        promos.push_back({e.blas, max_triangles + promo_used, gpu.triangles,
                          promo_scratch.address + scratch_at});
        GpuGeometry world = gpu;
        world.out_base = max_triangles + promo_used;
        world.flags = GeoWorld;
        promo_list.push_back(world);
        promo_used += gpu.triangles;
    }
    // The table: per-frame entries, then the world-space copies (expanded after them), then the
    // entries that are only measured (no triangles; their offset lies past the expand range).
    geometries.clear();
    probe_keys[ring].clear();
    geometries.insert(geometries.end(), dynamic_list.begin(), dynamic_list.end());
    for (size_t i = 0; i < dynamic_list.size(); ++i) {
        probe_keys[ring].push_back((dynamic_list[i].flags & GeoProbe) ? dynamic_keys[i] : 0);
    }
    u32 dispatch_triangles = frame_triangles;
    for (auto& world : promo_list) {
        world.triangle_offset = dispatch_triangles;
        dispatch_triangles += world.triangles;
        geometries.push_back(world);
        probe_keys[ring].push_back(0);
    }
    for (size_t i = 0; i < verify_list.size() && geometries.size() < max_geometries; ++i) {
        verify_list[i].triangle_offset = dispatch_triangles;
        geometries.push_back(verify_list[i]);
        probe_keys[ring].push_back(verify_keys[i]);
    }
    probe_ids[ring] = probe_id;
    // This pass's commands go out with the current tick: the slots are busy until it is reached.
    slot_ticks[slot] = scheduler.CurrentTick();
    probe_ticks[ring] = slot_ticks[slot];
    auto* table = static_cast<GpuGeometry*>(table_buffer.mapped) + size_t(slot) * max_geometries;
    std::memcpy(table, geometries.data(), geometries.size() * sizeof(GpuGeometry));
    vmaFlushAllocation(instance.GetAllocator(), table_buffer.allocation,
                       vk::DeviceSize(slot) * max_geometries * sizeof(GpuGeometry),
                       std::max<vk::DeviceSize>(geometries.size(), 1) * sizeof(GpuGeometry));
    const u32 num_geometries = u32(geometries.size());
    const u32 total_triangles = frame_triangles; // the per-frame BLAS
    // Instances: the per-frame BLAS first (when it has triangles), then the cached ones.
    if (total_triangles) {
        instances[0] = make_instance(blas_address, false);
    } else {
        instances.erase(instances.begin());
    }
    const u32 num_instances = u32(instances.size());
    const vk::DeviceSize instance_offset =
        vk::DeviceSize(slot) * MaxInstances * sizeof(vk::AccelerationStructureInstanceKHR);
    std::memcpy(static_cast<u8*>(instance_buffer.mapped) + instance_offset, instances.data(),
                instances.size() * sizeof(vk::AccelerationStructureInstanceKHR));
    vmaFlushAllocation(instance.GetAllocator(), instance_buffer.allocation, instance_offset,
                       std::max<vk::DeviceSize>(instances.size(), 1) *
                           sizeof(vk::AccelerationStructureInstanceKHR));
    const bool do_readback = !readback_pending && total_triangles > 0 &&
                             (recorded_total == 1 || frame % 600 == 300);
    if (do_readback) {
        readback_pending = true;
        readback_frame = frame;
        readback_triangles = total_triangles;
        readback_tick = scheduler.CurrentTick();
    }
    const bool do_dump = dump_request != dump_done && !dump_pending && stage >= 4 && dump_buffer.mapped;
    if (do_dump) {
        dump_pending = true;
        dump_frame = frame;
        dump_done = dump_request;
        dump_tick = scheduler.CurrentTick();
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
        .total_triangles = dispatch_triangles,
        .far_cut = ao_far > 0.0f ? ao_far + radius : 0.0f,
        .inv_view = inv_view,
    };
    const ProbeConstants probe{
        .positions_address = expand.positions_address,
        .num_geometries = num_geometries,
        .id = probe_id,
        .proj = expand.proj,
        .inv_view = inv_view,
    };
    const vk::Buffer probe_handle = probe_buffer.buffer;
    const vk::DeviceSize probe_offset = vk::DeviceSize(ring) * max_geometries * sizeof(ProbeStats);
    const vk::DeviceSize probe_bytes = vk::DeviceSize(max_geometries) * sizeof(ProbeStats);
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
        .normals = depth_normals ? 1u : 0u,
        .offset = origin_offset,
    };
    const vk::DeviceSize constants_offset = vk::DeviceSize(slot) * ConstantsStride;
    std::memcpy(static_cast<u8*>(constants_buffer.mapped) + constants_offset, &trace, sizeof(trace));
    vmaFlushAllocation(instance.GetAllocator(), constants_buffer.allocation, constants_offset, sizeof(trace));
    const vk::Buffer constants_handle = constants_buffer.buffer;
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

    const vk::Buffer table_handle = table_buffer.buffer;
    const vk::DeviceSize table_offset = vk::DeviceSize(slot) * max_geometries * sizeof(GpuGeometry);
    const vk::DeviceSize table_bytes = vk::DeviceSize(std::max<u32>(num_geometries, 1)) * sizeof(GpuGeometry);
    const vk::Buffer triangles_handle = triangle_buffer.buffer;
    const vk::DeviceSize triangles_bytes = triangle_buffer.size;
    const vk::DeviceAddress triangles_address = triangle_buffer.address;
    const vk::DeviceAddress blas_scratch_address = blas_scratch.address;
    const vk::DeviceAddress tlas_scratch_address = tlas_scratch.address;
    const vk::DeviceAddress instances_address = instance_buffer.address + instance_offset;
    const vk::ImageView ao_read = *ao_views[read], ao_write = *ao_views[write];
    const vk::ImageView blur_rw = *blur_view;
    const BlurConstants blur_h{.half_size = {float((w + 1) / 2), float((h + 1) / 2)}, .dir = {1, 0}, .sigma = sigma};
    const BlurConstants blur_v{.half_size = blur_h.half_size, .dir = {0, 1}, .sigma = sigma};
    const vk::ImageView motion = *motion_view;

    // Five crumbs, so a GPU hang names the stage; "stage" in the settings stops after that stage.
    const u32 last_stage = stage;
    const bool fast_trace = build_fast_trace;
    const vk::BuildAccelerationStructureFlagsKHR extra_flags =
        data_access ? vk::BuildAccelerationStructureFlagBitsKHR::eAllowDataAccess
                    : vk::BuildAccelerationStructureFlagsKHR{};
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
        [=, expand_pipeline = *expand_pipeline, expand_layout = *expand_layout,
         probe_pipeline = *probe_pipeline, probe_layout = *probe_layout](vk::CommandBuffer cmdbuf) {
            // The inputs: captured positions (vertex shaders), index buffers (uploads, or game
            // compute), the table and constants (host). Not ALL_COMMANDS: no render-target writes.
            memory_barrier(cmdbuf,
                           vk::PipelineStageFlagBits2::eVertexShader | Compute |
                               vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eHost,
                           vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite |
                               vk::AccessFlagBits2::eHostWrite,
                           Compute, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
            if (dispatch_triangles) {
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
                cmdbuf.dispatch((dispatch_triangles + 63) / 64, 1, 1);
            }
            if (num_geometries) { // the static cache's measurements, read back by the host later
                const vk::DescriptorBufferInfo table_info{table_handle, table_offset, table_bytes};
                const vk::DescriptorBufferInfo probe_info{probe_handle, probe_offset, probe_bytes};
                const std::array writes = {
                    vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &table_info},
                    vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                                           .pBufferInfo = &probe_info}};
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, probe_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, probe_layout, 0, writes);
                cmdbuf.pushConstants(probe_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(probe), &probe);
                cmdbuf.dispatch(num_geometries, 1, 1);
                memory_barrier(cmdbuf, Compute, vk::AccessFlagBits2::eShaderWrite,
                               vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
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
        // The static cache's new BLASes (world space, built for trace speed).
        if (!promos.empty()) {
            std::vector<vk::AccelerationStructureGeometryKHR> geos(promos.size());
            std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> infos(promos.size());
            std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges(promos.size());
            std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_ptrs(promos.size());
            for (size_t i = 0; i < promos.size(); ++i) {
                const auto& pr = promos[i];
                geos[i] = vk::AccelerationStructureGeometryKHR{
                    .geometryType = vk::GeometryTypeKHR::eTriangles,
                    .geometry = {.triangles = vk::AccelerationStructureGeometryTrianglesDataKHR{
                                     .vertexFormat = vk::Format::eR32G32B32Sfloat,
                                     .vertexData = {.deviceAddress = triangles_address +
                                                                     vk::DeviceSize(pr.out_base) * 36},
                                     .vertexStride = 3 * sizeof(float),
                                     .maxVertex = pr.triangles * 3 - 1,
                                     .indexType = vk::IndexType::eNoneKHR,
                                 }},
                    .flags = vk::GeometryFlagBitsKHR::eOpaque,
                };
                infos[i] = vk::AccelerationStructureBuildGeometryInfoKHR{
                    .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
                    .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
                    .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
                    .dstAccelerationStructure = pr.blas,
                    .geometryCount = 1,
                    .pGeometries = &geos[i],
                    .scratchData = {.deviceAddress = pr.scratch},
                };
                ranges[i] = vk::AccelerationStructureBuildRangeInfoKHR{.primitiveCount = pr.triangles};
                range_ptrs[i] = &ranges[i];
            }
            cmdbuf.buildAccelerationStructuresKHR(u32(infos.size()), infos.data(), range_ptrs.data());
        }
        if (!total_triangles) {
            memory_barrier(cmdbuf, AsBuild, vk::AccessFlagBits2::eAccelerationStructureWriteKHR, AsBuild,
                           vk::AccessFlagBits2::eAccelerationStructureReadKHR);
            return;
        }
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
                     extra_flags,
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
        const vk::AccelerationStructureBuildRangeInfoKHR tlas_range{.primitiveCount = num_instances};
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
            const vk::DescriptorBufferInfo constants_info{constants_handle, constants_offset, sizeof(trace)};
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
                                       .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
                vk::WriteDescriptorSet{.dstBinding = 6, .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .pBufferInfo = &constants_info}};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, trace_pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, trace_layout, 0, writes);
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

    ao_index = write;
    history_valid = true;

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
