// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: ray-traced AO, what the pass's translation units share (vk_rtao.cpp, vk_rtao_resources.cpp,
// vk_rtao_diagnostics.cpp): the GPU constant blocks and small helpers.

#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan::RtaoInternal {
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
    u32 pass_index;
    u32 num_chunks;
};
static_assert(sizeof(ProbeConstants) == 88 && offsetof(ProbeConstants, pass_index) == 80);
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
    u32 denoise;
    u32 subpixel;
    u32 falloff;
    u32 period;
};
struct TemporalConstants {
    std::array<float, 2> half_size;
    std::array<s32, 2> sub;
    float max_n;
    float clip_k;
    u32 history_ok;
    float clip_floor;
};
static_assert(sizeof(TemporalConstants) == 32);
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
static_assert(sizeof(TraceConstants) == 140 && offsetof(TraceConstants, falloff) == 132);
struct BlurConstants {
    std::array<float, 2> half_size;
    std::array<s32, 2> dir;
    float sigma;
    u32 use_n;
};
static_assert(sizeof(ApplyConstants) == 56);
static_assert(sizeof(BlurConstants) == 24);
/// The pass's one helper thread (bb:RtaoSplit, on otherwise idle cores): jobs run in order,
/// each hands its result back through a future. Started with the first job, joined on
/// destruction after the jobs left (the owner waits for its futures before freeing what they use).
class Worker {
public:
    ~Worker() {
        {
            std::scoped_lock lock{mutex};
            stop = true;
        }
        wake.notify_one();
        if (thread.joinable()) {
            thread.join();
        }
    }
    template <class F>
    auto Post(F&& job) -> std::future<decltype(job())> {
        using Result = decltype(job());
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(job));
        auto future = task->get_future();
        {
            std::scoped_lock lock{mutex};
            jobs.emplace_back([task] { (*task)(); });
            if (!thread.joinable()) {
                thread = std::thread([this] { Run(); });
            }
        }
        wake.notify_one();
        return future;
    }

private:
    void Run();
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::function<void()>> jobs;
    bool stop = false;
    std::thread thread;
};
inline float EnvFloat(const char* name, float fallback) {
    const char* value = std::getenv(name);
    return value ? std::strtof(value, nullptr) : fallback;
}
inline u32 EnvU32(const char* name, u32 fallback) {
    const char* value = std::getenv(name);
    return value ? u32(std::strtoul(value, nullptr, 10)) : fallback;
}
constexpr auto AsBuild = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;
constexpr auto Compute = vk::PipelineStageFlagBits2::eComputeShader;
// The trace runs in compute (ray queries) or as a ray-generation shader (pipeline=1).
constexpr auto TraceStages = Compute | vk::PipelineStageFlagBits2::eRayTracingShaderKHR;
} // namespace Vulkan::RtaoInternal
