// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the GPU's utilisation, always on and cheap (ticket 71, the frame-time governor's
// input): a timestamp at the start and at the end of every submission (both schedulers: the
// game's and the presenter's), outside render passes; the sum of the submissions' spans over the
// span of the last half second of GPU time. At 60 fps with vsync 0.84 means 14 ms of 16.7 busy;
// GPU-bound frames read near 1. BB_GPU_PROFILE's profiler labels every pass instead and ends
// render passes for it: too costly to leave on.

#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

class GpuBusy {
public:
    static void Init(const Instance& instance);
    /// Null before Init (or when the device has no timestamps on its queue).
    static GpuBusy* Get() noexcept {
        return instance_ptr;
    }

    /// A submission's command buffers were just begun: its first command (outside a pass).
    void Begin(Scheduler& scheduler);
    /// Before its command buffers end (outside a pass): closes the span; `tick` signals its end.
    void End(Scheduler& scheduler, u64 tick);

    /// Busy share of the GPU's time over the last window (0..1); negative before the first one.
    [[nodiscard]] float Utilisation() const noexcept {
        return utilisation.load(std::memory_order_relaxed);
    }
    /// Counts the windows measured (a new value: a new reading).
    [[nodiscard]] u32 Windows() const noexcept {
        return windows.load(std::memory_order_relaxed);
    }

private:
    explicit GpuBusy(const Instance& instance);
    void Collect();

    static inline GpuBusy* instance_ptr = nullptr;
    static constexpr u32 Pairs = 1024;
    static constexpr double WindowNs = 500e6;

    struct Pending {
        u32 pair;
        Scheduler* scheduler;
        u64 tick;
    };
    vk::Device device;
    vk::UniqueQueryPool pool;
    double period_ns = 1.0;
    std::mutex mutex;
    std::vector<u32> free_pairs;
    std::unordered_map<Scheduler*, u32> open; // the pair a scheduler's current submission writes
    std::deque<Pending> pending;
    double window_start = -1.0, window_last = 0.0, window_busy = 0.0;
    std::atomic<float> utilisation{-1.0f};
    std::atomic<u32> windows{0};
};

} // namespace Vulkan
