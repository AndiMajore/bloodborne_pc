// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_gpu_busy.h"

#include <algorithm>
#include <cstdio>

#include "render/scheduler.h"
#include "vulkan/instance.h"

namespace Vulkan {

void GpuBusy::Init(const Instance& instance) {
    if (!instance_ptr) {
        instance_ptr = new GpuBusy(instance);
    }
}

GpuBusy::GpuBusy(const Instance& instance) : device{instance.GetDevice()} {
    pool = Check<"create timestamp query pool">(device.createQueryPoolUnique({
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = Pairs * 2,
    }));
    device.resetQueryPool(*pool, 0, Pairs * 2);
    period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    free_pairs.reserve(Pairs);
    for (u32 i = Pairs; i-- > 0;) {
        free_pairs.push_back(i);
    }
}

void GpuBusy::Begin(Scheduler& scheduler) {
    u32 pair;
    {
        std::scoped_lock lock{mutex};
        Collect();
        if (free_pairs.empty() || open.contains(&scheduler)) {
            return; // the GPU this far behind (or a Begin twice): this submission goes uncounted
        }
        pair = free_pairs.back();
        free_pairs.pop_back();
        open[&scheduler] = pair;
    }
    scheduler.Record([pool = *pool, pair](vk::CommandBuffer cmdbuf) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, pair * 2);
    });
}

void GpuBusy::End(Scheduler& scheduler, u64 tick) {
    u32 pair;
    {
        std::scoped_lock lock{mutex};
        const auto it = open.find(&scheduler);
        if (it == open.end()) {
            return;
        }
        pair = it->second;
        open.erase(it);
        pending.push_back({pair, &scheduler, tick});
    }
    scheduler.Record([pool = *pool, pair](vk::CommandBuffer cmdbuf) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, pair * 2 + 1);
    });
}

void GpuBusy::Collect() {
    // In submission order; a scheduler's later ticks wait behind its earlier ones.
    while (!pending.empty()) {
        const Pending p = pending.front();
        if (!p.scheduler->IsFree(p.tick)) {
            break;
        }
        pending.pop_front();
        u64 stamps[2] = {};
        const auto result = device.getQueryPoolResults(*pool, p.pair * 2, 2, sizeof(stamps), stamps, sizeof(u64),
                                                       vk::QueryResultFlagBits::e64);
        device.resetQueryPool(*pool, p.pair * 2, 2);
        free_pairs.push_back(p.pair);
        if (result != vk::Result::eSuccess || stamps[1] < stamps[0]) {
            continue;
        }
        const double begin = double(stamps[0]) * period_ns, end = double(stamps[1]) * period_ns;
        if (window_start < 0.0) {
            window_start = begin;
        }
        window_busy += end - begin;
        window_last = std::max(window_last, end);
        if (window_last - window_start >= WindowNs) {
            const double share = window_busy / (window_last - window_start);
            utilisation.store(float(std::min(share, 1.0)), std::memory_order_relaxed);
            windows.fetch_add(1, std::memory_order_relaxed);
            window_start = -1.0;
            window_busy = 0.0;
            window_last = 0.0;
        }
    }
}

} // namespace Vulkan
