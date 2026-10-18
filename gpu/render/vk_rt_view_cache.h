// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a few views of the game's images (depth, normals, colour targets) for the RT lighting
// passes, reused while the image is the same (its handle and uid), the least recently used
// replaced (its old view destroyed once the GPU is past it).

#pragma once

#include <array>
#include "common/types.h"
#include "vulkan/vk_common.h"

namespace VideoCore {
struct Image;
}

namespace Vulkan {
class Instance;
class Scheduler;

class RtViewCache {
public:
    RtViewCache(const Instance& instance, Scheduler& scheduler) : instance{instance}, scheduler{scheduler} {}
    ~RtViewCache();
    RtViewCache(const RtViewCache&) = delete;
    RtViewCache& operator=(const RtViewCache&) = delete;

    /// A 2D view of the image's first level and layer.
    vk::ImageView Get(const VideoCore::Image& image, vk::Format format, vk::ImageAspectFlags aspect);

private:
    struct Entry {
        vk::Image image{};
        u64 uid = 0;
        vk::Format format{};
        vk::ImageAspectFlags aspect{};
        vk::ImageView view{};
        u64 last_use = 0;
    };
    const Instance& instance;
    Scheduler& scheduler;
    std::array<Entry, 6> entries{};
    u64 uses = 0;
};

} // namespace Vulkan
