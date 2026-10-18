// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_rt_view_cache.h"

#include "render/scheduler.h"
#include "textures/image.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"

namespace Vulkan {

RtViewCache::~RtViewCache() {
    const auto device = instance.GetDevice();
    for (auto& entry : entries) {
        if (entry.view) {
            device.destroyImageView(entry.view);
        }
    }
}

vk::ImageView RtViewCache::Get(const VideoCore::Image& image, vk::Format format, vk::ImageAspectFlags aspect) {
    const vk::Image handle = image.GetImage();
    Entry* oldest = &entries[0];
    for (auto& entry : entries) {
        if (entry.view && entry.image == handle && entry.uid == image.image_uid && entry.format == format &&
            entry.aspect == aspect) {
            entry.last_use = ++uses;
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
        .last_use = ++uses,
    };
    return oldest->view;
}

} // namespace Vulkan
