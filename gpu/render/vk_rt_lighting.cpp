// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_rt_lighting.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sys/stat.h>
#include "common/path_util.h"
#include "render/vk_rtao.h"

namespace Vulkan {

RtLighting::RtLighting(const Instance& instance, Scheduler& scheduler, VideoCore::TextureCache& texture_cache,
                       Runtime& runtime, CameraMotion& camera_motion, const RayTracedAO& rtao)
    : rtao{rtao} {
    lights = std::make_unique<RtLights>(camera_motion);
    shadows = std::make_unique<RtShadows>(instance, scheduler, texture_cache, runtime, camera_motion, rtao, *lights,
                                          settings);
    emission = std::make_unique<RtEmission>(instance, scheduler, texture_cache, runtime, camera_motion, rtao, *lights,
                                            settings);
    if (const char* file = std::getenv("BB_RTLIGHTING_FILE"); file && *file) {
        settings_path = file;
    } else {
        settings_path = (Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "rtlighting.txt").string();
    }
}

RtLighting::~RtLighting() = default;

void RtLighting::PollSettings() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_poll < std::chrono::seconds(1)) {
        return;
    }
    last_poll = now;
    struct stat st{};
    if (::stat(settings_path.c_str(), &st) != 0) {
        return; // no file: everything stays off
    }
    const long long mtime = static_cast<long long>(st.st_mtim.tv_sec) * 1000000000LL + st.st_mtim.tv_nsec;
    if (mtime == settings_mtime) {
        return;
    }
    settings_mtime = mtime;
    std::ifstream file(settings_path);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    RtLightingSettings::ApplyText(settings, text);
    std::printf("RT lighting: settings from %s: emission %.2f, fire light %.2f (base %.2f: %.2f x %.2f m, lift %.2f m, spark %.2f: "
                "%.2f m, wrap %.2f, smooth %.2f, switch %.2f, hold %u, merge %.2f m), sun shadow %d, spot shadow %d "
                "(%u, blur %u, soft %.2f m)\n",
                settings_path.c_str(), settings.emission, settings.fire_light, settings.fire_base,
                settings.fire_base_size, settings.fire_base_height, settings.fire_base_lift, settings.fire_spark, settings.fire_spark_size,
                settings.fire_wrap, settings.fire_smooth, settings.fire_switch, settings.fire_hold, settings.fire_merge,
                settings.sun_shadow ? 1 : 0,
                settings.spot_shadow ? 1 : 0, settings.spot_res, settings.spot_blur, settings.spot_soft);
}

void RtLighting::UpdateWants() {
    const bool on = rtao.Enabled();
    const bool fires = settings.fire_light > 0.0f && fire_reader;
    wants_constants = on && shadows->WantsConstants();
    // The light list: for the spot's shadow, a dump, or the fires' bases (anchored to its fire lights).
    wants_lights = on && (settings.spot_shadow || settings.lights_dump != dump_done ||
                          fires);
    wants_draws = on && shadows->WantsDraws();
    wants_emission = on && (settings.emission > 0.0f || fires);
}

void RtLighting::OnFrameStart() {
    if (!rtao.Enabled()) {
        return;
    }
    // The frame that ended: its light list (when it was collected), then this frame's settings.
    if (lights_collected) {
        lights->FinishFrame(frame, settings.lights_dump);
        dump_done = settings.lights_dump;
    }
    ++frame;
    PollSettings();
    UpdateWants();
    lights_collected = wants_lights;
    shadows->OnFrameStart(frame);
    emission->OnFrameStart(frame);
}

void RtLighting::OnGBufferPass(VideoCore::ImageId depth, VideoCore::ImageId normal) {
    shadows->OnGBufferPass(depth, normal);
    emission->OnGBufferPass(depth);
}

void RtLighting::OnDraw(VideoCore::ImageId color, u64 ps_hash, bool scene_depth) {
    if (wants_draws) {
        shadows->OnDraw(color, ps_hash);
    }
    if (emission->DrawsWanted()) {
        emission->OnDrawAfterOpaque(color, scene_depth);
    }
}

} // namespace Vulkan
