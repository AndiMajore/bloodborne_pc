// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: when the transparents' light is taken (ticket 63; vk_rt_emission.cpp). The opaque scene
// is kept at the first transparent draw (Begin); the game then draws the transparents into the
// scene colour with the scene depth (86 draws at the Central Yharnam burning stakes), then the
// full-screen fog composite without a depth target, then into other images. The emission is taken
// before the first draw that is not a transparent of the scene: another target, or no scene depth
// (with the fog included the whole scene would glow grey). Pure (tested in rtao-geometry-test).

#pragma once

#include <cstdint>

namespace EmissionRules {

class Trigger {
public:
    /// The opaque scene was kept; `scene` identifies the scene colour target.
    void Begin(uint64_t scene) {
        this->scene = scene;
        draws = 0;
        pending = true;
    }
    /// A draw after it: true when the emission is to be taken before this draw (once).
    bool OnDraw(uint64_t color, bool scene_depth) {
        if (!pending) {
            return false;
        }
        ++draws;
        if (color && (color != scene || !scene_depth)) {
            pending = false;
            return true;
        }
        return false;
    }
    /// Given up (a frame whose transparents never ended): no more draws wait.
    void Taken() {
        pending = false;
    }
    [[nodiscard]] bool Pending() const noexcept {
        return pending;
    }
    /// Draws since Begin (the one that ended the transparents included).
    [[nodiscard]] uint32_t Draws() const noexcept {
        return draws;
    }

private:
    uint64_t scene = 0;
    uint32_t draws = 0;
    bool pending = false;
};

} // namespace EmissionRules
