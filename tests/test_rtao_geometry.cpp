// SPDX-License-Identifier: GPL-2.0-or-later
// The ray-traced AO's logic without Vulkan: identities, the static cache's state machine, the
// settings parser, and the records shared with the shaders.
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <set>
#include <vector>
#include "video_core/renderer_vulkan/rtao_geometry.h"

using namespace Rtao;

static ProbeStats Stats(float x, float y, float z, u32 count = 300) {
    ProbeStats s{};
    s.count = count;
    s.centroid[0] = x; s.centroid[1] = y; s.centroid[2] = z;
    for (int i = 0; i < 3; ++i) {
        s.min[i] = s.centroid[i] - 1.0f;
        s.max[i] = s.centroid[i] + 1.0f;
        s.first[i] = s.centroid[i] + 0.5f;
    }
    return s;
}

static void Layouts() {
    // The shaders read these records by offset (rtao_geometry.glsl, rtao_probe.comp).
    static_assert(offsetof(GpuGeometry, index_address) == 16);
    static_assert(offsetof(GpuGeometry, restart) == 36);
    static_assert(offsetof(GpuGeometry, triangle_offset) == 40);
    static_assert(offsetof(GpuGeometry, out_base) == 48);
    static_assert(offsetof(GpuGeometry, flags) == 52);
    static_assert(offsetof(ProbeStats, centroid) == 8);
    static_assert(offsetof(ProbeStats, first) == 44);
}

static void Keys() {
    std::set<u64> keys;
    for (u64 base : {0ull, 1ull, 0xdeadbeefull, ~0ull}) {
        for (u32 occurrence = 0; occurrence < 64; ++occurrence) {
            const u64 key = KeyOf(base, occurrence);
            assert(key != 0);
            keys.insert(key);
        }
    }
    assert(keys.size() == 4 * 64); // no collisions in this small set
    assert(KeyOf(42, 3) == KeyOf(42, 3));
}

static void Matching() {
    const std::array<float, 3> camera{0.0f, 0.0f, 0.0f};
    const auto a = Stats(10.0f, 0.0f, 0.0f);
    auto b = a;
    b.centroid[0] += 0.015f; // within 2 cm + 0.02% of 10 m
    assert(StatsMatch(a, b, camera));
    b.centroid[0] += 0.02f;
    assert(!StatsMatch(a, b, camera));
    // Far away the tolerance grows: 2 cm + 0.02% of 500 m = 12 cm.
    const auto far = Stats(500.0f, 0.0f, 0.0f);
    auto far_moved = far;
    far_moved.first[1] += 0.1f;
    assert(StatsMatch(far, far_moved, camera));
    // Another vertex count is never the same draw.
    assert(!StatsMatch(a, Stats(10.0f, 0.0f, 0.0f, 301), camera));
    assert(!StatsMatch(Stats(0, 0, 0, 0), Stats(0, 0, 0, 0), camera));
    // Size limit: the box's diagonal.
    assert(!TooBig(a, 20.0f));   // 2 x 2 x 2 m
    assert(TooBig(a, 3.0f));     // diagonal 3.46 m
    assert(!TooBig(a, 0.0f));    // 0: no limit
}

static void StateMachine() {
    const std::array<float, 3> camera{0.0f, 0.0f, 0.0f};
    CacheState s;
    const auto here = Stats(5.0f, 1.0f, 2.0f);
    assert(Observe(s, here, false, camera) == Observation::First);
    for (u32 i = 0; i < 7; ++i) {
        assert(!Promotable(s, false, false, 8));
        assert(Observe(s, here, false, camera) == Observation::Same);
    }
    assert(s.stable == 8);
    assert(Promotable(s, false, false, 8));
    assert(!Promotable(s, true, false, 8));  // skinned
    assert(!Promotable(s, false, true, 8));  // too big
    s.cached = true;
    assert(!Promotable(s, false, false, 8)); // already cached
    // A reorder (another copy of the mesh in this slot) keeps the BLAS but stops promotion.
    assert(Observe(s, Stats(9, 9, 9), true, camera) == Observation::Reordered);
    assert(s.cached && s.stable == 0 && s.evictions == 0);
    // A real change of a cached draw: evict.
    assert(Observe(s, Stats(9, 9, 9), false, camera) == Observation::ChangedEvict);
    assert(!s.cached && s.evictions == 1 && s.stable == 1);
    // Not cached: a change is only a change.
    assert(Observe(s, Stats(1, 1, 1), false, camera) == Observation::Changed);
    assert(s.evictions == 1);
    // Three evictions: never again.
    for (int round = 0; round < 2; ++round) {
        for (u32 i = 0; i < 8; ++i) {
            Observe(s, Stats(2, 2, 2), false, camera);
        }
        s.cached = true;
        assert(Observe(s, Stats(3, 3, float(round)), false, camera) == Observation::ChangedEvict);
    }
    assert(s.evictions == 3);
    for (u32 i = 0; i < 20; ++i) {
        Observe(s, Stats(3, 3, 1), false, camera);
    }
    assert(!Promotable(s, false, false, 8));
}

static void Split() {
    // A 100 m x 100 m floor of 1 m quads (two triangles each), shuffled like a game's index order,
    // plus two inactive triangles.
    std::vector<float> tris;
    const auto tri = [&](float x0, float z0, float x1, float z1, float x2, float z2) {
        for (float v : {x0, 0.0f, z0, x1, 0.0f, z1, x2, 0.0f, z2}) {
            tris.push_back(v);
        }
    };
    for (int i = 0; i < 100 * 100; ++i) {
        const int q = (i * 7919) % (100 * 100); // a permutation of the quads
        const float x = float(q % 100), z = float(q / 100);
        tri(x, z, x + 1, z, x, z + 1);
        tri(x + 1, z, x + 1, z + 1, x, z + 1);
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    tri(nan, 0, 0, 0, 0, 0);
    tri(std::numeric_limits<float>::infinity(), 0, 0, 0, 0, 0);
    const u32 count = u32(tris.size() / 9);
    std::vector<float> out(tris.size());
    const auto groups = SplitTriangles(tris.data(), count, 10.0f, 4096, 16, out.data());
    u32 covered = 0, slivers = 0;
    for (const auto& g : groups) {
        assert(g.first_triangle == covered); // consecutive
        assert(g.box.Diagonal() <= 10.0f || g.triangles <= 16); // the floor may leave a sliver
        slivers += g.box.Diagonal() > 10.0f ? 1 : 0;
        assert(g.triangles <= 4096);
        // Every triangle of the piece lies in its box.
        for (u32 t = g.first_triangle; t < g.first_triangle + g.triangles; ++t) {
            for (int k = 0; k < 3; ++k) {
                const float* v = &out[size_t(t) * 9 + k * 3];
                for (int i = 0; i < 3; ++i) {
                    assert(v[i] >= g.box.min[i] && v[i] <= g.box.max[i]);
                }
            }
        }
        covered += g.triangles;
    }
    assert(covered == 20000); // the two inactive triangles dropped
    assert(slivers * 20 < groups.size());
    // 10 m diagonal: pieces of at most ~7 x 7 m, so at least 100 * 100 / 49 of them; the median
    // split keeps them balanced, not thousands of slivers.
    assert(groups.size() >= 204 && groups.size() <= 1024);
    // The triangle cap alone (no cell limit) and the floor on piece size.
    assert(SplitTriangles(tris.data(), count, 1e9f, 5000, 16, out.data()).size() == 4); // 20000 / 2^2 <= 5000
    assert(SplitTriangles(tris.data(), count, 0.1f, 1u << 30, 4096, out.data()).size() == 8);
    // Distance from a point to a box.
    Box b;
    const float lo[3] = {0, 0, 0}, hi[3] = {2, 1, 1};
    b.Add(lo);
    b.Add(hi);
    assert(b.Distance2({-3.0f, 0.5f, 0.5f}) == 9.0f && b.Distance2({1.0f, 0.5f, 0.5f}) == 0.0f);
}

static void SettingsParsing() {
    Settings s;
    bool reset = false;
    // One key per line, several per line (what the zsh-quoted helper once wrote), comments,
    // unknown keys and garbage.
    const u32 n = ApplyText(s, "rays=6\nnormals=0 snap=12 denoise=1\n# a comment rays=9\nbogus=3 far=x\nradius=2.5\n",
                            reset);
    assert(n == 5);
    assert(s.rays == 6 && !s.depth_normals && s.snap_far == 12.0f && s.denoise == 1 && s.radius == 2.5f);
    assert(reset); // the denoiser changed: no history across
    reset = false;
    ApplyText(s, "denoise=1", reset);
    assert(!reset);
    // Clamps.
    ApplyText(s, "rays=99 strength=3 history=1 debug=7 maxn=0 clipk=50 period=0 promote=1", reset);
    assert(s.rays == 32 && s.strength == 1.0f && s.history == 0.98f && s.debug == 4);
    assert(s.max_n == 1.0f && s.clip_k == 10.0f && s.noise_period == 1 && s.promote_after == 2);
    ApplyText(s, "rays=-3 stage=-1", reset);
    assert(s.rays == 1 && s.stage == 0);
    assert(!Apply(s, "nonsense", 1.0f));
}

int main() {
    Layouts();
    Keys();
    Matching();
    StateMachine();
    Split();
    SettingsParsing();
    std::puts("rtao-geometry-test: ok");
    return 0;
}
