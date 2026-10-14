# Ray-traced ambient occlusion (`BB_RTAO=1`)

Experimental. Replaces the game's screen-space AO with ray-traced AO: the G-buffer geometry the
game draws is captured each frame, built into Vulkan acceleration structures, and traced with
ray queries at half resolution. The result is written into the game's own AO mask, so the game's
lighting applies it exactly where it applied its SSAO. Everything is behind `BB_RTAO=1`; without
it the port is unchanged.

## Requirements

- `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations` (RADV on
  RDNA2 and later, NVIDIA, Intel Arc). Without them the feature logs why and stays off.
  `VK_KHR_ray_tracing_pipeline`, where present, runs the trace as a ray-tracing pipeline (faster
  on the RX 9070 XT); otherwise it runs as ray queries in a compute shader.
- Object motion vectors (an upscaler with motion vectors): the geometry comes from that capture.
  `BB_RTAO=1` makes the capture cover every vertex pipeline and doubles it to 8M vertices.
- The game's own SSAO pass must run: the "Disable SSAO" game patch (launcher, Game effects,
  SSAO unchecked) removes the mask RTAO writes into. The log says so once.

## Data flow per frame

1. G-buffer draws (five or more colour targets plus depth) store their post-vertex-shader clip
   positions in the object-motion capture. `Rasterizer` hands each to `RayTracedAO::AddGeometry`
   with its index buffer address, counts, topology, alpha-test and depth-write state, an identity
   hash and whether it is skinned. Strips, alpha-tested draws (foliage) and draws without depth
   write (decals) are left out.
2. The game's AO upsample pass (pixel shader `0xc45c51cb` into an R8 scene-size target) ends;
   `Record()` runs:
   - static cache: the probe results of four passes ago are read (once their scheduler tick is
     done) and the frame's draws sorted into cached instances, per-frame geometry and new cache
     builds (see below);
   - `rtao_expand.comp`: per-frame triangles into a view-space list (and world-space copies for
     cache builds); triangles entirely beyond `far` + `radius` become inactive;
   - `rtao_probe.comp`: world-space count, centroid, bounds and first vertex of probed draws;
   - BLAS builds (per-frame list, fast build; cache builds, fast trace), then a TLAS of the
     per-frame BLAS plus the cached instances placed with the view matrix;
   - the trace at half resolution (`rtao_trace_body.glsl`, as `rtao_trace.rgen` + `.rmiss` +
     `.rchit` or `rtao_trace.comp`): a geometric normal from four depth taps, optionally (`snap`) a
     short ray that snaps the origin onto the BVH surface (depth-biased draws sit centimetres
     off it), a Waechter-Binder ulp offset, then `rays` cosine-weighted rays over `radius`; temporal blend
     with a bilinear, depth-tested, clamped history (denoiser 1), or raw output (denoiser 2);
   - denoiser 2 (default): `rtao_temporal.comp` blends the raw AO with the reprojected history at
     1/N, N being the per-pixel count of accumulated frames (up to `maxn`, reset where the depth
     test fails), after clipping the history to the current 3x3 mean +- max(`clipk` sigma,
     `clipfloor`);
   - `rtao_blur.comp`: separable 7-tap depth-aware blur (`sigma`; with denoiser 2 twice as wide
     where N is 1, half as wide from N = 16);
   - `rtao_apply.comp`: depth-aware 4x4 upsample, `ao^power`, written as
     `min(game mask, lerp(1, ao, strength))` (or ours alone with `replace=1`).

### Static cache

A draw's identity is its shader, buffers, index range, counts and occurrence in the frame. A draw
whose world-space summary stays the same for `promote` observations gets its own BLAS (world
space, built for trace speed) and is instanced from then on; it leaves the per-frame BLAS and its
budget. Cached draws are rechecked every 8th pass and evicted when they change; three evictions
and a draw is never cached again. Copies of a mesh drawn several times a frame are matched by
content (the game sorts draws by distance) and leave the per-frame BLAS only as a group. Draws
with a character-size skeleton are never cached; draws bigger than `maxsize` stay per frame (whole
in a cached BLAS, their far triangles would come back into every ray's path). Least recently seen
BLASes go first beyond 1.5 GiB.

### Index buffers

The pass reads each draw's index buffer by device address when the game's AO pass ends, many draws
later. Small ranges (up to 16 KiB, streamed by the buffer cache through a ring that is reused once
the draw's tick is done) are copied from the game's memory at draw time into the RTAO's own ring.
With layer memory (`BB_LAYER_MEMORY=1`, default off on AMD) the buffer cache holds the mirrors it
retires during a frame until the pass is recorded. `BB_RTAO_TEST_MOVE=1` (with layer memory and
`BB_DRAW_PIPE=0`) forces every mirror back in place between the AO upsample and the pass: without
the hold (`BB_RTAO_HOLD=0`) that faults the GPU within a minute, with it it does not.

### Host rings

The geometry table, the instance list and the trace constants rotate over three slots, the probe
results over four. Each slot records the scheduler tick of the pass that used it; a slot is
reused only once the GPU has passed that tick, otherwise the pass skips the frame and the game's
AO stays. `ringtest=1` forces every pass onto one slot to exercise that path: at the Doll about
15% of passes skip, no fault.

## Settings

Environment at start (`BB_RTAO_*`), most also live in `<user>/rtao.txt` (`BB_RTAO_FILE`),
re-read within a second of a change. `~/.local/bin/rtao key=value` edits the file.

| Key | Env | Default | Meaning |
|---|---|---|---|
| `enabled` | | 1 | 0: the pass is not recorded (the capture stays on) |
| `strength` | `BB_RTAO_STRENGTH` | 1.0 | how much of the AO enters the mask |
| `power` | `BB_RTAO_POWER` | 2.0 | `ao^power` (1 = physically plain, 2 = the chosen look) |
| `radius` | `BB_RTAO_RADIUS` | 1.5 | ray length in metres |
| `rays` | `BB_RTAO_RAYS` | 2 | AO rays per half-res pixel (denoiser 1 wants 6) |
| `denoise` | `BB_RTAO_DENOISE` | 2 | 2: accumulated frame count + variance clipping; 1: fixed history weight |
| `maxn` | | 32 | denoiser 2: accumulated frames at most |
| `clipk`, `clipfloor` | | 2, 0.1 | denoiser 2: history clip to mean +- max(clipk sigma, clipfloor) |
| `subpixel` | | 0 | denoiser 2: 1 cycles the sampled pixel through the 2x2 block (thin features; less steady) |
| `falloff` | | 0 | 1: binary hits within a random length, linear to 0 at the radius (lighter) |
| `period` | | 8 | frames before a pixel's ray pattern repeats |
| `history` | `BB_RTAO_HISTORY` | 0.5 | denoiser 1: temporal weight (0.85 lagged visibly in motion) |
| `sigma` | `BB_RTAO_SIGMA` | 1.5 | half-res blur width in texels, 0 off |
| `blur` | | 1 | 4x4 depth-aware upsample, 0: 2x2 |
| `replace` | | 0 | 1: ignore the game's AO (or a shader pack's) in the mask |
| `far` | | 40 | no AO and no per-frame geometry beyond this distance (fades over the last quarter) |
| `snap` | | 0 | the surface snap runs up to this distance (0 off; no measurable effect with depth normals) |
| `normals` | | 1 | 1: geometric normal from the depth buffer, 0: the G-buffer's shading normal |
| `offset` | | 0 | metres along the normal on top of the ulp offset |
| `pipeline` | `BB_RTAO_PIPELINE` | 1 | the trace as a ray-tracing pipeline (where supported), 0: ray queries |
| | `BB_RTAO_DATA_ACCESS` | 1 | ALLOW_DATA_ACCESS on the per-frame BLAS (see measurements) |
| `cache` | `BB_RTAO_CACHE` | 1 | static cache on/off |
| `promote` | | 8 | matching observations before a draw is cached |
| `maxsize` | | 20 | bigger draws stay per frame (metres, bounding-box diagonal) |
| `build` | | 0 | 1: per-frame BLAS built for trace speed (no difference on RADV) |
| `alpha`, `decals` | | 0 | 1: keep alpha-tested draws / draws without depth write |
| | `BB_RTAO_MAX_TRIANGLES` | 3000000 | per-frame triangle budget (buffers are sized for it) |
| `stage` | `BB_RTAO_STAGE` | 5 | stop after: 1 expand, 2 BLAS, 3 TLAS, 4 trace, 5 apply |
| `debug` | `BB_RTAO_DEBUG` | 0 | 1 raw AO, 2 camera-ray test, 3 normal facing, 4 AO with t_min 0.1 |
| `dump` | | | change the value: 8 consecutive frames of the AO the apply gets go to `out/rtao_dump_<frame>_<k>.pgm`; the log line gives mean, mean frame-to-frame change and the share of pixels changing > 0.05 |
| `ringtest` | | 0 | 1: every pass on ring slot 0 (tests the busy-slot skip) |

## Measurements (RX 9070 XT, RADV, 2560x1440 render, FSR to 4K)

Measure with the camera still, `BB_GPU_PROFILE=1` (per-stage segments `rtao expand`, `rtao blas`,
`rtao tlas`, `rtao trace`, `rtao blur`, `rtao apply`) and the frame cap off (`BB_FPS_LIMIT=0`,
`BB_PRESENT_MODE=Mailbox`). The game paces itself to the 60 Hz vblank even then, so judge "GPU
busy" (frame time minus the profiler's "GPU idle" segment), not FPS. Session-to-session noise is
about 0.15 ms; A/B within one session with the live settings where possible.

| Spot | RT off GPU busy | RT on GPU busy | RT stages (expand + BLAS + trace + rest) |
|---|---|---|---|
| Hunter's Dream, the Doll (2026-10-14) | 13.4-13.7 ms | 16.6-16.8 ms | 3.9-4.1 ms (0.8 + 1.6 + 1.6 + ...) |
| Hunter's Dream start view, cache off / on (maxsize 20) | | | 3.08 / 2.90 ms |
| Looking straight down at cobblestones | | | trace alone 5.6 ms (before the changes below) |

Decisions measured at the Doll (A/B in one session unless noted):

| Change | Before | After |
|---|---|---|
| Trace as ray-tracing pipeline instead of ray queries in compute | trace 1.62-1.64 ms | 1.40-1.44 ms |
| Trace workgroups 8x4 instead of 8x8 (compute variant) | 1.78-1.81 ms | 1.77-1.78 ms (no effect, kept) |
| Per-frame BLAS without ALLOW_DATA_ACCESS (RDNA4 compressed pairs; three sessions A/B/A) | build 1.65, trace 1.79 ms | build 2.41, trace 1.73 ms: kept the flag |
| Cached BLASes compressed (no flag), cache on vs off | RT 4.53 ms | 4.20 ms |
| Depth-derived normals vs shading normals, self-hit share (pixels > 0.15 darker than with rays from 10 cm out; the snap was on in all four runs, a settings bug) | shading 6.4% / 9.1% | depth 4.4% / 6.2% |
| Snap ray on vs off (6 rays / 1 ray) | trace 1.43 / 0.27 ms | 1.55 / 0.50 ms |
| Trace vs ray count (pipeline, snap on) | 1 ray 0.50, 2 rays 0.70 ms | 6 rays 1.55, 12 rays 2.62 ms |
| Snap distance at 2 rays, Doll / looking down at the paving (pixels > 0.15 darker than snap 30 at 16 rays, 8-frame means; noise snap 30 vs 30: 0.85% / 0.55%) | snap 30: trace 0.71-0.75 / 0.70-0.78 ms | snap 0: 0.50 / 0.50 ms, darker 0.89% / 0.33%; 4 m: 0.55 / 0.54 ms, 0.84% / 0.20%: default 0 |
| Denoiser 2 at 2 rays vs denoiser 1 at 6 (still camera, 8 frames) | GPU busy 16.63 ms, 4.40% of pixels change > 0.05, mean 0.787 | 15.68 ms, 2.78%, mean 0.786 |
| Denoiser 2 clip floor (2 rays) | 0: mean 0.803 (biased light: all-miss 3x3 neighbourhoods clipped history to 1) | 0.1: 0.787 (raw trace mean 0.791) |
| Static-cache probe in chunks of 4096 indices | probe 0.61 ms | < 0.1 ms |
| Defaults at the end of 2026-10-14 vs RT off | GPU busy 13.37 ms | 15.16 ms (60 FPS; BLAS 1.61, trace 0.69, expand 0.18 ms) |

Known cost drivers: the per-frame BLAS (1.73M triangles at the Doll: 928k from static meshes
larger than `maxsize`, kept per frame for the per-triangle far cut, and 803k skinned) is rebuilt
from a non-indexed expansion every frame; every pixel near the camera is traced, so looking at
the ground is the worst case; instances that overlap the per-frame BLAS make nearby rays
traverse two trees.

## Memory

Fixed buffers at the default 3M-triangle budget (RX 9070 XT, RADV):

| What | MiB |
|---|---|
| Per-frame BLAS (3M triangles, ~200 B per triangle on RADV; same for fast build and fast trace) | 586 |
| Its build scratch | 400 |
| Triangle list (3M per-frame + 0.4M cache builds, 36 B each) | 117 |
| Cache build scratch, TLAS, rings, probe buffers | ~93 |
| Total RTAO fixed buffers | 1196 |
| Object-motion capture with `BB_RTAO=1` (8M positions x 2 frames) | 256 |

The static cache comes on top: at most `BB_RTAO_CACHE_MB` (1536) and at most a quarter of the
device heap's remaining room (VMA's budget minus usage), re-checked every 10 seconds. A failed
cache allocation halves the budget, trims the least recently seen BLASes and pauses cache builds
for 10 seconds; the 10-second report says so (no per-allocation messages).
`BB_RTAO_FAIL_ALLOC=N` makes every Nth cache allocation fail, to test that path. Compaction is not
used: on RADV it saves nothing.

## Debugging

- **GPU hangs**: breadcrumbs name the stage (`STUCK: pass rtao <stage>`); `stage=N` stops after a
  stage to bisect. With `BB_FAULT_WAIT=1` the process stays alive for `tools/debug_attach.sh`.
- **Geometry**: `debug=2` traces one camera ray per pixel and writes `t/|P|` (white where the BVH
  sits where the depth buffer says); `dump=` writes the image, `1.0 = 128`. The 600-frame log line
  reports draws, triangles, budget drops, skipped draw kinds, passes recorded and skipped; the
  first pass after start reads back expanded triangles (valid share, depth range).
- **Static cache**: the `RTAO: cache:` line (known draws, instanced, MiB, built, evicted, probe
  matches/changes).
- **Validation**: `BB_VK_VALIDATION=1` (core), `BB_VK_VALIDATION_SYNC=1` (synchronization),
  `BB_VK_VALIDATION_GPU=1` (GPU-assisted, about 25 FPS). RTAO objects carry debug names (`rtao
  trace`, `rtao triangles`, `rtao TLAS`, ...), so a message about them names them. To separate
  RTAO from the rest of the port, run once with `enabled=0` in `rtao.txt` from the start and
  compare the sets of message IDs. The layer reports each ID a limited number of times.
  State of 2026-10-14 at the Doll: no message names an RTAO object and GPU-AV found nothing in 453
  passes. Expected noise from the port: the motion capture's fragment input at location 31 (over
  128 components) and its unbound attachment 6 (more frequent with `BB_RTAO=1`, which captures
  every pipeline), a tessellation interface mismatch, FSR 4 cooperative matrices, a 4 GiB buffer,
  an sRGB storage view, present and render-pass-load hazards.

## Known limitations

- Foliage and other alpha-tested draws cast no AO (they would be solid quads); decals are left out.
- Strips and other non-list topologies are skipped (counted in the log).
- The capture's positions are post-vertex-shader: anything the game moves in a later stage is not
  seen.
- With the game's SSAO patched out there is nothing to write into.
- Draws of more than `maxsize` are rebuilt every frame (the cache would keep their far triangles).
