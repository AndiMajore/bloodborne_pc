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
     `.rchit` or `rtao_trace.comp`): a geometric normal from four depth taps, a short ray that snaps
     the origin onto the BVH surface (depth-biased draws sit centimetres off it), a
     Waechter-Binder ulp offset, then `rays` cosine-weighted rays over `radius`; temporal blend
     with a bilinear, depth-tested, clamped history;
   - `rtao_blur.comp`: separable 7-tap depth-aware blur (`sigma`);
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
| `rays` | `BB_RTAO_RAYS` | 6 | AO rays per half-res pixel |
| `history` | `BB_RTAO_HISTORY` | 0.5 | temporal weight (0.85 lagged visibly in motion) |
| `sigma` | `BB_RTAO_SIGMA` | 1.5 | half-res blur width in texels, 0 off |
| `blur` | | 1 | 4x4 depth-aware upsample, 0: 2x2 |
| `replace` | | 0 | 1: ignore the game's AO (or a shader pack's) in the mask |
| `far` | | 40 | no AO and no per-frame geometry beyond this distance (fades over the last quarter) |
| `snap` | | 30 | the surface snap runs up to this distance (0 off) |
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
| `dump` | | | change the value: the half-res AO image goes to `out/rtao_dump_<frame>.pgm` |
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
| Depth-derived normals + snap vs shading normals, self-hit share | 9.1% (shading, no snap) | 4.4% |

Known cost drivers: trace cost hardly depends on the ray count (it is traversal-bound: rays
start in dense geometry and misses walk the whole local BVH); every pixel near the camera is
traced, so looking at the ground is the worst case; the per-frame BLAS is rebuilt from a
non-indexed expansion every frame.

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
