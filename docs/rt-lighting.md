# Experimental ray-traced lighting (on the RTAO's scene)

Three effects that reuse the acceleration structures the ray-traced AO builds every frame
(`docs/rtao.md`). All experimental, off by default, and only available with the RTAO on a GPU
with Vulkan ray queries. In the launcher: Settings, "Ray tracing (experimental)", under the RTAO
switch.

## What they do

| Effect | Switch | What it replaces | How |
|---|---|---|---|
| Ray-traced bounce light | `rt=1` in `user/ssgi.txt` | SSGI's screen-space march | cosine-weighted rays (`rays=`, default 2 per half-res pixel, as long as SSGI's `radius`) against the RTAO's TLAS; a hit the camera sees adds the lit colour at that pixel, a hidden hit the surface cache's radiance (or nothing) |
| Ray-traced reflections | `BB_RT_REFLECT=1`, `user/rtreflect.txt` | the game's SSR result (where chosen) | after the game's SSR pass (ps `6c2323c2`, 1280x720 RGBA16F: colour x weight, weight in alpha), one mirror ray per texel; a hit is shaded from last frame's lit image at its reprojected position, or from the surface cache; the game's lighting pass applies the material's specular as it does to SSR. `mode=1` fills only SSR's misses, `mode=2` writes ours wherever the ray gives a value |
| Probe GI | `BB_RT_PROBES=1`, `user/rtprobes.txt` | SSGI (use one of them) | a 16x8x16 grid of probes around the camera (`spacing`, 3 m), 64 rays per probe per frame, radiance as L1 spherical harmonics with a hysteresis; per pixel the eight surrounding probes blended (trilinear, facing weight), albedo x irradiance added to the lit scene. Probes inside geometry are switched off; no depth maps yet, so thin walls can leak |
| Surface cache | `surface=1` in `user/rtao.txt` | (nothing; feeds the two above) | radiance per static pool triangle (8 bytes, ~23 MiB at the full pool), refreshed from the lit screen in slices (`surface_budget=`, 131072 triangles a frame); a ray hit on the static instance reads it when the hit itself is not on screen |

The RTAO exposes its scene to other passes (`RayTracedAO::FrameTlas`, `FrameSurface`): the TLAS of
this frame in view space, valid after the AO pass and until the next frame's.

## Live settings

- `user/ssgi.txt`: `rt` (0/1), `rays` (1-8), `debug` (1: the bounce alone; 2: classes, green = hit
  seen on screen, blue = from the surface cache, red = hidden, nothing).
- `user/rtreflect.txt`: `enabled`, `mode` (1/2), `far` (metres, default 40; the static scene only
  reaches the AO's distance plus its margin), `strength` (the weight written, 0-1), `debug` (1:
  ours alone, 2: classes, green = seen, cyan = surface cache, red = hidden, blue = miss, grey = left
  as SSR made it).
- `user/rtao.txt`: `surface` (0/1), `surface_budget`.
- `user/rtprobes.txt`: `enabled`, `strength` (default 4), `spacing` (metres), `rays` (1-64),
  `hysteresis`, `sky` (radiance of a miss), `bias` (metres along the normal), `debug` (1: the
  irradiance alone; 2: probe coverage, black = no probe reaches the pixel).

## Measured (RX 9070 XT, RADV, 0.6, FSR 4.1.1 2560x1440 -> 4K, the Doll, everything else on)

| | GPU busy |
|---|---|
| SSGI off / march / rays 2 / rays 4 | 14.87 / 15.44 / 15.97 / 16.02 ms |
| reflections off / mode 1 / mode 2 | 15.66 / 15.80 / 16.16 ms |
| surface cache | +0.05 ms |
| probe GI (instead of SSGI): nothing / SSGI / probes | 14.88 / 15.60 / 15.77 ms |
| everything on | 16.44 ms (60 fps) |

## Known limits

- Characters (the per-frame BLAS) are never in the surface cache: a hidden hit on a character
  stays dark.
- Surfaces the camera never saw stay dark; the cache bakes view-dependent highlights; lighting
  changes (the Dream's phases, flicker) catch up slice by slice.
- Reflections use mirror rays only (no roughness yet: the material targets' decoding is ticket
  58); no denoiser of their own; reach limited to the static BLAS's (AO distance + margin).
- The SSR target's semantics were read from the game's shader (not shipped): colour x weight with
  the weight in alpha for traced pixels; pixels SSR does not trace (reflections towards the
  camera) are left as they are.
- Push constants: the reflections need 248 bytes (RADV and NVIDIA have 256; the 128-byte minimum
  of the spec is not enough), checked at start.
