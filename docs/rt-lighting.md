# Experimental ray-traced lighting (on the RTAO's scene)

Effects that reuse the acceleration structures the ray-traced AO builds every frame
(`docs/rtao.md`). All experimental and **all off by default**; they need the RTAO on a GPU with
Vulkan ray queries. Prototypes on the branch `rt-lighting-exp`: none of them is judged yet, and
none is in `fork-0.6` (ticket 65).

The RTAO exports its scene and nothing else: `RayTracedAO::FrameTlas` (the TLAS of this frame in
view space, valid after the AO pass until the next frame's) and `FrameSurface` (the surface
cache). The reflections and the probes are classes of their own; the effects that hook the game's
draws (the light list, the shadows, the transparents' light) sit behind `RtLighting`
(`gpu/render/vk_rt_lighting.h`), the only one the rasterizer calls. Its hooks are gated: with the
RTAO off, or no effect wanting them, the rasterizer runs as without them (a flag test).

## The effects

| Effect | Switch (default) | What it replaces | How |
|---|---|---|---|
| Ray-traced bounce light | `rt=1` in `ssgi.txt` (0); launcher: "Ray-traced bounce light" | SSGI's screen-space march | cosine-weighted rays (`rays=`, 2 per half-res pixel, SSGI's `radius` long) against the TLAS; a hit the camera sees adds the lit colour at that pixel, a hidden hit the surface cache's radiance (or nothing); history clipped to the 3x3 neighbourhood, per-ray radiance capped |
| Ray-traced reflections | `BB_RT_REFLECT=1` (off), `rtreflect.txt` | the game's SSR result | after the game's SSR pass (ps `6c2323c2`, 1280x720 RGBA16F: colour x weight, weight in alpha), one ray per texel, spread over the material's specular lobe (`soften`, from the G-buffer's gloss); a hit is shaded from last frame's lit image at its reprojected position, or from the surface cache; the game builds its blurred SSR levels from our result and applies the material's specular. `replace_ssr=1` (default with the switch on) leaves the game's SSR draw out and writes the whole target |
| Probe GI | `BB_RT_PROBES=1` (off), `rtprobes.txt` | SSGI (use one of them) | a 16x8x16 grid of probes around the camera (`spacing` 3 m), 64 rays per probe a frame, L1 spherical harmonics with hysteresis; per pixel the eight surrounding probes blended. No depth maps yet: thin walls can leak |
| Surface cache | `surface=1` in `rtao.txt` (0) | (feeds the effects above) | radiance per static pool triangle (8 bytes, ~23 MiB at the full pool), refreshed from the lit screen in slices; a ray hit on the static instance reads it when the hit is not on screen |
| The transparents' light (emission, 63) | `emission=` in `rtlighting.txt` (0) | (feeds SSGI, probes, reflections) | fires, sparks and glows are blended after the opaque pass, so the effects never saw them. The opaque scene is kept at a quarter of its size at the first transparent draw; at the end of the transparents (the first draw into another target or without the scene depth: before the fog composite) the difference is their light; grey additions (smoke) dropped, the rest blurred. Read the frame after at a hit's screen position. Smoke drifting in front of a far surface makes that surface glow: the fire lights below replace it |
| Fire lights (63) | `fire_light=` in `rtlighting.txt` (0); needs SSGI `rt=1` | (adds to the RT bounce light) | from the emission, per 64x64 tile the brightest texel at the nearest depth under the flame, its power the tile's emission x its texels' world area, matched with last frame's lights by their place in the world and smoothed. Each light an upright cylinder (`fire_size`, `fire_height`) lit from and traced to a random point of it each frame (an area light the history averages), the cosine wrapped (`fire_wrap`); per half-res pixel one light picked by its unshadowed light per ray, one shadow ray. Off-screen fires give no light |
| Ray-traced shadows (64) | `sun_shadow=`, `spot_shadow=` in `rtlighting.txt` (0) | the game's sun shadow in its screen mask; the shadowed spot's 512x512 shadow map | sun: before the global pass and the sun's light read the mask, one ray per pixel towards the sun (its direction from the shadow resolve's constants) writes the shadow in the mask's region channel with the game's scale; where a ray finds nothing in the RT scene the game's shadow stays, shrunk by `sun_erode`. Spot: from the light list's record of the shadowed spot (the frame before), a `spot_res` map ray traced from the light (depth and its square as the game's variance map stores them), blurred (`spot_blur`) with the variance capped (`spot_soft`), swapped in for the game's at bind time. The two most invasive hooks on the branch (a game target written; a bound texture replaced by address) |
| The game's light list (59) | (a consumer: the spot shadow, `lights_dump`) | (none) | the game's light passes draw screen tiles per pixel-shader variant, each binding a buffer of light records (`gpu/render/rt_light_records.h`); decoded once a frame per buffer into world-space points and spots. Not used for lighting yet |

## Live settings (re-read while the game runs)

- `user/ssgi.txt` (SSGI): `rt` (0/1), `rays` (1-8), `rt_clip` (1.5: the history clipped to the
  3x3 mean +- that many sigma), `rt_max` (2: a ray's radiance cap), `debug` (1 the bounce alone;
  2 hit classes: green seen, blue surface cache, red hidden; 3 the emission image; 4 the fire
  lights alone, blue what `fire_shadow` takes away).
  The colour denoiser (tickets 36, 37, 42; off): `rt_denoise` (0/1: instead of the gather's own
  clipped history, the rays of the frame accumulated on the motion vectors where depth and normal
  agree, up to `denoise_frames` (16), the history clipped to the raw 3x3 mean +- `denoise_clip`
  sigma (2; 0 off), then `denoise_passes` (3) edge-avoiding a-trous passes weighted by the plane
  distance (`denoise_plane` 0.02 of the depth), the normals (`denoise_normal` 32) and the
  luminance in standard deviations of its variance (`denoise_luma` 4)).
- `user/rtreflect.txt` (reflections): `enabled`, `mode` (1 fill SSR's misses, 2 ours wherever
  valid), `replace_ssr` (1: the game's SSR draw left out, ~1.5 ms), `soften` (1: the material's
  lobe; 0 a mirror; 2-3 blurrier), `far` (40 m), `strength` (0-1), `debug` (1 ours alone; 2
  classes, green seen, cyan surface cache, red hidden, blue miss, grey SSR kept).
- `user/rtprobes.txt` (probe GI): `enabled`, `strength` (4), `spacing` (m), `rays` (1-64),
  `hysteresis`, `sky`, `bias` (m), `debug` (1 the irradiance alone; 2 coverage).
- `user/rtao.txt` (the AO; of the experiments only): `surface` (0/1), `surface_budget`.
- `user/rtlighting.txt` (`BB_RTLIGHTING_FILE`; everything below off by default):
  - transparents' light: `emission` (0-16), `emission_sat` (0.35; 0 all), `emission_blur` (8
    quarter-size texels), `emission_at` (1; 0 the first full-screen pass, never hit in
    Bloodborne), `emission_trace` (a new value logs one frame's draws after the opaque scene).
  - fire lights: `fire_light` (0-64), `fire_shadow` (0-1: the unshadowed light taken away again),
    `fire_range` (12 m), `fire_size` (0.75 m: the cylinder's radius), `fire_height` (1 m, a quarter
    below the light's point), `fire_wrap` (0.4), `fire_smooth` (0.9: last frame's share of the
    power), `fire_smooth_pos` (0.5: of the place), `fire_pull` (0.3 m), `fire_min` (0.3: a tile's
    brightest texel needed; a light goes out below half of it), `fire_debug` (1: the cylinders
    drawn into the scene, the light's point a yellow dot), `fire_switch` (0; e.g. 0.5: a light
    stays where it is while its place is in its tile, unless the tile's new brightest texel is that
    much brighter than the flame there, or the flame there went dark) and `fire_hold` (0; frames a
    light stays after it moved): against the flicker of a light chasing the brightest texel.
  - two-part fires (67; off): `fire_base` (0; e.g. 0.7: each fire's steady base, the lights within
    `fire_base_merge` (1.5 m, horizontally) merged, their summed power and place averaged
    (`fire_base_smooth` 0.98 a frame), x this, as a slim upright cylinder `fire_base_size` (0.3 m)
    by `fire_base_height` (2.5 m, 40 % below); a fire gone from the screen fades by
    `fire_base_fade` (0.985) a frame), `fire_spark` (0.5: the per-tile lights' power, now spheres of
    `fire_spark_size` 0.3 m). `fire_debug` draws the bases cyan.
  - shadows: `sun_shadow` (0/1), `sun_spread` (0.01 rad), `sun_erode` (4 px), `sun_far` (200 m),
    `sun_debug` (1 rays only, 2 reversed, 4 the mask in full shadow: a write test);
    `spot_shadow` (0/1), `spot_res` (1024), `spot_blur` (3 texels), `spot_soft` (0.2 m),
    `spot_near` (0.3 m).
  - light list: `lights_dump` (a new value logs the next frame's list; every 600 frames a summary).

## Measured (RX 9070 XT, RADV, 0.6, FSR 4.1.1 2560x1440 -> 4K)

0.6-pre1 itself is 1.1-1.4 ms slower than 0.51 at the Doll (ticket 53, a compute shader's outputs
kept in system RAM): none of these numbers compares with `docs/rtao.md`'s 0.51 ones without that.

The Doll, everything else on (2026-10-17):

| | GPU busy |
|---|---|
| SSGI off / march / rays 2 / rays 4 | 14.87 / 15.44 / 15.97 / 16.02 ms |
| reflections off / mode 1 / mode 2 | 15.66 / 15.80 / 16.16 ms |
| surface cache | +0.05 ms |
| probe GI (instead of SSGI): nothing / SSGI / probes | 14.88 / 15.60 / 15.77 ms |
| everything on | 16.44 ms (60 fps) |

Central Yharnam, the burning stakes (a frozen bench save; 2026-10-18; GPU busy = frame minus the
profile's GPU-idle labels): the launcher's default preset (RTAO medium, SSGI march) 15.78 ms;
everything on with Andreas's settings 16.35-16.50 ms, at the vblank. The game's SSR left out
(`replace_ssr`): -1.5 ms. The sun and spot shadows with the plain RTAO: +0.67 ms.

## Known limits

- Characters (the per-frame BLAS) are never in the surface cache: a hidden hit on one stays dark.
- Surfaces the camera never saw stay dark; the cache bakes view-dependent highlights.
- The colour denoiser (`rt_denoise`, 36/37/42) is a first version for the RT SSGI only and off; the
  reflections still rely on the game's SSR blur levels. Not yet judged in motion.
- Reach limited to the static BLAS's (the AO's distance + margin): far occluders of the sun are the
  game's (cascaded) shadow.
- The SSR target's semantics were read from the game's shader (not shipped): colour x weight with
  the weight in alpha; texels SSR does not trace (towards the camera) are left as they are, or
  empty with `replace_ssr`.
- Push constants: the reflections need 256 bytes (RADV and NVIDIA have 256), checked at start.
- Tests: `rt-lighting-test` (the light records of a captured frame, the emission's trigger, the
  settings) and `rtao-geometry-test`; not covered: the SSR texel semantics.
