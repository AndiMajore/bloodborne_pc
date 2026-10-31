**English** · [Русский](RAYTRACING.ru.md)

# Ray tracing (experimental)

bbport can replace part of the game's screen-space lighting with ray tracing through Vulkan. This
is a first, experimental version: it is off by default, it costs frame time, and it has been
measured on one graphics card only.

## What it does

The **Lite** preset, the one we recommend, turns on:

- **Ray-traced ambient occlusion (RTAO)** in place of the game's SSAO: soft shadows in corners,
  under objects and where things touch, including from geometry that is off screen. It is written
  into the game's own AO, so the game's lighting uses it the way it used its SSAO.
- **The moon's shadow by ray tracing**, at half resolution, wherever a ray finds an occluder. Where
  it finds none, the game's own shadow stays.
- **Adaptive quality**: when the GPU nears the frame limit (16.5 ms), the ray-traced effects step
  down, and they come back when there is room.

On an RX 9070 XT at 2560x1440 rendering with FSR to 4K, Lite costs about **1.7 ms per frame** and
holds 60 fps in most areas. In the heaviest parts of Central Yharnam the game alone uses most of
the frame, so frames can still go over.

## Requirements

- **Linux** with the Mesa RADV driver. Windows is not supported yet: its launcher has no ray-tracing
  settings.
- **A GPU with Vulkan ray tracing.** Tested on an **AMD Radeon RX 9070 XT** (RDNA4) only.
  - RDNA3 (RX 7000) should work but is untested; expect 1.5 to 2 times the cost.
  - RDNA2 (RX 6000) is slower: 2 to 4 ms per frame.
  - NVIDIA RTX, Intel Arc and AMD's other drivers are untested.
  - The launcher checks the GPU you chose and says so under the switch.
- **An upscaler with object motion vectors** (FSR, DLSS or TAA, with *Object motion vectors* on).
- **The game's SSAO on** (*Game effects*). Ray-traced AO writes into the game's AO; with SSAO
  patched out there is nothing to write into. Mods that remove the SSAO pass (bb_enhanced) have the
  same effect. The unmodded game and the usual mods (bloodborne_remaster, the visual upgrade mod)
  work.
- **Not together with GT-VBAO**: both replace the game's AO. The launcher turns one off when you
  turn the other on.
- Video memory: about 2 GB on a 16 GB card. On smaller cards the buffers shrink, and in crowded
  fights distant characters can drop out of the AO.

## Turning it on

Launcher, *Graphics*, group *Ray tracing (experimental)*:

1. Switch on **Ray-traced ambient occlusion (RTAO)**. If the switch is greyed out, its subtitle says
   what is missing.
2. **Ray-tracing preset: Lite.** *Quality* (Low, Medium, High) sets the AO's rays, radius and
   distance; Medium is the default.
3. Play. The *Log* page shows a line such as
   `Ray tracing: RTAO on (600 of 600 frames); moon shadow on (600 of 600 frames)` every few
   seconds. If an effect does not run, it says **OFF** and why, and the launcher shows a notice.

Changing any switch below the preset makes it *Custom*. Choose *Lite* again to go back.

## More effects (not judged yet)

The group *More ray-traced effects (not judged yet)* holds effects that work but have not been
checked in play. They may flicker, lag or cost several milliseconds:

- **Ray-traced bounce light**: the SSGI finds lit surfaces with rays instead of on screen (turns
  SSGI on). Fine bright detail can sparkle with FSR 4.
- **Ray-traced reflections**: on top of the game's screen-space reflections (turns them on); 1 to
  2 ms more.
- **Bounce light from probes**: a grid of probes around the camera, instead of the ray-traced
  bounce. Light can leak through thin walls.

## Known limits

- The AO darkens everywhere, also where the game itself shows no AO (the Workshop).
- Grass and leaves cast no AO at Low and Medium; at High they occlude by chance per pixel and frame,
  which averages out to their coverage.
- The moon's shadow from characters can slide across a wall while only the camera turns.

## Reporting a problem

Send the log (*Log*, *Export log…*) with the place in the game, your GPU and driver, and the last
*Ray tracing:* line. Questions go to the project's Discord, not to shadPS4.

Developer documentation: [rtao.md](rtao.md) (the AO) and [rt-lighting.md](rt-lighting.md) (the other
effects, all settings).
