# SPDX-License-Identifier: GPL-2.0-or-later
"""Shader packs (launcher: "Experimental shaders"): folders of replacement shaders, one file per
translated shader, named as the port's patch files: <stage>_0x<hash>_<perm>.spv (docs/DEBUGGING.md).
At launch the enabled packs are copied into <user>/shader/patch and BB_PATCH_SHADERS=1 is set.
Files put into the patch folder by hand are left alone: only what the launcher copied before is
removed again (a manifest beside the profile).
"""
import json
import re
import shutil
import struct
from pathlib import Path

PATCH_FILE = re.compile(r"^[a-z]{2}_0x[0-9a-f]{16}_[0-9]+\.spv$")


SPIRV_MAGIC = 0x07230203
# Opcodes and the one decoration this needs (SPIR-V 1.x).
OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE = 25, 26, 27
OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY, OP_TYPE_POINTER, OP_VARIABLE, OP_DECORATE = 28, 29, 32, 59, 71
DECORATION_BINDING = 33


def _words(data):
    if len(data) < 20 or len(data) % 4:
        return None
    words = list(struct.unpack(f"<{len(data) // 4}I", data))
    return words if words[0] == SPIRV_MAGIC else None


def interface(data):
    """The resource bindings of a SPIR-V module: {binding: kind}, kind "image", "sampler",
    "sampled_image" or "buffer" (anything else bound: uniform/storage blocks). None: not SPIR-V."""
    words = _words(data)
    if words is None:
        return None
    types, pointers, variables, bindings = {}, {}, {}, {}
    at = 5
    while at < len(words):
        count, op = words[at] >> 16, words[at] & 0xFFFF
        if count == 0:
            return None
        args = words[at + 1:at + count]
        if op in (OP_TYPE_IMAGE, OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE):
            types[args[0]] = {OP_TYPE_IMAGE: "image", OP_TYPE_SAMPLER: "sampler",
                              OP_TYPE_SAMPLED_IMAGE: "sampled_image"}[op]
        elif op in (OP_TYPE_ARRAY, OP_TYPE_RUNTIME_ARRAY):
            types[args[0]] = ("array", args[1])
        elif op == OP_TYPE_POINTER:
            pointers[args[0]] = args[2]
        elif op == OP_VARIABLE:
            variables[args[1]] = args[0]
        elif op == OP_DECORATE and len(args) >= 3 and args[1] == DECORATION_BINDING:
            bindings[args[0]] = args[2]
        at += count

    def kind(type_id):
        t = types.get(type_id, "buffer")
        while isinstance(t, tuple):
            t = types.get(t[1], "buffer")
        return t

    out = {}
    for var, binding in bindings.items():
        out[binding] = kind(pointers.get(variables.get(var), 0))
    return out


def adapt(data, reference):
    """A pack shader made to fit the port's own translation of the same shader (`reference`, the
    shader cache's .spv): (data, note). The ports differ in where they put images and samplers
    (0.6 adds a buffer of its own before them: one binding higher than on 0.51). Images and
    samplers shifted by one consistent amount are moved; buffers keep theirs. data None: the
    pack does not fit this version (left out)."""
    mine, theirs = interface(data), interface(reference)
    if mine is None:
        return None, "not a SPIR-V file"
    if theirs is None:
        return data, None  # nothing to compare with: as it is
    textures = {"image", "sampler", "sampled_image"}
    mine_tex = sorted(b for b, k in mine.items() if k in textures)
    theirs_tex = sorted(b for b, k in theirs.items() if k in textures)
    buffers_fit = all(theirs.get(b) == "buffer" for b, k in mine.items() if k == "buffer")
    if not mine_tex or not theirs_tex or not buffers_fit:
        fits = buffers_fit and all(theirs.get(b) == k for b, k in mine.items())
        return (data, None) if fits else (None, "its bindings do not match this version's shader")
    offset = theirs_tex[0] - mine_tex[0]
    if any(theirs.get(b + offset) != mine[b] for b in mine_tex):
        return None, "its image/sampler bindings do not match this version's shader"
    if offset == 0:
        return data, None
    words = _words(data)
    at = 5
    binding_slots = []
    while at < len(words):
        count, op = words[at] >> 16, words[at] & 0xFFFF
        if op == OP_DECORATE and count >= 4 and words[at + 2] == DECORATION_BINDING:
            binding_slots.append(at + 3)
        at += count
    for slot in binding_slots:
        if words[slot] in mine_tex:
            words[slot] += offset
    return struct.pack(f"<{len(words)}I", *words), f"image/sampler bindings moved by {offset:+d}"


def _reference(patch_dir, name):
    """The port's own translation of the shader a patch file replaces, from the shader cache
    (<user>/cache/<title>/0x<hash>_<perm>.spv), if this version translated it before."""
    user = Path(patch_dir).parent.parent
    for candidate in sorted(user.glob(f"cache/*/{name[3:]}")):
        try:
            return candidate.read_bytes()
        except OSError:
            continue
    return None


def pack_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and PATCH_FILE.match(p.name)),
                  key=lambda p: p.name)


def discover(root):
    """Named packs: folders of `root` holding at least one patch file."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if p.is_dir() and pack_files(p)),
                  key=lambda name: (name.casefold(), name))


def selected(root, config):
    """The enabled packs in load order (later packs win on the same shader)."""
    available = discover(root)
    try:
        profile = json.loads(Path(config).read_text(encoding="utf-8")) if config else {}
    except (OSError, ValueError):
        profile = {}
    disabled = set(profile.get("disabled", []))
    return [name for name in available if name not in disabled]


def apply(root, config, patch_dir, manifest=None, notes=None):
    """Copies the enabled packs' files into `patch_dir`; returns how many files are in place.
    `manifest` (a JSON file) remembers what was copied, so a pack switched off is removed again.
    Each file is fitted to this version's translation of its shader where the shader cache has
    one (adapt); `notes` (a list) gets a line per file changed or left out."""
    patch_dir = Path(patch_dir)
    manifest = Path(manifest) if manifest else None
    previous = []
    if manifest and manifest.is_file():
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8")).get("files", [])
        except (OSError, ValueError):
            previous = []
    for name in previous:
        if PATCH_FILE.match(name):
            (patch_dir / name).unlink(missing_ok=True)
    copied = []
    packs = selected(root, config)
    if packs:
        patch_dir.mkdir(parents=True, exist_ok=True)
        for name in packs:
            for source in pack_files(Path(root) / name):
                reference = _reference(patch_dir, source.name)
                if reference is None:
                    shutil.copyfile(source, patch_dir / source.name)
                else:
                    data, note = adapt(source.read_bytes(), reference)
                    if note is not None and notes is not None:
                        notes.append(f"{name}/{source.name}: {note}" +
                                     ("" if data is not None else " (left out)"))
                    if data is None:
                        continue
                    (patch_dir / source.name).write_bytes(data)
                if source.name not in copied:
                    copied.append(source.name)
    if manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"files": copied}, indent=2) + "\n", encoding="utf-8")
    return len(copied)
