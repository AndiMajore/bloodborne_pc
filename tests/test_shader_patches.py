# SPDX-License-Identifier: GPL-2.0-or-later
"""Shader packs (scripts/shader_patches.py): discovery, the profile, the copy at launch."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import shader_patches  # noqa: E402

NAME_A = "fs_0x0000000083da8467_0.spv"
NAME_B = "fs_0x00000000c45c51cb_0.spv"


class ShaderPatchesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "shaders"
        (self.root / "gtao").mkdir(parents=True)
        (self.root / "gtao" / NAME_A).write_bytes(b"A1")
        (self.root / "gtao" / "notes.txt").write_text("not a patch")
        (self.root / "upsample").mkdir()
        (self.root / "upsample" / NAME_A).write_bytes(b"A2")
        (self.root / "upsample" / NAME_B).write_bytes(b"B")
        (self.root / "empty").mkdir()
        (self.root / "bad").mkdir()
        (self.root / "bad" / "shader.spv").write_bytes(b"x")  # not named by hash: not a pack
        self.config = Path(self.tmp.name) / "shaders.json"
        self.manifest = Path(self.tmp.name) / "installed.json"
        self.patch_dir = Path(self.tmp.name) / "user" / "shader" / "patch"

    def tearDown(self):
        self.tmp.cleanup()

    def test_discover_only_folders_with_patch_files(self):
        self.assertEqual(shader_patches.discover(self.root), ["gtao", "upsample"])
        self.assertEqual(shader_patches.discover(self.root / "missing"), [])

    def test_apply_copies_enabled_packs_later_pack_wins(self):
        count = shader_patches.apply(self.root, self.config, self.patch_dir, self.manifest)
        self.assertEqual(count, 2)
        self.assertEqual((self.patch_dir / NAME_A).read_bytes(), b"A2")
        self.assertEqual((self.patch_dir / NAME_B).read_bytes(), b"B")
        self.assertEqual(json.loads(self.manifest.read_text())["files"], [NAME_A, NAME_B])

    def test_disabled_pack_removed_again_manual_files_kept(self):
        shader_patches.apply(self.root, self.config, self.patch_dir, self.manifest)
        (self.patch_dir / "fs_0x0000000000000001_0.spv").write_bytes(b"mine")
        self.config.write_text(json.dumps({"disabled": ["upsample"]}))
        count = shader_patches.apply(self.root, self.config, self.patch_dir, self.manifest)
        self.assertEqual(count, 1)
        self.assertEqual((self.patch_dir / NAME_A).read_bytes(), b"A1")
        self.assertFalse((self.patch_dir / NAME_B).exists())
        self.assertTrue((self.patch_dir / "fs_0x0000000000000001_0.spv").exists())
        self.config.write_text(json.dumps({"disabled": ["upsample", "gtao"]}))
        self.assertEqual(shader_patches.apply(self.root, self.config, self.patch_dir, self.manifest), 0)
        self.assertFalse((self.patch_dir / NAME_A).exists())


if __name__ == "__main__":
    unittest.main()


def module(bindings):
    """A minimal SPIR-V module declaring resources: {binding: "image" | "sampler" | "buffer"}."""
    words = [0x07230203, 0x00010300, 0, 1000, 0]
    ids = iter(range(1, 1000))
    image_t, sampler_t, float_t, struct_t = next(ids), next(ids), next(ids), next(ids)
    words += [(3 << 16) | 22, float_t, 32]                                     # OpTypeFloat
    words += [(9 << 16) | 25, image_t, float_t, 1, 0, 0, 0, 1, 0]              # OpTypeImage 2D
    words += [(2 << 16) | 26, sampler_t]                                       # OpTypeSampler
    words += [(3 << 16) | 30, struct_t, float_t]                               # OpTypeStruct
    pointee = {"image": image_t, "sampler": sampler_t, "buffer": struct_t}
    for binding, kind in bindings.items():
        pointer, var = next(ids), next(ids)
        words += [(4 << 16) | 32, pointer, 0, pointee[kind]]                   # OpTypePointer
        words += [(4 << 16) | 59, pointer, var, 0]                             # OpVariable
        words += [(4 << 16) | 71, var, 33, binding]                            # Binding
    return struct.pack(f"<{len(words)}I", *words)


OLD = {0: "buffer", 2: "buffer", 3: "image", 4: "image", 7: "sampler", 8: "sampler"}
NEW = {0: "buffer", 1: "buffer", 2: "buffer", 3: "buffer", 4: "image", 5: "image",
       8: "sampler", 9: "sampler"}


class FitToVersionTest(unittest.TestCase):
    def test_interface(self):
        self.assertEqual(shader_patches.interface(module(OLD)), OLD)
        self.assertIsNone(shader_patches.interface(b"not spirv at all..."))

    def test_moves_images_and_samplers(self):
        data, note = shader_patches.adapt(module(OLD), module(NEW))
        self.assertEqual(note, "image/sampler bindings moved by +1")
        self.assertEqual(shader_patches.interface(data),
                         {0: "buffer", 2: "buffer", 4: "image", 5: "image", 8: "sampler", 9: "sampler"})
        back, note = shader_patches.adapt(data, module(OLD))
        self.assertEqual(note, "image/sampler bindings moved by -1")
        self.assertEqual(shader_patches.interface(back), OLD)

    def test_same_version_untouched(self):
        data, note = shader_patches.adapt(module(OLD), module(OLD))
        self.assertEqual((data, note), (module(OLD), None))

    def test_mismatch_left_out(self):
        other = {0: "buffer", 3: "image", 4: "sampler", 5: "image"}
        data, note = shader_patches.adapt(module(OLD), module(other))
        self.assertIsNone(data)
        self.assertIn("do not match", note)

    def test_apply_fits_from_the_shader_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "packs" / "ao").mkdir(parents=True)
            (tmp / "packs" / "ao" / NAME_A).write_bytes(module(OLD))
            (tmp / "user" / "cache" / "CUSA03173").mkdir(parents=True)
            (tmp / "user" / "cache" / "CUSA03173" / NAME_A[3:]).write_bytes(module(NEW))
            notes = []
            patch = tmp / "user" / "shader" / "patch"
            self.assertEqual(shader_patches.apply(tmp / "packs", None, patch, None, notes), 1)
            self.assertEqual(shader_patches.interface((patch / NAME_A).read_bytes())[4], "image")
            self.assertEqual(notes, [f"ao/{NAME_A}: image/sampler bindings moved by +1"])
