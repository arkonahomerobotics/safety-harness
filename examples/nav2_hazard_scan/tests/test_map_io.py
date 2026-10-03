"""map_io.py against the bundled real warehouse map and small hand-built malformed inputs --
every validation path must raise MapLoadError with a clear message, never a bare traceback (the
CLI's whole "clear errors on malformed input" contract depends on that).

Run with: python3 -m unittest discover -s examples/nav2_hazard_scan/tests -v
(from the repo root, with numpy/pillow/pyyaml installed -- see the example README)
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from map_io import FREE, OCCUPIED, MapLoadError, load_occupancy_grid  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED_MAP = os.path.join(_HERE, "..", "maps", "warehouse.yaml")


def _write_map(tmpdir, *, image=None, yaml_overrides=None, omit_keys=(), raw_yaml_text=None, image_name="m.pgm"):
    """Writes a small synthetic map (<tmpdir>/m.yaml + image_name) and returns the yaml path.
    image: 2D uint8 array (default: a tiny 4x4 checkerboard-ish grid with clear free/occupied/mid
    pixels). yaml_overrides: dict merged over the default metadata. omit_keys: keys to delete
    from the metadata before writing. raw_yaml_text: if given, write this verbatim instead of
    building one, for testing non-dict/invalid-YAML files directly."""
    yaml_path = os.path.join(tmpdir, "m.yaml")
    if raw_yaml_text is not None:
        with open(yaml_path, "w") as f:
            f.write(raw_yaml_text)
        return yaml_path

    if image is None:
        image = np.array([
            [255, 255, 128, 0],
            [255, 255, 128, 0],
            [128, 128, 128, 0],
            [0, 0, 0, 0],
        ], dtype=np.uint8)
    image_path = os.path.join(tmpdir, image_name)
    Image.fromarray(image, mode="L").save(image_path)

    meta = {
        "image": image_name,
        "resolution": 0.05,
        "origin": [-1.0, -2.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
    }
    if yaml_overrides:
        meta.update(yaml_overrides)
    for k in omit_keys:
        meta.pop(k, None)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(meta, f)
    return yaml_path


class LoadBundledMapTest(unittest.TestCase):
    def test_bundled_warehouse_map_loads(self):
        grid = load_occupancy_grid(BUNDLED_MAP)
        self.assertEqual(grid.cells.shape, (1674, 1006))
        self.assertAlmostEqual(grid.resolution_m, 0.03)
        self.assertEqual(len(grid.origin_xy), 2)


class MalformedMapTest(unittest.TestCase):
    def test_missing_yaml_file(self):
        with self.assertRaises(MapLoadError):
            load_occupancy_grid("/definitely/does/not/exist.yaml")

    def test_not_valid_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, raw_yaml_text="image: [this is: not, valid")
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_yaml_not_a_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, raw_yaml_text="- just\n- a\n- list\n")
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_missing_required_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, omit_keys=["resolution"])
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_missing_image_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"image": "nope.pgm"})
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_non_numeric_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"resolution": "fast"})
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_negative_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"resolution": -0.05})
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_short_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"origin": [1.0]})
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_thresholds_inverted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"occupied_thresh": 0.2, "free_thresh": 0.8})
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_unsupported_mode_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"mode": "raw"})
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_unsupported_mode_typo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"mode": "trinery"})  # typo, not a real mode
            with self.assertRaises(MapLoadError):
                load_occupancy_grid(path)

    def test_error_messages_name_the_file_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, omit_keys=["resolution"])
            try:
                load_occupancy_grid(path)
                self.fail("expected MapLoadError")
            except MapLoadError as exc:
                self.assertIn("resolution", str(exc))
                self.assertIn(path, str(exc))


class ModeHandlingTest(unittest.TestCase):
    def test_trinary_mode_is_the_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp)
            grid = load_occupancy_grid(path)
            # top-left pixel (255, free) and bottom-right pixel (0, occupied) -- row 0 is the
            # image's top row per map_io's own documented convention.
            self.assertEqual(grid.cells[0, 0], FREE)
            self.assertEqual(grid.cells[3, 3], OCCUPIED)

    def test_scale_mode_binarizes_the_middle_band_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"mode": "scale"})
            grid = load_occupancy_grid(path)
            self.assertEqual(grid.cells.shape, (4, 4))
            # Still cleanly free/occupied at the unambiguous corners regardless of mode.
            self.assertEqual(grid.cells[0, 0], FREE)
            self.assertEqual(grid.cells[3, 3], OCCUPIED)

    def test_negate_flips_free_and_occupied(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_map(tmp, yaml_overrides={"negate": 1})
            grid = load_occupancy_grid(path)
            self.assertEqual(grid.cells[0, 0], OCCUPIED)  # was FREE under negate=0
            self.assertEqual(grid.cells[3, 3], FREE)  # was OCCUPIED under negate=0


if __name__ == "__main__":
    unittest.main()
