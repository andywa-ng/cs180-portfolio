"""Behavioral checks: known translations, brightness changes, and valid overlap."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from proj1 import align, compose, downsample, read_channels, search_translation, to_float


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(180)

    def test_single_recovers_signed_xy_with_both_metrics(self):
        reference = self.rng.random((121, 139), dtype=np.float32)
        for dx, dy in ((7, -5), (-8, 6), (0, 0)):
            moving = np.roll(reference, (-dy, -dx), axis=(0, 1))
            for metric in ("ncc", "l2"):
                with self.subTest(shift=(dx, dy), metric=metric):
                    actual, _ = align(moving, reference, method="single", radius=10, metric=metric)
                    self.assertEqual(actual, (dx, dy))

    def test_ncc_handles_positive_gain_and_brightness_offset(self):
        reference = self.rng.random((121, 139), dtype=np.float32)
        moving = 0.6 * np.roll(reference, (4, -9), axis=(0, 1)) + 0.2
        self.assertEqual(align(moving, reference, method="single")[0], (9, -4))

    def test_pyramid_recovers_large_shift_on_odd_dimensions(self):
        reference = self.rng.random((701, 809), dtype=np.float32)
        # Low-frequency structure survives coarse downsampling.
        for _ in range(4):
            reference = (reference + np.roll(reference, 1, 0) + np.roll(reference, 1, 1)) / 3
        moving = np.roll(reference, (29, -73), axis=(0, 1))
        shift, trace = align(moving, reference, coarsest_size=180)
        self.assertEqual(shift, (73, -29))
        self.assertGreater(len(trace), 1)

    def test_composition_excludes_all_wraparound_pixels(self):
        b = self.rng.random((80, 100), dtype=np.float32)
        g = np.roll(b, (5, -7), axis=(0, 1))
        r = np.roll(b, (-8, 4), axis=(0, 1))
        rgb, (x0, y0, x1, y1) = compose((b, g, r), (7, -5), (-4, 8))
        for c in range(3):
            np.testing.assert_array_equal(rgb[..., c], b[y0:y1, x0:x1])

    def test_uint8_uint16_equivalence(self):
        a = np.array([0, 51, 255], dtype=np.uint8)
        np.testing.assert_allclose(to_float(a), to_float(a.astype(np.uint16)*257))

    def test_bgr_split_discards_remainder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "plate.tif"
            plate = np.vstack([np.full((24, 30), v, np.uint16) for v in (0, 32768, 65535)])
            plate = np.vstack((plate, np.zeros((2, 30), np.uint16)))
            Image.fromarray(plate).save(path)
            b, g, r = read_channels(path)
            self.assertEqual(b.shape, (24, 30))
            self.assertEqual(float(b.mean()), 0)
            self.assertAlmostEqual(float(g.mean()), 32768/65535, places=6)
            self.assertEqual(float(r.mean()), 1)

    def test_downsample_preserves_constant_and_reduces_aliasing(self):
        np.testing.assert_allclose(downsample(np.ones((41, 45), np.float32)), 1)
        checker = (np.indices((80, 80)).sum(axis=0) % 2).astype(np.float32)
        np.testing.assert_allclose(downsample(checker)[2:-2, 2:-2], 0.5)

    def test_constant_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no contrast"):
            search_translation(np.ones((100, 100)), np.ones((100, 100)))


if __name__ == "__main__":
    unittest.main()
