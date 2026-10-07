"""Guards against false matches, data mutation, and invalid synthesis."""
import unittest
from unittest.mock import patch

import numpy as np

from core import Dataset, extract_probe, hit_at, inject, intersects_labels
from evaluate import summarize_probes
from onnx_cpu import preprocess, OnnxCPU
from roi import crop_box, predict_roi


class ProbeTests(unittest.TestCase):
    def test_roi_restores_coordinates_with_one_inference(self):
        image = np.full((100, 200), 150, np.uint8)
        calls = []
        def detector(model, pixels, setting):
            calls.append(pixels.shape)
            return [[1, 2, 11, 12, .8]], {}
        result, _ = predict_roi(None, image, {}, detector, (20, 30, 80, 70))
        self.assertEqual(calls, [(40, 60)])
        self.assertEqual(result, [[21, 32, 31, 42, .8]])

    def test_roi_unreliable_foreground_uses_full_frame(self):
        image = np.full((100, 200), 150, np.uint8)
        self.assertEqual(crop_box(image), (0, 0, 200, 100))

    def test_roi_contains_product_and_stays_inside_image(self):
        image = np.full((100, 200), 240, np.uint8)
        image[20:80, 10:130] = 120
        x0, y0, x1, y1 = crop_box(image)
        self.assertTrue(0 <= x0 < 10 < 130 < x1 <= 200)
        self.assertTrue(0 <= y0 < 20 < 80 < y1 <= 100)

    def test_translation_preserves_signal_and_original(self):
        original = np.full((32, 40), 120, dtype=np.uint8)
        signal = np.array([[0, 0, 0], [0, -31.2, -7.4], [0, -6.1, 0]], np.float32)
        a, abox = inject(original, signal, 8, 9)
        b, bbox = inject(original, signal, 25, 21)
        for out, box in [(a, abox), (b, bbox)]:
            x0, y0, x1, y1 = box
            np.testing.assert_array_equal(out[y0:y1, x0:x1].astype(int)-120, np.rint(signal))
            mask = np.ones(original.shape, bool)
            mask[y0:y1, x0:x1] = False
            np.testing.assert_array_equal(out[mask], original[mask])
        self.assertTrue((original == 120).all())

    def test_invalid_insertion_rejected_without_mutation(self):
        im = np.full((20, 20), 15, np.uint8)
        with self.assertRaises(ValueError):
            inject(im, np.full((5, 5), -20), 10, 10)
        with self.assertRaises(ValueError):
            inject(im, np.zeros((5, 5)), 0, 10)
        self.assertTrue((im == 15).all())

    def test_unrelated_detection_does_not_count(self):
        predictions = [[10, 10, 20, 20, .99], [101, 101, 109, 109, .31]]
        self.assertEqual(hit_at(predictions, [99, 99, 111, 111]), .31)
        self.assertEqual(hit_at(predictions[:1], [99, 99, 111, 111]), 0)

    def test_existing_targets_are_excluded(self):
        boxes = [(.5, .5, .1, .1)]
        self.assertTrue(intersects_labels(51, 51, 7, 7, boxes, (100, 100)))
        self.assertFalse(intersects_labels(80, 80, 7, 7, boxes, (100, 100)))

    def test_recover_compact_signal_without_rectangular_patch(self):
        im = np.full((80, 80), 150, np.uint8)
        im[39:42, 39:42] = 90
        probe = extract_probe(im, (.5, .5, .125, .125))
        signal = probe["signal"]
        self.assertLessEqual(signal.shape[0], 9)
        self.assertLessEqual(signal.shape[1], 9)
        self.assertTrue((signal[0] == 0).all())
        self.assertTrue((signal[:, 0] == 0).all())
        self.assertAlmostEqual(probe["background"].mean(), 150, places=2)

    def test_test_split_cannot_be_opened(self):
        ds = Dataset()
        self.assertIsNone(ds.splits["test"])
        self.assertEqual({r.split for r in ds.rows.values()}, {"train", "val"})
        with patch("core.Image.open", side_effect=AssertionError("Image read attempted")):
            with self.assertRaises(KeyError):
                ds.image("not-in-train-or-val")

    def test_paired_controls_cannot_inflate_success(self):
        scores = [(0, 0, .8), (.8, .8, .9), (0, .7, .9), (0, 0, .2)]
        rows = [{"machine": "1", "grid_y": 0, "grid_x": 0, "background_burst": "burst1",
                 "template_id": "t1", "before_conf": b, "sham_conf": s, "after_conf": a}
                for b, s, a in scores]
        result = summarize_probes(rows, .42, 2)
        self.assertEqual(result["overall"]["n"], 4)
        self.assertEqual(result["overall"]["new_hit"], 1)
        self.assertEqual(result["overall"]["eligible"], 2)
        self.assertEqual(result["overall"]["new_hit_fraction_all"], .25)
        self.assertEqual(result["overall"]["hit_fraction_unconfounded"], .5)
        self.assertEqual(result["n_background_bursts"], 1)
        self.assertIsNone(result["cells"][-1]["new_hit_fraction_all"])

    def test_onnx_preprocessing_matches_reference(self):
        from ultralytics.data.augment import LetterBox
        rng = np.random.default_rng(42)
        for shape in [(332, 316), (332, 352), (332, 412), (444, 576)]:
            im = rng.integers(0, 256, shape, dtype=np.uint8)
            bgr = np.repeat(im[:, :, None], 3, axis=2)
            reference = LetterBox((640, 640), auto=False)(image=bgr)
            reference = np.ascontiguousarray(reference.transpose(2, 0, 1)[None]).astype(np.float32)/255
            actual, *_ = preprocess(im, 640)
            np.testing.assert_array_equal(actual, reference)

    def test_onnx_coordinates_return_to_native_image(self):
        im = np.full((100, 200), 150, np.uint8)
        _, gain, left, top = preprocess(im, 640)
        native = np.array([30, 40, 50, 60], dtype=np.float32)
        scaled = native*gain + np.array([left, top, left, top])
        class Session:
            def run(self, *args):
                return [np.array([[[*scaled, .8, 0], [0, 0, 1, 1, .05, 0]]], np.float32)]
        model = OnnxCPU.__new__(OnnxCPU)
        model.size, model.input_name, model.session = 640, "images", Session()
        result, _ = model.predict(im, conf=.42)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0][:4], native, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
