"""End-to-end integration test."""

import os
import json
import unittest
import numpy as np

TASK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REF_DIR = os.path.join(TASK_DIR, 'evaluation', 'reference_outputs')

import sys
sys.path.insert(0, TASK_DIR)


class TestEndToEnd(unittest.TestCase):
    """Verify that the ehtim reference outputs demonstrate the expected physics."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(REF_DIR, 'metrics.json')) as f:
            cls.metrics = json.load(f)
        cls.gt = np.load(os.path.join(REF_DIR, 'ground_truth.npy'))

    def test_vis_cal_is_excellent(self):
        """Vis RML on calibrated data should be the best."""
        m = self.metrics['Vis RML (cal)']
        self.assertGreater(m['ncc'], 0.9)
        self.assertLess(m['nrmse'], 0.4)

    def test_vis_corrupt_fails(self):
        """Vis RML on corrupted data should fail catastrophically."""
        m = self.metrics['Vis RML (corrupt)']
        self.assertLess(m['ncc'], 0.1)

    def test_closure_corrupt_is_robust(self):
        """Closure-only on corrupted data should remain robust."""
        m = self.metrics['Closure-only (corrupt)']
        self.assertGreater(m['ncc'], 0.6)

    def test_closure_outperforms_vis_on_corrupt(self):
        """Closure-only should beat Vis RML on corrupted data."""
        ncc_clo = self.metrics['Closure-only (corrupt)']['ncc']
        ncc_vis = self.metrics['Vis RML (corrupt)']['ncc']
        self.assertGreater(ncc_clo, ncc_vis)

    def test_ground_truth_shape(self):
        self.assertEqual(self.gt.shape, (64, 64))

    def test_ground_truth_normalized(self):
        np.testing.assert_allclose(self.gt.sum(), 1.0, rtol=1e-6)

    def test_reference_images_exist(self):
        for name in ['closure-only_corrupt', 'vis_rml_cal']:
            path = os.path.join(REF_DIR, f'{name}.npy')
            self.assertTrue(
                os.path.exists(path),
                f"Missing reference output: {path}")


if __name__ == '__main__':
    unittest.main()
