import os
import json
import pytest
import numpy as np

# Resolve paths relative to task root
TASK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "preprocessing")

import sys
sys.path.insert(0, TASK_DIR)
from src.preprocessing import load_observation, load_metadata, normalize_image, prepare_data


class TestLoadObservation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.fixture = np.load(os.path.join(FIXTURE_DIR, "output_load_observation.npz"),
                                allow_pickle=True)
        self.obs = load_observation(os.path.join(TASK_DIR, "data"))

    def test_returns_dict_with_img_key(self):
        assert "img" in self.obs

    def test_shape(self):
        expected_shape = tuple(self.fixture["output_img_shape"])
        assert self.obs["img"].shape == expected_shape

    def test_value_range(self):
        assert self.obs["img"].min() >= 0


class TestLoadMetadata:
    @pytest.fixture(autouse=True)
    def setup(self):
        with open(os.path.join(FIXTURE_DIR, "output_load_metadata.json")) as f:
            self.expected = json.load(f)
        self.metadata = load_metadata(os.path.join(TASK_DIR, "data"))

    def test_required_keys(self):
        required = ["image_size", "num_lines", "num_iter", "step_size",
                     "patch_size", "stride", "state_num", "sigma"]
        for key in required:
            assert key in self.metadata, f"Missing key: {key}"

    def test_values_match(self):
        for key in self.expected:
            assert self.metadata[key] == self.expected[key], f"Mismatch for {key}"


class TestNormalizeImage:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.input_fix = np.load(os.path.join(FIXTURE_DIR, "input_normalize_image.npz"))
        self.output_fix = np.load(os.path.join(FIXTURE_DIR, "output_normalize_image.npz"))

    def test_output_range(self):
        result = normalize_image(self.input_fix["img"])
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_values(self):
        result = normalize_image(self.input_fix["img"])
        np.testing.assert_allclose(result, self.output_fix["img"], rtol=1e-6)


class TestPrepareData:
    def test_returns_four_elements(self):
        result = prepare_data(os.path.join(TASK_DIR, "data"))
        assert len(result) == 4

    def test_image_normalized(self):
        img, mask, y, metadata = prepare_data(os.path.join(TASK_DIR, "data"))
        np.testing.assert_allclose(img.min(), 0.0, atol=1e-10)
        np.testing.assert_allclose(img.max(), 1.0, atol=1e-10)

    def test_mask_is_boolean(self):
        img, mask, y, metadata = prepare_data(os.path.join(TASK_DIR, "data"))
        assert mask.dtype == bool
