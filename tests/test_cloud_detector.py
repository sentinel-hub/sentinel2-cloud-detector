"""
Tests for cloud_detector.py module
"""
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2cloudless import S2PixelCloudDetector

DATA = np.load(os.path.join(os.path.dirname(__file__), "TestInputs", "input_arrays.npz"))


@pytest.fixture(name="cloud_detector")
def cloud_detector_fixture() -> S2PixelCloudDetector:
    return S2PixelCloudDetector(all_bands=True)


@pytest.mark.parametrize("data, result", [(DATA["s2_im"], DATA["cl_probs"]), (DATA["s2_im"][0], DATA["cl_probs"][0])])
def test_get_cloud_probability_maps(cloud_detector: S2PixelCloudDetector, data: np.ndarray, result: np.ndarray) -> None:
    cloud_probs = cloud_detector.get_cloud_probability_maps(data)
    assert_allclose(cloud_probs, result, rtol=1e-5)


@pytest.mark.parametrize("data, result", [(DATA["s2_im"], DATA["cl_mask"]), (DATA["s2_im"][0], DATA["cl_mask"][0])])
def test_get_cloud_masks(cloud_detector: S2PixelCloudDetector, data: np.ndarray, result: np.ndarray) -> None:
    cloud_mask = cloud_detector.get_cloud_masks(data)
    assert_array_equal(cloud_mask, result)


@pytest.mark.parametrize(
    "data, result", [(DATA["cl_probs"], DATA["cl_mask"]), (DATA["cl_probs"][0], DATA["cl_mask"][0])]
)
def test_get_mask_from_prob(cloud_detector: S2PixelCloudDetector, data: np.ndarray, result: np.ndarray) -> None:
    cloud_mask_from_prob = cloud_detector.get_mask_from_prob(data)
    assert_array_equal(cloud_mask_from_prob, result)


def test_cloud_detector_failure_wrong_number_of_bands() -> None:
    """Value error is raised because get_cloud_probability_maps expecting only 10 bands but all 13 are given"""
    cloud_detector = S2PixelCloudDetector(all_bands=False)
    with pytest.raises(ValueError):
        cloud_detector.get_cloud_probability_maps(DATA["s2_im"])
