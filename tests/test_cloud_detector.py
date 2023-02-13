import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2cloudless import S2PixelCloudDetector

DATA = np.load(os.path.join(os.path.dirname(__file__), "TestInputs", "input_arrays.npz"))


@pytest.fixture(name="cloud_detector")
def cloud_detector_fixture() -> S2PixelCloudDetector:
    return S2PixelCloudDetector(all_bands=True)


@pytest.mark.parametrize("data", [np.array([]), np.ones((5,)), np.ones((5, 3, 2, 1, 3)), np.ones((5, 4, 5))])
@pytest.mark.parametrize("method", ["get_cloud_masks", "get_cloud_probability_maps"])
def test_incorrect_dimension_input_fails(cloud_detector: S2PixelCloudDetector, method: str, data: np.ndarray) -> None:
    test_method = getattr(cloud_detector, method)
    with pytest.raises(ValueError):
        test_method(data)


@pytest.mark.parametrize("data, result", [(DATA["s2_im"], DATA["cl_probs"])])
def test_get_cloud_probability_maps(cloud_detector: S2PixelCloudDetector, data: np.ndarray, result: np.ndarray) -> None:
    cloud_probs = cloud_detector.get_cloud_probability_maps(data)
    assert cloud_probs.dtype == np.float32
    assert_allclose(cloud_probs, result, rtol=1e-5)


@pytest.mark.parametrize("data, result", [(DATA["s2_im"], DATA["cl_mask"])])
def test_get_cloud_masks(cloud_detector: S2PixelCloudDetector, data: np.ndarray, result: np.ndarray) -> None:
    cloud_mask = cloud_detector.get_cloud_masks(data)
    assert cloud_mask.dtype == np.uint8
    assert_array_equal(cloud_mask, result)


@pytest.mark.parametrize("data, result", [(DATA["cl_probs"], DATA["cl_mask"])])
def test_get_mask_from_prob(cloud_detector: S2PixelCloudDetector, data: np.ndarray, result: np.ndarray) -> None:
    cloud_mask_from_prob = cloud_detector.get_mask_from_prob(data)
    assert cloud_mask_from_prob.dtype == np.uint8
    assert_array_equal(cloud_mask_from_prob, result)


def test_cloud_detector_failure_wrong_number_of_bands() -> None:
    """Value error is raised because get_cloud_probability_maps expecting only 10 bands but all 13 are given"""
    cloud_detector = S2PixelCloudDetector(all_bands=False)
    with pytest.raises(ValueError):
        cloud_detector.get_cloud_probability_maps(DATA["s2_im"])
