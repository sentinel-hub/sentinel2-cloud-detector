"""
Tests for cloud_detector.py module
"""
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2cloudless import S2PixelCloudDetector


@pytest.fixture(name="data", scope="module")
def data_fixture():
    return np.load(os.path.join("TestInputs", "input_arrays.npz"))


@pytest.fixture(name="cloud_detector", scope="module")
def cloud_detector_fixture():
    return S2PixelCloudDetector(all_bands=True)


@pytest.fixture(name="cloud_detector_not_all", scope="module")
def cloud_detector_not_all_fixture():
    return S2PixelCloudDetector(all_bands=False)


def test_get_cloud_probability_maps(cloud_detector, data):
    cloud_probs = cloud_detector.get_cloud_probability_maps(data["s2_im"])
    assert_allclose(cloud_probs, data["cl_probs"], atol=1e-14)

    single_temporal_cloud_probs = cloud_detector.get_cloud_probability_maps(data["s2_im"][0, ...])
    assert_allclose(single_temporal_cloud_probs, data["cl_probs"][0, ...], atol=1e-14)


def test_get_cloud_masks(cloud_detector, data):
    cloud_mask = cloud_detector.get_cloud_masks(data["s2_im"])
    assert_array_equal(cloud_mask, data["cl_mask"])

    single_temporal_cloud_mask = cloud_detector.get_cloud_masks(data["s2_im"][0, ...])
    assert_array_equal(single_temporal_cloud_mask, data["cl_mask"][0, ...])


def test_get_mask_from_prob(cloud_detector, data):
    cloud_mask_from_prob = cloud_detector.get_mask_from_prob(data["cl_probs"])
    assert_array_equal(cloud_mask_from_prob, data["cl_mask"])

    single_temporal_cloud_mask_from_probs = cloud_detector.get_mask_from_prob(data["cl_probs"][0, ...])
    assert_array_equal(single_temporal_cloud_mask_from_probs, data["cl_mask"][0, ...])


def test_get_cloud_probability_maps_invalid(cloud_detector_not_all, data):
    with pytest.raises(ValueError):
        cloud_detector_not_all.get_cloud_probability_maps(data["s2_im"])
