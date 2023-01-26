"""
Tests for cloud_detector.py module
"""
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from s2cloudless import S2PixelCloudDetector

pytestmark = pytest.mark.fast


def test_pixel_cloud_detector():
    data_path = os.path.join("TestInputs", "input_arrays.npz")
    data = np.load(data_path)

    cloud_detector = S2PixelCloudDetector(all_bands=True)
    cloud_probs = cloud_detector.get_cloud_probability_maps(data["s2_im"])
    cloud_mask_from_prob = cloud_detector.get_mask_from_prob(cloud_probs)
    cloud_mask = cloud_detector.get_cloud_masks(data["s2_im"])

    assert_allclose(cloud_probs, data["cl_probs"], atol=1e-14)
    assert_array_equal(cloud_mask, data["cl_mask"])
    assert_array_equal(cloud_mask_from_prob, data["cl_mask"])

    single_temporal_cloud_probs = cloud_detector.get_cloud_probability_maps(data["s2_im"][0, ...])
    single_temporal_cloud_mask_from_probs = cloud_detector.get_mask_from_prob(single_temporal_cloud_probs)
    single_temporal_cloud_mask = cloud_detector.get_cloud_masks(data["s2_im"][0, ...])

    assert_array_equal(single_temporal_cloud_probs, cloud_probs[0, ...])
    assert_array_equal(single_temporal_cloud_mask, cloud_mask[0, ...])
    assert_array_equal(single_temporal_cloud_mask_from_probs, cloud_mask[0, ...])

    cloud_detector = S2PixelCloudDetector(all_bands=False)
    with pytest.raises(ValueError):
        cloud_detector.get_cloud_probability_maps(data["s2_im"])
