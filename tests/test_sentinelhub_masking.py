"""
Tests for sentinelhub_masking.py module
"""
import datetime as dt
import os
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from sentinelhub import CRS, BBox, SHConfig
from sentinelhub.testing_utils import assert_statistics_match

from s2cloudless import CloudMaskRequest, NoDataAvailableException, S2PixelCloudDetector

pytestmark = pytest.mark.sh_integration

BBOX1 = BBox([-90.9216499, 14.4190528, -90.8186531, 14.5520163], crs=CRS.WGS84)
BBOX2 = BBox(((624024.4, 8214123.1), (661906.6, 8276948.7)), crs=CRS(32738))


@pytest.fixture(name="config")
def config_fixture() -> SHConfig:
    config = SHConfig()

    for param in config.get_params():
        env_variable = param.upper()
        if os.environ.get(env_variable):
            setattr(config, param, os.environ.get(env_variable))

    return config


@pytest.mark.parametrize(
    "input_params, mask_shape, clm_stats, clp_stats",
    [
        (
            dict(
                cloud_detector=S2PixelCloudDetector(threshold=0.6, average_over=2, dilation_size=5, all_bands=True),
                bbox=BBOX1,
                time=("2017-12-01", "2017-12-31"),
                size=(60, 81),
            ),
            (7, 81, 60),
            dict(
                exp_min=0,
                exp_max=1,
                exp_mean=0.343827,
                exp_median=0,
            ),
            dict(
                exp_min=0.00011,
                exp_max=0.99999,
                exp_mean=0.23959,
                exp_median=0.01897,
            ),
        ),
        (
            dict(
                cloud_detector=S2PixelCloudDetector(),
                bbox=BBOX2,
                time=[dt.datetime(2016, 7, 18, 7, 14, 4)],
                resolution=(100, 100),
                time_difference=dt.timedelta(hours=1),
            ),
            (1, 628, 379),
            dict(
                exp_min=0,
                exp_max=1,
                exp_mean=0.05164,
                exp_median=0,
            ),
            dict(
                exp_min=0.00011,
                exp_max=0.99966,
                exp_mean=0.055365,
                exp_median=0.0141596,
            ),
        ),
        (
            dict(
                cloud_detector=S2PixelCloudDetector(),
                bbox=BBOX1,
                time=("2021-01-01", "2021-01-10"),
                size=(250, 250),
                time_difference=dt.timedelta(hours=1),
                maxcc=0.1,
                downsampling="BICUBIC",
            ),
            (1, 250, 250),
            dict(
                exp_min=0,
                exp_max=1,
                exp_mean=0.87632,
                exp_median=1,
            ),
            dict(
                exp_min=0.00352,
                exp_max=0.99966,
                exp_mean=0.789795,
                exp_median=0.953203,
            ),
        ),
    ],
    ids=["basic", "resolution", "maxcc,downsampling"],
)
def test_cloud_mask_request(
    input_params: Dict[str, Any],
    mask_shape: Tuple[int, int, int],
    clm_stats: Dict[str, float],
    clp_stats: Dict[str, float],
    config: SHConfig,
) -> None:
    """Integration tests for CloudMasKRequest class that interacts with Sentinel Hub service"""
    request = CloudMaskRequest(config=config, **input_params)

    masks = request.get_cloud_masks()
    assert_statistics_match(masks, exp_shape=mask_shape, exp_dtype=np.dtype(np.int8), **clm_stats, abs_delta=1e-4)

    prob_masks = request.get_probability_masks(non_valid_value=-50)
    assert_statistics_match(
        prob_masks, exp_shape=mask_shape, exp_dtype=np.dtype(np.float64), **clp_stats, abs_delta=1e-4
    )

    timestamps = request.get_timestamps()
    assert isinstance(timestamps, list)
    assert len(timestamps) == mask_shape[0]
    assert all(isinstance(timestamp, dt.datetime) for timestamp in timestamps)

    data, data_mask = request._download_bands_and_valid_data_mask()
    band_num = 13 if request.cloud_detector.all_bands else 10
    assert data.shape == mask_shape + (band_num,)
    assert data.dtype == np.float32

    assert data_mask.shape == mask_shape
    assert data_mask.dtype == bool


def test_no_data_available_request(config: SHConfig) -> None:
    """Tests an exception raised by CloudMaskRequest"""
    cloud_detector = S2PixelCloudDetector()
    with pytest.raises(NoDataAvailableException):
        CloudMaskRequest(
            cloud_detector, bbox=BBOX1, time=("2021-01-01", "2021-01-10"), size=(250, 250), maxcc=0.01, config=config
        )
