"""
Tests for sentinelhub_masking.py module
"""
import datetime as dt
import os

import numpy as np
import pytest

from sentinelhub import CRS, BBox, SHConfig
from sentinelhub.testing_utils import assert_statistics_match

from s2cloudless import CloudMaskRequest, NoDataAvailableException, S2PixelCloudDetector

pytestmark = pytest.mark.fast

BBOX1 = BBox([-90.9216499, 14.4190528, -90.8186531, 14.5520163], crs=CRS.WGS84)
BBOX2 = BBox(((624024.4, 8214123.1), (661906.6, 8276948.7)), crs=CRS.UTM_38S)


@pytest.fixture(name="config")
def config_fixture():
    config = SHConfig()

    for param in config.get_params():
        env_variable = param.upper()
        if os.environ.get(env_variable):
            setattr(config, param, os.environ.get(env_variable))

    return config


@pytest.mark.parametrize(
    "input_params,stats",
    [
        (
            dict(
                cloud_detector=S2PixelCloudDetector(threshold=0.6, average_over=2, dilation_size=5, all_bands=True),
                bbox=BBOX1,
                time=("2017-12-01", "2017-12-31"),
                size=(60, 81),
            ),
            dict(
                clm_min=0,
                clm_max=1,
                clm_mean=0.343827,
                clm_median=0,
                clp_min=0.00011,
                clp_max=0.99999,
                clp_mean=0.23959,
                clp_median=0.01897,
                mask_shape=(7, 81, 60),
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
            dict(
                clm_min=0,
                clm_max=1,
                clm_mean=0.05164,
                clm_median=0,
                clp_min=0.00011,
                clp_max=0.99966,
                clp_mean=0.055365,
                clp_median=0.0141596,
                mask_shape=(1, 628, 379),
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
            dict(
                clm_min=0,
                clm_max=1,
                clm_mean=0.87632,
                clm_median=1,
                clp_min=0.00352,
                clp_max=0.99966,
                clp_mean=0.789795,
                clp_median=0.953203,
                mask_shape=(1, 250, 250),
            ),
        ),
    ],
    ids=["basic", "resolution", "maxcc,downsampling"],
)
def test_cloud_mask_request(input_params, stats, config):
    """Integration tests for CloudMasKRequest class that interacts with Sentinel Hub service"""
    request = CloudMaskRequest(config=config, **input_params)

    masks = request.get_cloud_masks()
    assert_statistics_match(
        masks,
        exp_shape=stats["mask_shape"],
        exp_dtype=np.int8,
        exp_min=stats["clm_min"],
        exp_max=stats["clm_max"],
        exp_mean=stats["clm_mean"],
        exp_median=stats["clm_median"],
        abs_delta=1e-4,
    )

    prob_masks = request.get_probability_masks(non_valid_value=-50)
    assert_statistics_match(
        prob_masks,
        exp_shape=stats["mask_shape"],
        exp_dtype=np.float64,
        exp_min=stats["clp_min"],
        exp_max=stats["clp_max"],
        exp_mean=stats["clp_mean"],
        exp_median=stats["clp_median"],
        abs_delta=1e-4,
    )

    timestamps = request.get_timestamps()
    assert isinstance(timestamps, list)
    assert len(timestamps) == stats["mask_shape"][0]
    assert all(isinstance(timestamp, dt.datetime) for timestamp in timestamps)

    data = request.get_data()
    band_num = 13 if request.cloud_detector.all_bands else 10
    assert data.shape == stats["mask_shape"] + (band_num,)
    assert data.dtype == np.float32

    data_mask = request.get_data_mask()
    assert data_mask.shape == stats["mask_shape"]
    assert data_mask.dtype == bool


def test_no_data_available_request(config):
    """Tests an exception raised by CloudMaskRequest"""
    cloud_detector = S2PixelCloudDetector()
    with pytest.raises(NoDataAvailableException):
        CloudMaskRequest(
            cloud_detector, bbox=BBOX1, time=("2021-01-01", "2021-01-10"), size=(250, 250), maxcc=0.01, config=config
        )
