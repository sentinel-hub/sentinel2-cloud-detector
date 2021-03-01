"""
Tests for sentinelhub_masking.py module
"""
import os
import datetime as dt

import pytest
import numpy as np
from sentinelhub import SHConfig, BBox, CRS

from s2cloudless import S2PixelCloudDetector, CloudMaskRequest


BBOX1 = BBox([-90.9216499, 14.4190528, -90.8186531, 14.5520163], crs=CRS.WGS84)
BBOX2 = BBox(((624024.4, 8214123.1), (661906.6, 8276948.7)), crs=CRS.UTM_38S)


@pytest.fixture(name='config')
def config_fixture():
    config = SHConfig()

    for param in config.get_params():
        env_variable = param.upper()
        if os.environ.get(env_variable):
            setattr(config, param, os.environ.get(env_variable))

    return config


@pytest.mark.parametrize('input_params,stats', [
    (dict(cloud_detector=S2PixelCloudDetector(threshold=0.6, average_over=2, dilation_size=5, all_bands=True),
          bbox=BBOX1, time=('2017-12-01', '2017-12-31'), size=(60, 81)),
     dict(clm_min=0, clm_max=1, clm_mean=0.343827, clm_median=0, clp_min=0.00011, clp_max=0.99999, clp_mean=0.23959,
          clp_median=0.01897, mask_shape=(7, 81, 60))),
    (dict(cloud_detector=S2PixelCloudDetector(), bbox=BBOX2, time='2016-07-18T07:14:04', resolution=(100, 100),
          time_difference=dt.timedelta(hours=1)),
     dict(clm_min=0, clm_max=1, clm_mean=0.05164, clm_median=0, clp_min=0.00011, clp_max=0.99966, clp_mean=0.055365,
          clp_median=0.0141596, mask_shape=(1, 628, 379)))
])
def test_cloud_mask_request(input_params, stats, config, subtests):
    """ Integration tests for CloudMasKRequest class that interacts with Sentinel Hub service
    """
    request = CloudMaskRequest(config=config, **input_params)

    masks = request.get_cloud_masks()
    _test_numpy_data(subtests, masks, exp_shape=stats['mask_shape'], exp_dtype=np.int8, exp_min=stats['clm_min'],
                     exp_max=stats['clm_max'], exp_mean=stats['clm_mean'], exp_median=stats['clm_median'], delta=1e-4)

    prob_masks = request.get_probability_masks(non_valid_value=-50)
    _test_numpy_data(subtests, prob_masks, exp_shape=stats['mask_shape'], exp_dtype=np.float64, exp_min=stats['clp_min'],
                     exp_max=stats['clp_max'], exp_mean=stats['clp_mean'], exp_median=stats['clp_median'], delta=1e-4)

    timestamps = request.get_timestamps()
    assert isinstance(timestamps, list)
    assert len(timestamps) == stats['mask_shape'][0]
    assert all(isinstance(timestamp, dt.datetime) for timestamp in timestamps)

    data = request.get_data()
    band_num = 13 if request.cloud_detector.all_bands else 10
    assert data.shape == stats['mask_shape'] + (band_num,)
    assert data.dtype == np.float32

    data_mask = request.get_data_mask()
    assert data_mask.shape == stats['mask_shape']
    assert data_mask.dtype == bool


def _test_numpy_data(subtests, data, *, exp_shape=None, exp_dtype=None, exp_min=None, exp_max=None, exp_mean=None,
                     exp_median=None, delta=None):
    if delta is None:
        delta = 1e-1 if np.issubdtype(data.dtype, np.integer) else 1e-4

    for exp_stat, stat_val, stat_name in [(exp_shape, data.shape, 'shape'), (exp_dtype, data.dtype, 'dtype')]:
        if exp_stat is None:
            continue

        with subtests.test(msg=stat_name):
            assert stat_val == exp_stat, f'Expected {stat_name} {exp_stat}, got {stat_val}'

    data = data[~np.isnan(data)]
    for exp_stat, stat_func, stat_name in [(exp_min, np.amin, 'min'), (exp_max, np.amax, 'max'),
                                           (exp_mean, np.mean, 'mean'), (exp_median, np.median, 'median')]:
        if exp_stat is None:
            continue

        stat_val = stat_func(data)
        with subtests.test(msg=stat_name):
            assert abs(stat_val - exp_stat) < delta, f'Expected {stat_name} {exp_stat}, got {stat_val}'
