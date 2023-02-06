"""
Tests for utils.py module
"""
import datetime as dt
import os
from typing import List, Tuple

import numpy as np
import pytest

from sentinelhub import CRS, BBox, SHConfig

from s2cloudless.utils import download_bands_and_valid_data_mask, get_s2_evalscript, get_timestemps

BBOX1 = BBox([-90.9216499, 14.4190528, -90.8186531, 14.5520163], crs=CRS.WGS84)
BBOX2 = BBox(((624024.4, 8214123.1), (661906.6, 8276948.7)), crs=CRS(32738))
DAY = dt.timedelta(seconds=24 * 60 * 60)


@pytest.fixture(name="config")
def config_fixture() -> SHConfig:
    config = SHConfig()

    for param in config.get_params():
        env_variable = param.upper()
        if os.environ.get(env_variable):
            setattr(config, param, os.environ.get(env_variable))

    return config


@pytest.mark.parametrize("all_bands", [True, False])
@pytest.mark.parametrize("reflectance", [True, False])
def test_get_s2_evalscript(all_bands: bool, reflectance: bool) -> None:
    evalscript = get_s2_evalscript(all_bands=all_bands, reflectance=reflectance)

    assert isinstance(evalscript, str)

    bands_num_str = "bands: 14" if all_bands else "bands: 11"
    assert bands_num_str in evalscript

    input_units_str = 'units: "reflectance"' if reflectance else 'units: "DN"'
    assert input_units_str in evalscript
    output_sample_type_str = 'sampleType: "FLOAT32"' if reflectance else 'sampleType: "UINT16"'
    assert output_sample_type_str in evalscript


pytestmark = pytest.mark.sh_integration


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ({"bbox": BBOX1, "time_interval": (dt.datetime(2021, 1, 1), dt.datetime(2021, 1, 1))}, []),
        (
            {"bbox": BBOX1, "time_interval": (dt.datetime(2021, 1, 1), dt.datetime(2021, 1, 10))},
            [
                dt.datetime(2021, 1, 4, 16, 39, 29),
                dt.datetime(2021, 1, 4, 16, 39, 44),
                dt.datetime(2021, 1, 9, 16, 39, 29),
                dt.datetime(2021, 1, 9, 16, 39, 43),
            ],
        ),
        (
            {
                "bbox": BBOX1,
                "time_interval": (dt.datetime(2021, 1, 1), dt.datetime(2021, 1, 10)),
                "time_difference": dt.timedelta(seconds=600),
            },
            [dt.datetime(2021, 1, 4, 16, 39, 29), dt.datetime(2021, 1, 9, 16, 39, 29)],
        ),
    ],
)
def test_get_timestemps(test_input: dict, config: SHConfig, expected: List[dt.datetime]) -> None:
    timestemps = get_timestemps(**test_input, config=config)
    assert len(timestemps) == len(expected)
    assert all([val.replace(tzinfo=None) == exp for exp, val in zip(expected, timestemps)])


@pytest.mark.parametrize(
    "test_input, expected_shape",
    [
        ({"bbox": BBOX1, "timestamps": [dt.datetime(2021, 1, 4)], "size": (60, 81), "all_bands": True}, (1, 81, 60)),
        (
            {
                "bbox": BBOX1,
                "timestamps": [dt.datetime(2021, 1, 4, 16, 39, 29), dt.datetime(2021, 1, 9, 16, 39, 29)],
                "size": (60, 81),
                "all_bands": True,
            },
            (2, 81, 60),
        ),
        (
            {
                "bbox": BBOX1,
                "timestamps": [dt.datetime(2021, 1, 4, 16, 39, 29), dt.datetime(2021, 1, 9, 16, 39, 29)],
                "size": (60, 81),
                "all_bands": False,
            },
            (2, 81, 60),
        ),
    ],
)
def test_download_bands_and_valid_data_mask(
    test_input: dict, config: SHConfig, expected_shape: Tuple[int, int, int]
) -> None:
    bands, mask = download_bands_and_valid_data_mask(**test_input, config=config)
    band_num = 10 if not test_input["all_bands"] else 13
    assert bands.shape == expected_shape + (band_num,)
    assert bands.dtype == np.float32
    assert mask.shape == expected_shape
    assert mask.dtype == bool
