from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from sentinelhub import CRS, BBox

from s2cloudless.utils import download_bands_and_valid_data_mask

BBOX1 = BBox((-90.9216499, 14.4190528, -90.8186531, 14.5520163), crs=CRS.WGS84)
BBOX2 = BBox(((620000, 8210000), (660000, 8270000)), crs=CRS(32738))


@pytest.mark.parametrize(
    ("test_input", "expected_shape"),
    [
        ({"bbox": BBOX1, "timestamps": [dt.datetime(2021, 1, 4)], "size": (60, 81)}, (1, 81, 60, 13)),
        (
            {
                "bbox": BBOX1,
                "timestamps": [dt.datetime(2021, 1, 4, 16, 39, 29), dt.datetime(2021, 1, 9, 16, 39, 29)],
                "size": (60, 81),
                "all_bands": True,
            },
            (2, 81, 60, 13),
        ),
        (
            {
                "bbox": BBOX1,
                "timestamps": [dt.datetime(2021, 1, 4, 16, 39, 29), dt.datetime(2021, 1, 9, 16, 39, 29)],
                "size": (60, 81),
                "all_bands": False,
            },
            (2, 81, 60, 10),
        ),
        (
            {
                "bbox": BBOX2,
                "timestamps": [dt.datetime(2016, 7, 18, 7, 14, 4)],
                "resolution": (200, 200),
                "all_bands": False,
            },
            (1, 300, 200, 10),
        ),
    ],
)
@pytest.mark.sh_integration()
def test_download_bands_and_valid_data_mask(test_input: dict, expected_shape: tuple[int, int, int]) -> None:
    bands, mask = download_bands_and_valid_data_mask(**test_input)
    assert bands.shape == expected_shape
    assert bands.dtype == np.float32
    assert mask.shape == expected_shape[:-1]
    assert mask.dtype == bool
