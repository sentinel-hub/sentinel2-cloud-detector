"""Module providing various utilities."""

from __future__ import annotations

import datetime as dt

import cv2
import numpy as np

from sentinelhub import BBox, DataCollection, MimeType, SentinelHubDownloadClient, SentinelHubRequest, SHConfig
from sentinelhub.evalscript import generate_evalscript

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
MODEL_BAND_IDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]
MODEL_BANDS = [S2_BANDS[band_idx] for band_idx in MODEL_BAND_IDS]


# pylint: disable-msg=too-many-locals
def download_bands_and_valid_data_mask(
    bbox: BBox,
    timestamps: list[dt.datetime],
    *,
    data_collection: DataCollection = DataCollection.SENTINEL2_L1C,
    config: SHConfig | None = None,
    size: tuple[int, int] | None = None,
    resolution: tuple[float, float] | None = None,
    all_bands: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Download all data required for running the cloud-masking process."""

    client = SentinelHubDownloadClient(config=config)

    bands_output = "bands"
    data_mask_output = "dataMask"
    bands = S2_BANDS if all_bands else MODEL_BANDS
    evalscript = generate_evalscript(
        data_collection=data_collection,
        bands=bands,
        meta_bands=[data_mask_output],
        merged_bands_output=bands_output,
        prioritize_dn=True,
    )

    api_requests = []
    for timestamp in timestamps:
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=(timestamp, timestamp),
                )
            ],
            responses=[
                SentinelHubRequest.output_response(bands_output, MimeType.TIFF),
                SentinelHubRequest.output_response(data_mask_output, MimeType.TIFF),
                SentinelHubRequest.output_response("userdata", MimeType.JSON),
            ],
            bbox=bbox,
            size=size,
            resolution=resolution,
            config=config,
        )

        api_requests.extend(request.download_list)

    responses = client.download(api_requests)

    bands, mask = [], []
    for response in responses:
        bands.append(response[f"{bands_output}.tif"] * response["userdata.json"]["norm_factor"])
        mask.append(response[f"{data_mask_output}.tif"])

    return np.array(bands, dtype=np.float32), np.array(mask, dtype=bool)


def cv2_disk(radius: int) -> np.ndarray:
    """Recreates the disk structural element from skimage.morphology using OpenCV."""
    return cv2.circle(  # type: ignore[call-overload]
        np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8), (radius, radius), radius, color=1, thickness=-1
    )
