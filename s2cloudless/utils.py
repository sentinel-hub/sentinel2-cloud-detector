"""Module providing various utilities."""

import datetime as dt
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sentinelhub import (
    BBox,
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
    filter_times,
)
from sentinelhub.evalscript import generate_evalscript

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
MODEL_BAND_IDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]
MODEL_BANDS = [S2_BANDS[band_idx] for band_idx in MODEL_BAND_IDS]


def get_timestamps(
    bbox: BBox,
    time_interval: Tuple[dt.datetime, dt.datetime],
    *,
    maxcc: Optional[float] = None,
    data_collection: DataCollection = DataCollection.SENTINEL2_L1C,
    config: Optional[SHConfig] = None,
    time_difference: Optional[dt.timedelta] = None,
) -> List[dt.datetime]:
    """Get the list of timestamps for which data is available. Takes into account the bbox and time interval."""
    time_difference = time_difference if time_difference else dt.timedelta(seconds=0)

    catalog = SentinelHubCatalog(config=config)
    cloud_cover_query = None
    if maxcc is not None:
        cloud_cover_query = f"eo:cloud_cover < {100 * float(maxcc)}"

    search_iterator = catalog.search(
        data_collection,
        bbox=bbox,
        time=time_interval,
        filter=cloud_cover_query,
        fields={
            "include": [
                "properties.datetime",
            ],
            "exclude": [],
        },
    )

    return filter_times(search_iterator.get_timestamps(), time_difference)


# pylint: disable-msg=too-many-locals
def download_bands_and_valid_data_mask(
    bbox: BBox,
    timestamps: List[dt.datetime],
    *,
    data_collection: DataCollection = DataCollection.SENTINEL2_L1C,
    config: Optional[SHConfig] = None,
    size: Optional[Tuple[int, int]] = None,
    resolution: Optional[Tuple[float, float]] = None,
    all_bands: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Download all data required for running the cloud-masking process."""

    client = SentinelHubDownloadClient(config=config)

    bands_output = "bands"
    data_mask_output = "dataMask"
    bands = S2_BANDS if all_bands else MODEL_BANDS
    evalscript = generate_evalscript(
        data_collection=data_collection,
        bands=bands,
        meta_bands=[data_mask_output],
        merged_output=bands_output,
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
    return cv2.circle(
        np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8), (radius, radius), radius, color=1, thickness=-1
    )
