"""Module providing various utilities."""

import datetime as dt
from typing import List, Optional, Tuple

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

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
MODEL_BAND_IDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]

DATA_EVALSCRIPT_TEMPLATE = """
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: [{bands}],
      units: "{input_units}"
    }}],
    output: {{
      bands: {band_number},
      sampleType: "{output_sample_type}"
    }}
  }};
}}
{metadata_evalscript}
function evaluatePixel(sample) {{
  return [{sample_bands}];
}}
"""

METADATA_EVALSCRIPT_TEMPLATE = """
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {
    "norm_factor":  inputMetadata.normalizationFactor
  }
}
"""


def get_s2_evalscript(all_bands: bool = False, reflectance: bool = False) -> str:
    """Provides an `s2cloudless` suited evalscript to download Sentinel-2 data

    :param all_bands: If `True` the evalscript will use all bands. Otherwise it will use only bands needed for cloud
        masking.
    :param reflectance: If `True` the evalscript will define reflectance values. Otherwise it will use digital
        numbers together with normalization factors to rescale them.
    :return: An evalscript to be used with the `sentinelhub` package.
    """
    bands = S2_BANDS
    if not all_bands:
        bands = [bands[index] for index in MODEL_BAND_IDS]
    bands = bands + ["dataMask"]

    return DATA_EVALSCRIPT_TEMPLATE.format(
        bands=", ".join(f'"{band}"' for band in bands),
        sample_bands=", ".join(f"sample.{band}" for band in bands),
        band_number=len(bands),
        input_units="reflectance" if reflectance else "DN",
        output_sample_type="FLOAT32" if reflectance else "UINT16",
        metadata_evalscript="" if reflectance else METADATA_EVALSCRIPT_TEMPLATE,
    ).strip("\n ")


def get_timestemps(
    bbox: BBox,
    time_interval: Tuple[dt.datetime, dt.datetime],
    *,
    maxcc: Optional[float] = None,
    data_collection: DataCollection = DataCollection.SENTINEL2_L1C,
    config: Optional[SHConfig] = None,
    time_difference: Optional[dt.timedelta] = None,
) -> List[dt.datetime]:
    """Get the list of timestamps from within date range for which data of the bbox is available."""
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
    """Downloaded specifid data"""

    client = SentinelHubDownloadClient(config=config)

    api_requests = []
    for timestamp in timestamps:
        request = SentinelHubRequest(
            evalscript=get_s2_evalscript(all_bands=all_bands, reflectance=False),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=(timestamp, timestamp),
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF),
                SentinelHubRequest.output_response("userdata", MimeType.JSON),
            ],
            bbox=bbox,
            size=size,
            resolution=resolution,
            config=config,
        )

        api_requests.append(request)

    responses = client.download([api_request.download_list[0] for api_request in api_requests])

    # pylint: disable=not-an-iterable
    data = np.asarray([response["default.tif"] for response in responses], dtype=np.float32)
    norm_factors = [response["userdata.json"]["norm_factor"] for response in responses]
    del responses

    normalized_bands = (np.round(array * factor, 4) for array, factor in zip(data[..., :-1], norm_factors))
    return np.asarray(list(normalized_bands), dtype=np.float32), data[..., -1] != 0
