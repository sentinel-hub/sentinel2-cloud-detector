"""
Module using sentinelhub-py to interact with Sentinel Hub services
"""
import datetime as dt
from typing import Any, List, Optional, Tuple, Union

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
    parse_time_interval,
)
from sentinelhub.types import RawTimeIntervalType, RawTimeType

from s2cloudless.cloud_detector import S2PixelCloudDetector

from .utils import get_s2_evalscript

RawTimeIterableType = Union[RawTimeType, RawTimeIntervalType, List[RawTimeIntervalType]]


class NoDataAvailableException(RuntimeError):
    """Raise in case there is no Sentinel-2 data available for give request"""


class CloudMaskRequest:
    """Obtains data from Sentinel Hub service and calculates cloud masks and probabilities for all dates in a time
    range
    """

    def __init__(
        self,
        cloud_detector: S2PixelCloudDetector,
        bbox: BBox,
        time: RawTimeIterableType,
        *,
        size: Optional[Tuple[int, int]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        maxcc: Optional[float] = None,
        time_difference: Optional[dt.timedelta] = None,
        data_folder: Optional[str] = None,
        data_collection: DataCollection = DataCollection.SENTINEL2_L1C,
        config: Optional[SHConfig] = None,
        **kwargs: Any,
    ):
        """
        :param cloud_detector: An instance of a cloud detector object
        :param bbox: Bounding box describing the area of interest.
        :param time: A time interval of the request.
        :param size: Size of the image.
        :param resolution: Resolution of the image. It has to be in units compatible with the given CRS.
        :param maxcc: Maximum accepted cloud coverage of an image. Float between 0.0 and 1.0. Default is 1.0.
        :param time_difference: A minimal time difference between timestamps for which data will be requested.
        :param data_folder: A location of the directory where downloaded data will be saved.
        :param data_collection: A Sentinel-2 L1C collection from where data will be collected.
        :param config: An instance of config class to override parameters from the saved configuration.
        :param kwargs: Additional arguments to be passed to `SentinelHubRequest.input_data`, e.g. `upsampling` or
            `downsampling`.
        """
        self.cloud_detector = cloud_detector
        self.bbox = bbox
        self.time = time
        self.size = size
        self.resolution = resolution
        self.maxcc = maxcc
        self.time_difference = time_difference or dt.timedelta(seconds=0)
        self.data_folder = data_folder
        self.data_collection = DataCollection(data_collection)
        self.config = config.copy() if config else SHConfig()
        self.config.sh_base_url = self.data_collection.service_url
        self.kwargs = kwargs

        self.api_requests = self._prepare_api_requests()

    def __len__(self) -> int:
        """Provide a number of acquisitions (i.e. the same as number of cloud masks)"""
        return len(self.api_requests)

    def _prepare_api_requests(self) -> List[SentinelHubRequest]:
        """Prepare a list of Process API requests defining what data will be downloaded"""
        timestamps = self.get_timestamps()
        evalscript = get_s2_evalscript(all_bands=self.cloud_detector.all_bands, reflectance=False)

        api_requests = []
        for timestamp in timestamps:
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=self.data_collection,
                        time_interval=(timestamp - self.time_difference, timestamp + self.time_difference),
                        **self.kwargs,
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response("default", MimeType.TIFF),
                    SentinelHubRequest.output_response("userdata", MimeType.JSON),
                ],
                bbox=self.bbox,
                size=self.size,
                resolution=self.resolution,
                config=self.config,
                data_folder=self.data_folder,
            )
            api_requests.append(request)

        return api_requests

    def get_timestamps(self) -> List[dt.datetime]:
        """Get the list of timestamps from within date range for which data of the bbox is available.

        :return: A list of timestamps
        """
        timestamps: Optional[List[dt.datetime]] = None
        if isinstance(self.time, list):
            if all(isinstance(timestamp, dt.datetime) for timestamp in self.time):
                timestamps = self.time  # type:  ignore[assignment]
        else:
            time_interval = parse_time_interval(self.time)
            timestamps = self._get_timestamps_from_catalog(time_interval)

        if timestamps is None:
            raise ValueError("There are no available timestamps.")

        timestamps = filter_times(timestamps, self.time_difference)

        if not timestamps:
            raise NoDataAvailableException("There are no Sentinel-2 images available for given parameters")

        return timestamps

    def _get_timestamps_from_catalog(
        self, time_interval: Tuple[Optional[dt.datetime], Optional[dt.datetime]]
    ) -> List[dt.datetime]:
        """Collects a list of timestamps from Sentinel Hub Catalog API"""
        catalog = SentinelHubCatalog(config=self.config)

        cloud_cover_query = None
        if self.maxcc is not None:
            cloud_cover_query = f"eo:cloud_cover < {100 * float(self.maxcc)}"

        search_iterator = catalog.search(
            self.data_collection,
            bbox=self.bbox,
            time=time_interval,
            filter=cloud_cover_query,
            fields={
                "include": [
                    "properties.datetime",
                ],
                "exclude": [],
            },
        )
        return search_iterator.get_timestamps()

    def _download_bands_and_valid_data_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        """Downloads band data and valid mask. Return parameters bands, data_mask"""
        download_requests = [api_request.download_list[0] for api_request in self.api_requests]
        client = SentinelHubDownloadClient(config=self.config)

        responses = client.download(download_requests)

        # pylint: disable=not-an-iterable
        data = np.asarray([response["default.tif"] for response in responses], dtype=np.float32)
        norm_factors = [response["userdata.json"]["norm_factor"] for response in responses]
        del responses

        bands = data[..., :-1]
        data_mask = data[..., -1] != 0

        normalized_bands = (np.round(array * factor, 4) for array, factor in zip(bands, norm_factors))
        bands = np.asarray(list(normalized_bands), dtype=np.float32)
        return bands, data_mask

    def get_probability_masks(self, non_valid_value: int = 0) -> np.ndarray:
        """
        Get probability maps of areas for each available date. The pixels without valid data are assigned
        non_valid_value.

        :param non_valid_value: Value to be assigned to non valid data pixels
        :return: Probability map of shape `(times, height, width)` and `dtype=numpy.float64`
        """
        # pylint: disable=invalid-unary-operand-type

        bands, data_mask = self._download_bands_and_valid_data_mask()
        probability_masks = self.cloud_detector.get_cloud_probability_maps(bands)

        probability_masks[~data_mask] = non_valid_value
        return probability_masks

    def get_cloud_masks(self, threshold: Optional[float] = None, non_valid_value: int = 0) -> np.ndarray:
        """The binary cloud mask is computed on the fly. Be cautious. The pixels without valid data are assigned
        non_valid_value.

        :param threshold: A float from [0,1] specifying threshold
        :param non_valid_value: Value which will be assigned to pixels without valid data
        :return: Binary cloud masks of shape `(times, height, width)` and `dtype=numpy.int8`
        """
        # pylint: disable=invalid-unary-operand-type

        cloud_masks = self.cloud_detector.get_mask_from_prob(self.get_probability_masks(), threshold)
        _, data_mask = self._download_bands_and_valid_data_mask()

        cloud_masks[~data_mask] = non_valid_value

        return cloud_masks
