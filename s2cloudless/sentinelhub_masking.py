"""
Module using sentinelhub-py to interact with Sentinel Hub services
"""
import datetime as dt

import numpy as np

from sentinelhub import (
    DataCollection,
    MimeType,
    SentinelHubCatalog,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    SHConfig,
    filter_times,
    parse_time_interval,
)

from .utils import get_s2_evalscript


class NoDataAvailableException(RuntimeError):
    """Raise in case there is no Sentinel-2 data available for give request"""


class CloudMaskRequest:
    """Obtains data from Sentinel Hub service and calculates cloud masks and probabilities for all dates in a time
    range
    """

    def __init__(
        self,
        cloud_detector,
        bbox,
        time,
        *,
        size=None,
        resolution=None,
        maxcc=None,
        time_difference=None,
        data_folder=None,
        data_collection=DataCollection.SENTINEL2_L1C,
        config=None,
        **kwargs
    ):
        """
        :param cloud_detector: An instance of a cloud detector object
        :type cloud_detector: S2PixelCloudDetector
        :param bbox: Bounding box describing the area of interest.
        :type bbox: sentinelhub.BBox
        :param time: A time interval of the request.
        :type time: str or (str, str) or datetime.date or (datetime.date, datetime.date) or datetime.datetime or
            (datetime.datetime, datetime.datetime)
        :param size: Size of the image.
        :type size: Tuple[int, int]
        :param resolution: Resolution of the image. It has to be in units compatible with the given CRS.
        :type resolution: Tuple[float, float]
        :param maxcc: Maximum accepted cloud coverage of an image. Float between 0.0 and 1.0. Default is 1.0.
        :type maxcc: float or None
        :param time_difference: A minimal time difference between timestamps for which data will be requested.
        :type time_difference: datetime.timedelta or None
        :param data_folder: A location of the directory where downloaded data will be saved.
        :type data_folder: str or None
        :param data_collection: A Sentinel-2 L1C collection from where data will be collected.
        :type data_collection: DataCollection
        :param config: An instance of config class to override parameters from the saved configuration.
        :type config: SHConfig or None
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

        self.timestamps = None
        self.bands = None
        self.data_mask = None
        self.probability_masks = None

        self.api_requests = self._prepare_api_requests()

    def __len__(self):
        """Provide a number of acquisitions (i.e. the same as number of cloud masks)"""
        return len(self.api_requests)

    def __iter__(self):
        """Iterate over probability masks, cloud masks and bands"""
        cloud_masks = self.get_cloud_masks()
        return zip(self.probability_masks, cloud_masks, self.bands)

    def _prepare_api_requests(self):
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
                        **self.kwargs
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

    def get_timestamps(self):
        """Get the list of timestamps from within date range for which data of the bbox is available.

        :return: A list of timestamps
        :rtype: list(datetime.datetime)
        """
        if self.timestamps is None:
            if isinstance(self.time, list) and all(isinstance(timestamp, dt.datetime) for timestamp in self.time):
                self.timestamps = self.time
            else:
                time_interval = parse_time_interval(self.time)
                self.timestamps = self._get_timestamps_from_catalog(time_interval)

            self.timestamps = filter_times(self.timestamps, self.time_difference)

        if not self.timestamps:
            raise NoDataAvailableException("There are no Sentinel-2 images available for given parameters")

        return self.timestamps

    def _get_timestamps_from_catalog(self, time_interval):
        """Collects a list of timestamps from Sentinel Hub Catalog API"""
        catalog = SentinelHubCatalog(config=self.config)

        cloud_cover_query = None
        if self.maxcc is not None:
            cloud_cover_query = {"eo:cloud_cover": {"lt": 100 * float(self.maxcc)}}

        search_iterator = catalog.search(
            self.data_collection,
            bbox=self.bbox,
            time=time_interval,
            query=cloud_cover_query,
            fields={
                "include": [
                    "properties.datetime",
                ],
                "exclude": [],
            },
        )
        return search_iterator.get_timestamps()

    def get_data(self):
        """Returns downloaded bands

        :return: numpy array of shape `(times, height, width, bands)`
        :rtype: numpy.ndarray
        """
        if self.bands is None:
            self._download_bands_and_valid_data_mask()
        return self.bands

    def get_data_mask(self):
        """Returns valid data mask.

        :return: numpy array of shape `(times, height, width)`
        :rtype: numpy.ndarray
        """
        if self.data_mask is None:
            self._download_bands_and_valid_data_mask()
        return self.data_mask

    def _download_bands_and_valid_data_mask(self):
        """Downloads band data and valid mask. Sets parameters self.bands, self.data_mask"""
        download_requests = [api_request.download_list[0] for api_request in self.api_requests]
        client = SentinelHubDownloadClient(config=self.config)

        responses = client.download(download_requests)

        data = np.asarray([response["default.tif"] for response in responses], dtype=np.float32)
        norm_factors = [response["userdata.json"]["norm_factor"] for response in responses]
        del responses

        self.bands = data[..., :-1]
        self.data_mask = data[..., -1] != 0

        normalized_bands = (np.round(array * factor, 4) for array, factor in zip(self.bands, norm_factors))
        self.bands = np.asarray(list(normalized_bands), dtype=np.float32)

    def get_probability_masks(self, non_valid_value=0):
        """
        Get probability maps of areas for each available date. The pixels without valid data are assigned
        non_valid_value.

        :param non_valid_value: Value to be assigned to non valid data pixels
        :type non_valid_value: float
        :return: Probability map of shape `(times, height, width)` and `dtype=numpy.float64`
        :rtype: numpy.ndarray
        """
        # pylint: disable=invalid-unary-operand-type
        if self.probability_masks is None:
            self.get_data()
            self.probability_masks = self.cloud_detector.get_cloud_probability_maps(self.bands)

        self.probability_masks[~self.data_mask] = non_valid_value
        return self.probability_masks

    def get_cloud_masks(self, threshold=None, non_valid_value=0):
        """The binary cloud mask is computed on the fly. Be cautious. The pixels without valid data are assigned
        non_valid_value.

        :param threshold: A float from [0,1] specifying threshold
        :type threshold: float
        :param non_valid_value: Value which will be assigned to pixels without valid data
        :type non_valid_value: int in range `[-254, 255]`
        :return: Binary cloud masks of shape `(times, height, width)` and `dtype=numpy.int8`
        :rtype: numpy.ndarray
        """
        # pylint: disable=invalid-unary-operand-type
        self.get_probability_masks()

        cloud_masks = self.cloud_detector.get_mask_from_prob(self.probability_masks, threshold)
        cloud_masks[~self.data_mask] = non_valid_value

        return cloud_masks
