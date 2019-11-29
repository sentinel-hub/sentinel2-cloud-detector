"""
Module for making pixel-based classification on Sentinel-2 L1C imagery
"""

import copy
import os
import numpy as np

from scipy.ndimage.filters import convolve
from skimage.morphology import disk, dilation
from lightgbm import Booster

from sentinelhub import CustomUrlParam, MimeType

from .PixelClassifier import PixelClassifier


MODEL_FILENAME = 'pixel_s2_cloud_detector_lightGBM_v0.1.txt'

MODEL_EVALSCRIPT = 'return [B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12]'
S2_BANDS_EVALSCRIPT = 'return [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12]'


class S2PixelCloudDetector:
    """
    Sentinel Hub's pixel-based cloud detector for Sentinel-2 imagery.

    Classifier takes as an input Sentinel-2 image of shape n x m x 13 (all 13 bands)
    or n x m x 10 (bands 1, 2, 4, 5, 8, 8A, 9, 10, 11, 12) and returns a raster
    binary cloud mask of shape n x m, where 0 (1) indicates clear sky (cloudy) pixel.
    The classifier can instead of a raster cloud mask return a cloud probability map
    of shape n x m, where each pixel's value is bound between 0 (clear-sky-like pixel)
    and 1 (cloud-like pixel).

    User can control cloud probability threshold and/or post-processing steps -
    convolution with disk (with user defined filter size) and dilation with disk
    (with user defined filter size).

    :param threshold: Cloud probability threshold. All pixels with cloud probability above
                      threshold value are masked as cloudy pixels. Default is 0.4.
    :type threshold: float
    :param all_bands: Flag specifying that input images will consists of all 13 Sentinel-2 bands.
    :type all_bands: bool
    :param average_over: Size of the disk in pixels for performing convolution (averaging probability
                         over pixels). Value 0 means do not perform this post-processing step.
                         Default is 1.
    :type average_over: int
    :param dilation_size: Size of the disk in pixels for performing dilation. Value 0 means do not perform
                          this post-processing step. Default is 1.
    :type dilation_size: int
    :param model_filename: Location of the serialised model. If None the default model provided with the
                           package is loaded.
    :type model_filename: str or None
    """
    BAND_IDXS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]

    # pylint: disable=invalid-name
    def __init__(self, threshold=0.4, all_bands=False, average_over=1, dilation_size=1, model_filename=None):

        self.threshold = threshold
        self.all_bands = all_bands
        self.average_over = average_over
        self.dilation_size = dilation_size

        if model_filename is None:
            package_dir = os.path.dirname(__file__)
            model_filename = os.path.join(package_dir, 'models', MODEL_FILENAME)
        self.model_filename = model_filename

        self._classifier = None

        if average_over > 0:
            self.conv_filter = disk(average_over) / np.sum(disk(average_over))

        if dilation_size > 0:
            self.dilation_filter = disk(dilation_size)

    @property
    def classifier(self):
        """
        Provides a classifier object. It also loads it if it hasn't been loaded yet. This way the classifier is loaded
        only when it is actually required.
        """
        if self._classifier is None:
            self._classifier = PixelClassifier(Booster(model_file=self.model_filename))

        return self._classifier

    def get_cloud_probability_maps(self, X, **kwargs):
        """
        Runs the cloud detection on the input images (dimension n_images x n x m x 10
        or n_images x n x m x 13) and returns an array of cloud probability maps (dimension
        n_images x n x m). Pixel values close to 0 indicate clear-sky-like pixels, while
        values close to 1 indicate pixels covered with clouds.

        :param X: input Sentinel-2 image obtained with Sentinel-Hub's WMS/WCS request
                  (see https://github.com/sentinel-hub/sentinelhub-py)
        :type X: numpy array (shape n_images x n x m x 10 or n x m x 13)
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: cloud probability map
        :rtype: numpy array (shape n_images x n x m)
        """
        band_num = X.shape[-1]
        exp_bands = 13 if self.all_bands else len(self.BAND_IDXS)
        if band_num != exp_bands:
            raise ValueError("Parameter 'all_bands' is set to {}. Therefore expected band data with {} bands, "
                             "got {} bands".format(self.all_bands, exp_bands, band_num))

        if self.all_bands:
            X = X[..., self.BAND_IDXS]

        return self.classifier.image_predict_proba(X, **kwargs)[..., 1]

    def get_cloud_masks(self, X, **kwargs):
        """
        Runs the cloud detection on the input images (dimension n_images x n x m x 10
        or n_images x n x m x 13) and returns the raster cloud mask (dimension n_images x n x m).
        Pixel values equal to 0 indicate pixels classified as clear-sky, while values
        equal to 1 indicate pixels classified as clouds.

        :param X: input Sentinel-2 image obtained with Sentinel-Hub's WMS/WCS request
                  (see https://github.com/sentinel-hub/sentinelhub-py)
        :type X: numpy array (shape n_images x n x m x 10 or n x m x 13)
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: raster cloud mask
        :rtype: numpy array (shape n_images x n x m)
        """

        cloud_probs = self.get_cloud_probability_maps(X, **kwargs)

        return self.get_mask_from_prob(cloud_probs)

    def get_mask_from_prob(self, cloud_probs, threshold=None):
        """
        Returns cloud mask by applying morphological operations -- convolution and dilation --
        to input cloud probabilities.

        :param cloud_probs: cloud probability map
        :type cloud_probs: numpy array of cloud probabilities (shape n_images x n x m)
        :param threshold: A float from [0,1] specifying threshold
        :type threshold: float
        :return: raster cloud mask
        :rtype: numpy array (shape n_images x n x m)
        """
        threshold = self.threshold if threshold is None else threshold

        if self.average_over:
            cloud_masks = np.asarray([convolve(cloud_prob, self.conv_filter) > threshold
                                      for cloud_prob in cloud_probs], dtype=np.int8)
        else:
            cloud_masks = (cloud_probs > threshold).astype(np.int8)

        if self.dilation_size:
            cloud_masks = np.asarray([dilation(cloud_mask, self.dilation_filter) for cloud_mask in cloud_masks],
                                     dtype=np.int8)
        return cloud_masks


class CloudMaskRequest:
    """
    Retrieves cloud probability maps of an area for all available dates in a date range.

    The user can then efficiently derive binary clouds masks based on a threshold.

    :param ogc_request: An instance of WmsRequest or WcsRequest (defined in sentinelhub-py package). The cloud mask
                        request creates a copy of this request and sets two custom url parameters: turns of the logo and
                        adds a request for transparency layer, which is used to determine the (non) valid data pixels.
    :type ogc_request: data_request.WmsRequest or data_request.WcsRequest
    :param threshold: Defines cloud and non-cloud in the binary cloud mask.
    :type threshold: float
    :param average_over: The size of neighborhood in averaging of probabilities in the
                         postprocessing step of the cloud detector.
    :type average_over: int
    :param dilation_size: The size of the structural element with which we dilate in the postprocessing.
    :type dilation_size: int
    :param model_filename: Location of the serialised model. If None the default model provided with the package
                           is loaded.
    :type model_filename: str or None
    :param all_bands: If ``True`` all S-2 bands will be downloaded and if ``False`` only required bands will
        be downloaded. In both cases only required bands will be used for further processing.
    :type all_bands: bool
    """
    # pylint: disable=invalid-unary-operand-type
    def __init__(self, ogc_request, *, threshold=0.4, average_over=4, dilation_size=2, model_filename=None,
                 all_bands=False):
        self.threshold = threshold
        self.average_over = average_over
        self.dilation_size = dilation_size
        self.all_bands = all_bands

        self.cloud_detector = S2PixelCloudDetector(threshold=threshold, average_over=average_over, all_bands=all_bands,
                                                   dilation_size=dilation_size, model_filename=model_filename)

        self.ogc_request = copy.deepcopy(ogc_request)
        self._prepare_ogc_request_params()

        self.bands = None
        self.probability_masks = None
        self.valid_data = None

    def _prepare_ogc_request_params(self):
        """ Method makes sure that correct parameters will be used for download of S-2 bands.
        """
        self.ogc_request.image_format = MimeType.TIFF_d32f
        if self.ogc_request.custom_url_params is None:
            self.ogc_request.custom_url_params = {}
        self.ogc_request.custom_url_params.update({
            CustomUrlParam.SHOWLOGO: False,
            CustomUrlParam.TRANSPARENT: True,
            CustomUrlParam.EVALSCRIPT: S2_BANDS_EVALSCRIPT if self.all_bands else MODEL_EVALSCRIPT,
            CustomUrlParam.ATMFILTER: 'NONE'
        })
        self.ogc_request.create_request(reset_wfs_iterator=False)

    def __len__(self):
        return len(self.get_dates())

    def __iter__(self):
        self.get_probability_masks()
        cloud_masks = self.get_cloud_masks()
        return iter(
            [(self.probability_masks[idx], cloud_masks[idx], self.bands[idx]) for idx, _ in enumerate(self.bands)]
        )

    def get_dates(self):
        """ Get the list of dates from within date range for which data of the bbox is available.

        :return: A list of dates
        :rtype: list(datetime.datetime)
        """
        return self.ogc_request.get_dates()

    def get_data(self):
        """ Returns downloaded bands

        :return: numpy array of shape `(times, height, width, bands)`
        :rtype: numpy.ndarray
        """
        if self.bands is None:
            self._set_band_and_valid_mask()
        return self.bands

    def get_valid_data(self):
        """ Returns valid data mask.

        :return: numpy array of shape `(times, height, width)`
        :rtype: numpy.ndarray
        """
        if self.valid_data is None:
            self._set_band_and_valid_mask()
        return self.valid_data

    def _set_band_and_valid_mask(self):
        """ Downloads band data and valid mask. Sets parameters self.bands, self.valid_data
        """
        data = np.asarray(self.ogc_request.get_data())
        self.bands = data[..., :-1]
        self.valid_data = (data[..., -1] == 1.0).astype(np.bool)

    def get_probability_masks(self, non_valid_value=0):
        """
        Get probability maps of areas for each available date. The pixels without valid data are assigned
        non_valid_value.

        :param non_valid_value: Value to be assigned to non valid data pixels
        :type non_valid_value: float
        :return: Probability map of shape `(times, height, width)` and `dtype=numpy.float64`
        :rtype: numpy.ndarray
        """
        if self.probability_masks is None:
            self.get_data()
            self.probability_masks = self.cloud_detector.get_cloud_probability_maps(self.bands)

        self.probability_masks[~self.valid_data] = non_valid_value
        return self.probability_masks

    def get_cloud_masks(self, threshold=None, non_valid_value=False):
        """ The binary cloud mask is computed on the fly. Be cautious. The pixels without valid data are assigned
        non_valid_value.

        :param threshold: A float from [0,1] specifying threshold
        :type threshold: float
        :param non_valid_value: Value which will be assigned to pixels without valid data
        :type non_valid_value: int in range `[-254, 255]`
        :return: Binary cloud masks of shape `(times, height, width)` and `dtype=numpy.int8`
        :rtype: numpy.ndarray
        """
        self.get_probability_masks()

        cloud_masks = self.cloud_detector.get_mask_from_prob(self.probability_masks, threshold)
        cloud_masks[~self.valid_data] = non_valid_value

        return cloud_masks
