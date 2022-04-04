"""
Module for making pixel-based classification on Sentinel-2 L1C imagery
"""
import os

import numpy as np
from lightgbm import Booster
from scipy.ndimage.filters import convolve
from skimage.morphology import dilation, disk

from .pixel_classifier import PixelClassifier
from .utils import MODEL_BAND_IDS

MODEL_FILENAME = "pixel_s2_cloud_detector_lightGBM_v0.1.txt"


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

    def __init__(self, threshold=0.4, all_bands=False, average_over=1, dilation_size=1, model_filename=None):

        self.threshold = threshold
        self.all_bands = all_bands
        self.average_over = average_over
        self.dilation_size = dilation_size

        if model_filename is None:
            package_dir = os.path.dirname(__file__)
            model_filename = os.path.join(package_dir, "models", MODEL_FILENAME)
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

    def get_cloud_probability_maps(self, data, **kwargs):
        """
        Runs the cloud detection on the input images (dimension n_images x n x m x 10
        or n_images x n x m x 13) and returns an array of cloud probability maps (dimension
        n_images x n x m). Pixel values close to 0 indicate clear-sky-like pixels, while
        values close to 1 indicate pixels covered with clouds.

        :param data: A stack of Sentinel-2 images with all required bands in the correct order
        :type data: numpy array (shape n_images x n x m x 10 or n x m x 13)
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: cloud probability map
        :rtype: numpy array (shape n_images x n x m)
        """
        is_single_temporal = data.ndim == 3
        if is_single_temporal:
            data = data[np.newaxis, ...]

        band_num = data.shape[-1]
        exp_bands = 13 if self.all_bands else len(MODEL_BAND_IDS)
        if band_num != exp_bands:
            raise ValueError(
                f"Parameter 'all_bands' is set to {self.all_bands}. Therefore expected band data with "
                f"{exp_bands} bands, got {band_num} bands"
            )

        if self.all_bands:
            data = data[..., MODEL_BAND_IDS]

        proba = self.classifier.image_predict_proba(data, **kwargs)[..., 1]

        if is_single_temporal:
            return proba[0]
        return proba

    def get_cloud_masks(self, data, **kwargs):
        """
        Runs the cloud detection on the input images (dimension n_images x n x m x 10
        or n_images x n x m x 13) and returns the raster cloud mask (dimension n_images x n x m).
        Pixel values equal to 0 indicate pixels classified as clear-sky, while values
        equal to 1 indicate pixels classified as clouds.

        :param data: A stack of Sentinel-2 images with all required bands in the correct order
        :type data: numpy array (shape n_images x n x m x 10 or n x m x 13)
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: raster cloud mask
        :rtype: numpy array (shape n_images x n x m)
        """
        is_single_temporal = data.ndim == 3
        if is_single_temporal:
            data = data[np.newaxis, ...]

        cloud_probs = self.get_cloud_probability_maps(data, **kwargs)
        cloud_masks = self.get_mask_from_prob(cloud_probs)

        if is_single_temporal:
            return cloud_masks[0]
        return cloud_masks

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
            cloud_masks = np.asarray(
                [convolve(cloud_prob, self.conv_filter) > threshold for cloud_prob in cloud_probs], dtype=np.int8
            )
        else:
            cloud_masks = (cloud_probs > threshold).astype(np.int8)

        if self.dilation_size:
            cloud_masks = np.asarray(
                [dilation(cloud_mask, self.dilation_filter) for cloud_mask in cloud_masks], dtype=np.int8
            )
        return cloud_masks
