"""Module for pixel-based classification on Sentinel-2 L1C imagery."""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
from lightgbm import Booster

from .pixel_classifier import PixelClassifier
from .utils import MODEL_BAND_IDS, cv2_disk

MODEL_FILENAME = "pixel_s2_cloud_detector_lightGBM_v0.1.txt"


class S2PixelCloudDetector:
    """
    Sentinel Hub's pixel-based cloud detector for Sentinel-2 imagery.

    Classifier takes as an input Sentinel-2 images of shape `(N, height, width, 13)` (`N` images with all 13 bands) or
    `(N, height, width, 10)` (bands 1, 2, 4, 5, 8, 8A, 9, 10, 11, 12) and returns a binary cloud mask of shape
    `(N, height, width)`, where 0 indicates clear sky and 1 indicates clouds.

    The classifier can also return a cloud probability map of shape `(N, height, width)`, where each pixel's value
    is bound between 0 (clear-sky-like pixel) and 1 (cloud-like pixel).

    User can control cloud probability threshold and/or post-processing steps - convolution with disk (with user defined
    filter size) and dilation with disk (with user defined filter size).

    :param threshold: Cloud probability threshold. All pixels with cloud probability above threshold value are masked
        as cloudy pixels. Default is 0.4.
    :param all_bands: Flag specifying that input images will consists of all 13 Sentinel-2 bands.
    :param average_over: Size of the disk in pixels for performing convolution (averaging probability over pixels).
        Value `None` means do not perform this post-processing step. Default is 1.
    :param dilation_size: Size of the disk in pixels for performing dilation. Value `None` means it does not perform
        this post-processing step. Default is 1.
    :param model_filename: Location of the serialized model. If `None` the default model provided with the package is
        loaded.
    """

    def __init__(
        self,
        threshold: float = 0.4,
        all_bands: bool = False,
        average_over: int | None = 1,
        dilation_size: int | None = 1,
        model_filename: str | None = None,
    ):
        self.threshold = threshold
        self.all_bands = all_bands
        self.average_over = average_over
        self.dilation_size = dilation_size

        if model_filename is None:
            package_dir = os.path.dirname(__file__)
            model_filename = os.path.join(package_dir, "models", MODEL_FILENAME)
        self.model_filename = model_filename

        self._classifier: PixelClassifier | None = None

        if average_over is not None and average_over > 0:
            disk = cv2_disk(average_over)
            self.conv_filter = disk / np.sum(disk)

        if dilation_size is not None and dilation_size > 0:
            self.dilation_filter = cv2_disk(dilation_size)

    @property
    def classifier(self) -> PixelClassifier:
        """Provides a classifier object by utilizing lazy-loading to avoid multiple IO operations."""
        if self._classifier is None:
            self._classifier = PixelClassifier(Booster(model_file=self.model_filename))

        return self._classifier

    @staticmethod
    def _check_data_dimension(data: np.ndarray, correct_dimension: int) -> None:
        if data.ndim != correct_dimension:
            msg = (
                f"Data should be of dimension {correct_dimension}. Single-image data can be adjusted by using"
                " `data[np.newaxis, ...]`."
            )
            raise ValueError(msg)

    def get_cloud_probability_maps(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Runs the cloud detection on the input images of shape `(N, height, width, 13)` (all 13 bands) or
        `(N, height, width, 10)` (bands 1, 2, 4, 5, 8, 8A, 9, 10, 11, 12) and returns a cloud probability map of shape
        `(N, height, width)`, where values near 0 indicate high probability of clear sky and values near 1 indicate
        high probability of clouds.

        :param data: A stack of Sentinel-2 images with all required bands in the correct order
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: cloud probability map of shape `(N, height, width)`
        """

        self._check_data_dimension(data, 4)
        band_num = data.shape[-1]
        exp_bands = 13 if self.all_bands else len(MODEL_BAND_IDS)
        if band_num != exp_bands:
            raise ValueError(f"Parameter `all_bands` is set to {self.all_bands}, but images have {band_num} bands.")

        if self.all_bands:
            data = data[..., MODEL_BAND_IDS]

        proba = self.classifier.image_predict_proba(data, **kwargs)[..., 1]

        return proba.astype(np.float32)

    def get_cloud_masks(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Runs the cloud detection on the input images of shape `(N, height, width, 13)` (all 13 bands) or
        `(N, height, width, 10)` (bands 1, 2, 4, 5, 8, 8A, 9, 10, 11, 12) and returns a cloud mask of shape
        `(N, height, width)`, where 0 indicates clear sky and 1 indicates clouds.

        :param data: A stack of Sentinel-2 images with all required bands in the correct order
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: raster cloud mask of shape `(N, height, width)`
        """
        self._check_data_dimension(data, 4)
        cloud_probs = self.get_cloud_probability_maps(data, **kwargs)

        return self.get_mask_from_prob(cloud_probs)

    def get_mask_from_prob(self, cloud_probs: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Returns cloud mask by applying convolution and dilation to cloud probabilities.

        :param cloud_probs: cloud probability map of shape `(N, height, width)`
        :param threshold: A float from [0,1] specifying the probability threshold for mask creation
        :return: cloud mask of shape `(N, height, width)`
        """
        self._check_data_dimension(cloud_probs, 3)
        threshold = self.threshold if threshold is None else threshold

        if self.average_over:
            cloud_masks = np.asarray(
                [
                    cv2.filter2D(cloud_prob, -1, self.conv_filter, borderType=cv2.BORDER_REFLECT) > threshold
                    for cloud_prob in cloud_probs
                ],
                dtype=np.uint8,
            )
        else:
            cloud_masks = (cloud_probs > threshold).astype(np.int8)

        if self.dilation_size:
            cloud_masks = np.asarray(
                [cv2.dilate(cloud_mask, self.dilation_filter) for cloud_mask in cloud_masks], dtype=np.uint8
            )

        return cloud_masks
