"""
Module implementing pixel-based classifier
"""
from typing import Any

import numpy as np
from lightgbm import Booster


class PixelClassifier:
    """
    Pixel classifier extends a receptive field of a classifier over an entire image.
    The classifier's receptive field is in case of PixelClassifier a pixel (i.e, it
    has dimension of (1,1))

    Pixel classifier divides the image into individual pixels, runs classifier over
    them, and finally produces a classification mask of the same size as image.

    The classifier can be of a type that is explicitly supported (e.g. lightgbm.Booster) or of any type as long as
    it has the following two methods implemented:
        - predict(data)
        - predict_proba(data)

    This is true for all classifiers that follow scikit-learn's API.
    The APIs of scikit-learn's objects is described
    at: http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects.
    """

    def __init__(self, classifier: Any):
        """
        :param classifier: An instance of trained classifier that will be executed over an entire image
        """
        self._check_classifier(classifier)
        self.classifier = classifier

    @staticmethod
    def _check_classifier(classifier: Any) -> None:
        """
        Checks if the classifier is of correct type or if it implements predict and predict_proba methods
        """
        if isinstance(classifier, Booster):
            return

        for method_name in ("predict", "predict_proba"):
            method = getattr(classifier, method_name, None)
            if not callable(method):
                raise ValueError(f"Classifier does not have a {method_name} method!")

    @staticmethod
    def extract_pixels(data: np.ndarray) -> np.ndarray:
        """Extracts pixels from data array

        :param data: Array of images to be classified.
        :return: Reshaped 2D array
        :raises: ValueError is input array has wrong dimensions
        """
        if len(data.shape) != 4:
            raise ValueError(
                "Array of input images has to be a 4-dimensional array of shape"
                "[n_images, n_pixels_y, n_pixels_x, n_bands]"
            )

        return data.reshape((-1, data.shape[3]))

    def image_predict(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predicts class labels for the entire image.

        :param data: Array of images to be classified.
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: raster classification map
        """
        if isinstance(self.classifier, Booster):
            raise NotImplementedError(
                "An instance of lightgbm.Booster can only return prediction probabilities, "
                "use PixelClassifier.image_predict_proba instead"
            )

        pixels = self.extract_pixels(data)
        predictions = self.classifier.predict(pixels, **kwargs)

        return predictions.reshape(*data.shape[0:3])

    def image_predict_proba(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predicts class probabilities for the entire image.

        :param data: Array of images to be classified.
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: classification probability map
        """
        pixels = self.extract_pixels(data)

        if isinstance(self.classifier, Booster):
            probabilities = self.classifier.predict(pixels, **kwargs)
            probabilities = np.vstack((1.0 - probabilities, probabilities)).transpose()
        else:
            probabilities = self.classifier.predict_proba(pixels, **kwargs)

        return probabilities.reshape(*data.shape[0:3], probabilities.shape[-1])
