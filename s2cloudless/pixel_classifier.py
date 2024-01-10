"""Module for pixel-based classifiers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from lightgbm import Booster


class ClassifierType(Protocol):
    """Defines the necessary classifier interface."""

    def predict(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=missing-function-docstring,invalid-name
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pylint: disable=missing-function-docstring,invalid-name
        ...


class PixelClassifier:
    """
    Applies a pixel based classifier over a stack of images.

    The classifier can be of a type that is explicitly supported (e.g. `lightgbm.Booster`) or of any class with the
    methods `predict` and `predict_proba`.

    This is true for all classifiers that follow scikit-learn's API, which is described at:
    http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects
    """

    def __init__(self, classifier: Booster | ClassifierType):
        """
        :param classifier: An instance of trained classifier to be executed over images
        """
        self._check_classifier(classifier)
        self.classifier = classifier

    @staticmethod
    def _check_classifier(classifier: Booster | ClassifierType) -> None:
        """Checks if the classifier is suitable."""
        if isinstance(classifier, Booster):
            return

        for method_name in ("predict", "predict_proba"):
            method = getattr(classifier, method_name, None)
            if not callable(method):
                raise ValueError(f"Classifier does not have a {method_name} method!")

    def image_predict(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predicts class labels for the entire image.

        :param data: Array of images of shape `(N, height, width, bands)` to be classified.
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: Raster classification map of shape `(N, height, width)`
        """
        if isinstance(self.classifier, Booster):
            raise NotImplementedError(
                "An instance of `lightgbm.Booster` can only return prediction probabilities, use the"
                "`PixelClassifier.image_predict_proba` instead."
            )

        pixels = data.reshape((-1, data.shape[-1]))
        predictions = self.classifier.predict(pixels, **kwargs)

        return predictions.reshape(*data.shape[:-1])

    def image_predict_proba(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predicts class probabilities for the entire image.

        :param data: Array of images of shape `(N, height, width, bands)` to be classified.
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: Classification probability map of shape `(N, height, width, n_classes)` where `n_classes` is 2 for
            cloud predictors
        """
        pixels = data.reshape((-1, data.shape[-1]))

        if isinstance(self.classifier, Booster):
            proba = self.classifier.predict(pixels, **kwargs)
            probabilities = np.vstack([1.0 - proba, proba]).transpose()  # type: ignore[operator, list-item]
        else:
            probabilities = self.classifier.predict_proba(pixels, **kwargs)

        return probabilities.reshape(*data.shape[:-1], probabilities.shape[-1])
