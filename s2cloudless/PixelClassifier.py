"""
Module implementing pixel-based classifier
"""
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
        - predict(X)
        - predict_proba(X)

    This is true for all classifiers that follow scikit-learn's API.
    The APIs of scikit-learn's objects is described
    at: http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects.
    """
    # pylint: disable=invalid-name
    def __init__(self, classifier):
        """
        :param classifier: An instance of trained classifier that will be executed over an entire image
        :type classifier: Booster or object that implements methods predict and predict_proba
        """
        self._check_classifier(classifier)
        self.classifier = classifier

    @staticmethod
    def _check_classifier(classifier):
        """
        Checks if the classifier is of correct type or if it implements predict and predict_proba methods
        """
        if isinstance(classifier, Booster):
            return

        predict = getattr(classifier, 'predict', None)
        if not callable(predict):
            raise ValueError('Classifier does not have a predict method!')

        predict_proba = getattr(classifier, 'predict_proba', None)
        if not callable(predict_proba):
            raise ValueError('Classifier does not have a predict_proba method!')

    @staticmethod
    def extract_pixels(X):
        """ Extracts pixels from array X

        :param X: Array of images to be classified.
        :type X: numpy array, shape = [n_images, n_pixels_y, n_pixels_x, n_bands]
        :return: Reshaped 2D array
        :rtype: numpy array, [n_samples*n_pixels_y*n_pixels_x,n_bands]
        :raises: ValueError is input array has wrong dimensions
        """
        if len(X.shape) != 4:
            raise ValueError('Array of input images has to be a 4-dimensional array of shape'
                             '[n_images, n_pixels_y, n_pixels_x, n_bands]')

        new_shape = X.shape[0] * X.shape[1] * X.shape[2], X.shape[3]
        pixels = X.reshape(new_shape)
        return pixels

    def image_predict(self, X, **kwargs):
        """
        Predicts class labels for the entire image.

        :param X: Array of images to be classified.
        :type X: numpy array, shape = [n_images, n_pixels_y, n_pixels_x, n_bands]
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: raster classification map
        :rtype: numpy array, [n_samples, n_pixels_y, n_pixels_x]
        """
        pixels = self.extract_pixels(X)

        if isinstance(self.classifier, Booster):
            raise NotImplementedError('An instance of lightgbm.Booster can only return prediction probabilities, '
                                      'use PixelClassifier.image_predict_proba instead')

        predictions = self.classifier.predict(pixels, **kwargs)

        return predictions.reshape(X.shape[0], X.shape[1], X.shape[2])

    def image_predict_proba(self, X, **kwargs):
        """
        Predicts class probabilities for the entire image.

        :param X: Array of images to be classified.
        :type X: numpy array, shape = [n_images, n_pixels_y, n_pixels_x, n_bands]
        :param kwargs: Any keyword arguments that will be passed to the classifier's prediction method
        :return: classification probability map
        :rtype: numpy array, [n_samples, n_pixels_y, n_pixels_x]
        """
        pixels = self.extract_pixels(X)

        if isinstance(self.classifier, Booster):
            probabilities = self.classifier.predict(pixels, **kwargs)
            probabilities = np.vstack((1. - probabilities, probabilities)).transpose()
        else:
            probabilities = self.classifier.predict_proba(pixels, **kwargs)

        return probabilities.reshape(X.shape[0], X.shape[1], X.shape[2], probabilities.shape[1])
