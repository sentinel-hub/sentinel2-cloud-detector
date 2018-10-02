"""
Module implementing pixel-based classifier
"""


class PixelClassifier:
    """
    Pixel classifier extends a receptive field of a classifier over an entire image.
    The classifier's receptive field is in case of PixelClassifier a pixel (i.e, it
    has dimension of (1,1))

    Pixel classifier divides the image into individual pixels, runs classifier over
    them, and finally produces a classification mask of the same size as image.

    The classifier can be of any type as long as it has the following two methods
    implemented:
        - predict(X)
        - predict_proba(X)

    This is true for all classifiers that follow scikit-learn's API.
    The APIs of scikit-learn's objects is described
    at: http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects.

    :param classifier: trained classifier that will be executed over an entire image
    :type classifier: any classifier with predict(X) and predict_proba(X) methods
    """
    # pylint: disable=invalid-name
    def __init__(self, classifier):
        self.receptive_field = (1, 1)
        self._check_classifier(classifier)
        self.classifier = classifier

    @staticmethod
    def _check_classifier(classifier):
        """
        Check if the classifier implements predict and predict_proba methods.
        """
        predict = getattr(classifier, "predict", None)
        if not callable(predict):
            raise ValueError('Classifier does not have a predict method!')

        predict_proba = getattr(classifier, "predict_proba", None)
        if not callable(predict_proba):
            raise ValueError('Classifier does not have a predict_proba method!')

    @staticmethod
    def extract_pixels(X):
        """ Extract pixels from array X

        :param X: Array of images to be classified.
        :type X: numpy array, shape = [n_images, n_pixels_y, n_pixels_x, n_bands]
        :return: Reshaped 2D array
        :rtype: numpy array, [n_samples*n_pixels_y*n_pixels_x,n_bands]
        :raises: ValueError is input array has wrong dimensions
        """
        if len(X.shape) != 4:
            raise ValueError('Array of input images has to be a 4-dimensional array of shape'
                             '[n_images, n_pixels_y, n_pixels_x, n_bands]')

        new_shape = (X.shape[0] * X.shape[1] * X.shape[2], X.shape[3],)
        pixels = X.reshape(new_shape)
        return pixels

    def image_predict(self, X):
        """
        Predicts class label for the entire image.

        :param X: Array of images to be classified.
        :type X: numpy array, shape = [n_images, n_pixels_y, n_pixels_x, n_bands]

        :return: raster classification map
        :rtype: numpy array, [n_samples, n_pixels_y, n_pixels_x]
        """

        pixels = self.extract_pixels(X)

        predictions = self.classifier.predict(pixels)

        return predictions.reshape(X.shape[0], X.shape[1], X.shape[2])

    def image_predict_proba(self, X):
        """
        Predicts class probabilities for the entire image.

        :param X: Array of images to be classified.
        :type X: numpy array, shape = [n_images, n_pixels_y, n_pixels_x, n_bands]

        :return: classification probability map
        :rtype: numpy array, [n_samples, n_pixels_y, n_pixels_x]
        """

        pixels = self.extract_pixels(X)

        probabilities = self.classifier.predict_proba(pixels)

        return probabilities.reshape(X.shape[0], X.shape[1], X.shape[2], probabilities.shape[1])
