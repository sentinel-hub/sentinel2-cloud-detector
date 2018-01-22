import numpy as np
import os.path
from .PixelClassifier import PixelClassifier
from skimage.morphology import disk, dilation
from scipy.ndimage.filters import convolve

from sklearn.externals import joblib


MODEL_FILENAME = 'pixel_s2_cloud_detector_lightGBM_v0.1.joblib.dat'
MODEL_EVALSCRIPT = 'return [B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12]'


class S2PixelCloudDetector(object):
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
            model_filename = os.path.join(package_dir, 'models', MODEL_FILENAME)

        self._load_classifier(model_filename)

        # indices of the 10 bands that are used by the classifier to
        # make the classification. This is used in case the input is
        # n x m x 13 (all bands)
        self.band_idxs = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]

        if average_over > 0:
            self.conv_filter = disk(average_over)/np.sum(disk(average_over))

        if dilation_size > 0:
            self.dilation_filter = disk(dilation_size)

    def _load_classifier(self, filename):
        """
        Loads the classifier.
        """
        self.classifier = PixelClassifier(joblib.load(filename))

    def get_cloud_probability_maps(self, X):
        """
        Runs the cloud detection on the input images (dimension n_images x n x m x 10
        or n_images x n x m x 13) and returns an array of cloud probability maps (dimension
        n_images x n x m). Pixel values close to 0 indicate clear-sky-like pixels, while
        values close to 1 indicate pixels covered with clouds.

        :param X: input Sentinel-2 image obtained with Sentinel-Hub's WMS/WCS request
                  (see https://github.com/sentinel-hub/sentinelhub-py)
        :type X: numpy array (shape n_images x n x m x 10 or n x m x 13)
        :return: cloud probability map
        :rtype: numpy array (shape n_images x n x m)
        """
        if self.all_bands:
            X = X[..., self.band_idxs]

        return self.classifier.image_predict_proba(X)[..., 1]

    def get_cloud_masks(self, X):
        """
        Runs the cloud detection on the input images (dimension n_images x n x m x 10
        or n_images x n x m x 13) and returns the raster cloud mask (dimension n_images x n x m).
        Pixel values equal to 0 indicate pixels classified as clear-sky, while values
        equal to 1 indicate pixels classified as clouds.

        :param X: input Sentinel-2 image obtained with Sentinel-Hub's WMS/WCS request
                  (see https://github.com/sentinel-hub/sentinelhub-py)
        :type X: numpy array (shape n_images x n x m x 10 or n x m x 13)
        :return: raster cloud mask
        :rtype: numpy array (shape n_images x n x m)
        """

        cloud_probs = self.get_cloud_probability_maps(X)

        if self.average_over:
            cloud_masks = np.asarray([convolve(cloud_prob, self.conv_filter) > self.threshold
                                      for cloud_prob in cloud_probs], dtype=np.int8)
        else:
            cloud_masks = (cloud_probs > self.threshold).astype(np.int8)

        if self.dilation_size:
            cloud_masks = np.asarray([dilation(cloud_mask, self.dilation_filter) for cloud_mask in cloud_masks],
                                     dtype=np.int8)

        return cloud_masks


class CloudMaskRequest:
    """
    Retrieves cloud probability maps of an area for all available dates in a date range.

    The user can then efficiently derive binary clouds masks based on a threshold.

    :param ogc_request: An instance of WmsRequest or WcsRequest (defined in sentinelhub-py package).
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
    :param all_bands: Tells the cloud detector that it will be passed all bands (instead of just the 10 that it needs).
                      Set to ``False`` by default.
    :type all_bands: bool
    """

    def __init__(self, ogc_request, *, threshold=0.4, average_over=4,
                 dilation_size=2, model_filename=None, all_bands=False):
        self.average_over = average_over
        self.dilation_size = dilation_size
        self.threshold = threshold

        self.cloud_detector = S2PixelCloudDetector(threshold=threshold, average_over=average_over, all_bands=all_bands,
                                                   dilation_size=dilation_size, model_filename=model_filename)
        self.ogc_bands_request = ogc_request

        self.bands = None
        self.probability_masks = None

    def __len__(self):
        return len(self.bands)

    def __iter__(self):
        self.get_probability_masks()
        cloud_masks = self.get_cloud_masks()
        return iter(
            [(self.probability_masks[idx], cloud_masks[idx], self.bands[idx]) for idx, _ in
             enumerate(self.bands)])

    def get_dates(self):
        """
        Get the list of dates from within date range for which data of the bbox is available.
        :return: List[datetime.datetime]
        """
        return self.ogc_bands_request.get_dates()

    def get_probability_masks(self):
        """
        Get probability maps of areas for each available date.
        :return: np.ndarray
        """
        if self.probability_masks is None:
            self.get_data()
            self.probability_masks = self.cloud_detector.get_cloud_probability_maps(np.array(self.bands))
        return self.probability_masks

    def get_data(self):
        """
        w-times-h rasters for all n dates.
        :return: np.ndarray
        """
        if self.bands is None:
            self.bands = self.ogc_bands_request.get_data()
        return self.bands

    def get_cloud_masks(self, threshold=None):
        """ The binary cloud mask is computed on the fly. Be cautious.

        :param threshold: A float from [0,1] specifying threshold
        :return: Binary cloud masks
        """
        self.get_probability_masks()
        threshold = self.threshold if threshold is None else threshold

        if self.average_over:
            cloud_masks = np.asarray(
                [convolve(cloud_prob, self.cloud_detector.conv_filter) > threshold for cloud_prob in
                 self.probability_masks], dtype=np.int8)
        else:
            cloud_masks = (self.probability_masks > threshold).astype(np.int8)

        if self.dilation_size:
            cloud_masks = np.asarray(
                [dilation(cloud_mask, self.cloud_detector.dilation_filter) for cloud_mask in cloud_masks],
                dtype=np.int8)

        return cloud_masks
