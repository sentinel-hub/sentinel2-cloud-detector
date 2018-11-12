import unittest
import logging

import numpy as np

from s2cloudless.PixelClassifier import PixelClassifier
from s2cloudless.S2PixelCloudDetector import S2PixelCloudDetector

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(module)s:%(lineno)d [%(levelname)s] %(funcName)s  %(message)s')


class TestPixelClassifier(unittest.TestCase):

    def test_check_classifier(self):
        self.assertRaises(ValueError, PixelClassifier, "test")

    def test_extract_pixels(self):
        w, x, y, z = np.ones(5), np.ones((5, 5)), np.ones((5, 5, 5)), np.ones((5, 5, 5, 5))
        cloud_detector = S2PixelCloudDetector()
        self.assertRaisesRegex(ValueError,
                               "Array of input images has to be a 4-dimensional array of shape",
                               cloud_detector.classifier.extract_pixels, w)
        self.assertRaisesRegex(ValueError,
                               "Array of input images has to be a 4-dimensional array of shape",
                               cloud_detector.classifier.extract_pixels, x)
        self.assertRaisesRegex(ValueError,
                               "Array of input images has to be a 4-dimensional array of shape",
                               cloud_detector.classifier.extract_pixels, y)
        self.assertTrue(len(cloud_detector.classifier.extract_pixels(z).shape) == 2)
        self.assertTrue(cloud_detector.classifier.extract_pixels(z).shape[0] == 5*5*5)


if __name__ == '__main__':
    unittest.main()
