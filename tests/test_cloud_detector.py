import unittest
import logging

import numpy as np

import os

from s2cloudless.S2PixelCloudDetector import S2PixelCloudDetector

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(module)s:%(lineno)d [%(levelname)s] %(funcName)s  %(message)s')


class TestCloudDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.wrk_dir = os.path.dirname(os.path.realpath(__file__))
        cls.input_dir = 'TestInputs'

    def test_cloud_detector(self):
        templates = np.load(os.path.join(self.wrk_dir, '..', 's2cloudless', self.input_dir, 'input_arrays.npz'))
        cloud_detector = S2PixelCloudDetector(all_bands=True)
        cloud_probs = cloud_detector.get_cloud_probability_maps(templates['s2_im'])
        cloud_mask = cloud_detector.get_cloud_masks(templates['s2_im'])
        self.assertTrue(np.isclose(cloud_probs, templates['cl_probs']).all())
        self.assertTrue(np.array_equal(cloud_mask, templates['cl_mask']))


if __name__ == '__main__':
    unittest.main()
