import unittest
import os

from sentinelhub import TestSentinelHub


class TestS2Cloudless(TestSentinelHub):

    INPUT_DATA_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 's2cloudless',
                                   'TestInputs', 'input_arrays.npz')


if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(os.path.realpath(__file__)))
    runner = unittest.TextTestRunner()
    runner.run(suite)
