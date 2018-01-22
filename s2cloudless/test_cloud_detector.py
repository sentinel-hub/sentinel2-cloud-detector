""" Python script to test cloud detector """
import logging
import os.path
import numpy as np
import matplotlib.pyplot as plt
from s2cloudless import S2PixelCloudDetector

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_sentinelhub_cloud_detector(display=False):
    """
    Runs the classification on the test scene and compares the results with the provided
    cloud probability map and probability mask. Used to verify the installation of the
    package.

    :param display: flag to turn on or off the display of results.
    :type display: bool, default is False
    """
    # Load arrays
    package_dir = os.path.dirname(__file__)
    templates_filename = os.path.join(package_dir, './TestInputs/input_arrays.npz')
    templates = np.load(templates_filename)

    # Classifier instance, image has all bands
    cloud_detector = S2PixelCloudDetector(all_bands=True)

    # Compute cloud probabilities
    cloud_probs = cloud_detector.get_cloud_probability_maps(templates['s2_im'])

    # Compute cloud mask
    cloud_mask = cloud_detector.get_cloud_masks(templates['s2_im'])

    # Check predicted results match templates
    probs_ok = np.isclose(cloud_probs, templates['cl_probs']).all()
    mask_ok = np.array_equal(cloud_mask, templates['cl_mask'])

    if not probs_ok:
        LOGGER.info('Test FAILED.\nCloud probabilities DO NOT match templates.')

    if not mask_ok:
            LOGGER.info('Test FAILED.\nCloud masks DO NOT match templates.')

    if mask_ok and probs_ok:
        LOGGER.info('Test OK.\nCloud probabilities and cloud masks match templates.')

    # Display results
    if display:
        fig = plt.figure(figsize=(20, 10))
        plt.subplot(131)
        plt.imshow(np.squeeze(templates['s2_im'])[:, :, [3, 2, 1]])
        plt.title('RGB image')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(np.squeeze(templates['s2_im'])[:, :, [3, 2, 1]])
        plt.imshow(np.squeeze(templates['cl_mask']), vmin=0, vmax=1, alpha=.4)
        plt.title('RGB image with template mask overlay')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(np.squeeze(templates['s2_im'])[:, :, [3, 2, 1]])
        plt.imshow(np.squeeze(cloud_mask),vmin=0, vmax=1, alpha=.4)
        plt.title('RGB image with predicted mask overlay')
        plt.axis('off')
        plt.show()
