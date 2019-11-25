import unittest
import numpy as np

from test_all import TestS2Cloudless

from sentinelhub import WmsRequest, WcsRequest, BBox, CRS, MimeType, CustomUrlParam, TestCaseContainer

from s2cloudless import S2PixelCloudDetector, CloudMaskRequest


class TestCloudDetector(TestS2Cloudless):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.templates = np.load(cls.INPUT_DATA_FILE)

    def test_cloud_detector(self):
        cloud_detector = S2PixelCloudDetector(all_bands=True)
        cloud_probs = cloud_detector.get_cloud_probability_maps(self.templates['s2_im'])
        cloud_mask = cloud_detector.get_cloud_masks(self.templates['s2_im'])
        self.assertTrue(np.array_equal(cloud_probs, self.templates['cl_probs']))
        self.assertTrue(np.array_equal(cloud_mask, self.templates['cl_mask']))

        cloud_detector = S2PixelCloudDetector(all_bands=False)
        self.assertRaises(ValueError, lambda: cloud_detector.get_cloud_probability_maps(self.templates['s2_im']))


class TestCloudMaskRequest(TestS2Cloudless):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        bbox1 = BBox([-90.9216499, 14.4190528, -90.8186531, 14.5520163], crs=CRS.WGS84)  # From examples
        bbox2 = BBox([46.16, -16.15, 46.51, -15.58], crs=CRS.WGS84)  # From sentinelhub-py examples
        cls.custom_url_params = {
            CustomUrlParam.SHOWLOGO: True,
            CustomUrlParam.TRANSPARENT: False,
            CustomUrlParam.EVALSCRIPT: 'return [B01]',
            CustomUrlParam.ATMFILTER: 'DOS1'
        }

        cls.wms_request = WmsRequest(layer='S2-DOS1', bbox=bbox1, time=('2017-12-01', '2017-12-31'), width=60,
                                     height=None, image_format=MimeType.TIFF, custom_url_params=cls.custom_url_params,
                                     instance_id=cls.CONFIG.instance_id)
        cls.wcs_request = WcsRequest(layer='S2-ATMCOR', bbox=bbox2, time='2016-07-18T07:14:04',
                                     resx='100m', resy='100m', image_format=MimeType.PNG, data_folder='.')

        cls.test_cases = [
            TestCaseContainer('WMS', CloudMaskRequest(cls.wms_request, threshold=0.6, average_over=2, dilation_size=5),
                              clm_min=0, clm_max=1, clm_mean=0.343827, clm_median=0, clp_min=0.00011, clp_max=0.99999,
                              clp_mean=0.23959, clp_median=0.01897, mask_shape=(7, 81, 60)),
            TestCaseContainer('WCS, partial no data', CloudMaskRequest(cls.wcs_request, all_bands=True), clm_min=0,
                              clm_max=1, clm_mean=0.04468, clm_median=0, clp_min=-50.0, clp_max=0.999635,
                              clp_mean=-7.5472468, clp_median=0.011568, mask_shape=(1, 634, 374))
        ]

    def test_get_cloud_masks(self):
        for test_case in self.test_cases:
            masks = test_case.request.get_cloud_masks()

            self.test_numpy_data(masks, exp_shape=test_case.mask_shape, exp_dtype=np.int8,
                                 exp_min=test_case.clm_min, exp_max=test_case.clm_max,
                                 exp_mean=test_case.clm_mean, exp_median=test_case.clm_median,
                                 test_name=test_case.name, delta=1e-4)

    def test_get_cloud_probabilities(self):
        for test_case in self.test_cases:
            prob_masks = test_case.request.get_probability_masks(non_valid_value=-50)

            self.test_numpy_data(prob_masks, exp_shape=test_case.mask_shape, exp_dtype=np.float64,
                                 exp_min=test_case.clp_min, exp_max=test_case.clp_max,
                                 exp_mean=test_case.clp_mean, exp_median=test_case.clp_median,
                                 test_name=test_case.name, delta=1e-4)

    def test_requests_unchanged(self):
        self.assertEqual(self.wms_request.custom_url_params, self.custom_url_params,
                         msg='Custom url params were changed')
        self.assertTrue(self.wms_request.custom_url_params[CustomUrlParam.SHOWLOGO],
                        msg='Custom url params were changed')

        self.assertEqual(self.wms_request.image_format, MimeType.TIFF, msg='Image format of WMS class was changed')
        self.assertEqual(self.wcs_request.image_format, MimeType.PNG, msg='Image format of WCS class was changed')


if __name__ == '__main__':
    unittest.main()
