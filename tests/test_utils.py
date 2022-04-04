"""
Tests for utils.py module
"""
import pytest

from s2cloudless import get_s2_evalscript


@pytest.mark.parametrize("all_bands", [True, False])
@pytest.mark.parametrize("reflectance", [True, False])
def test_get_s2_evalscript(all_bands, reflectance):
    evalscript = get_s2_evalscript(all_bands=all_bands, reflectance=reflectance)

    assert isinstance(evalscript, str)

    bands_num_str = "bands: 14" if all_bands else "bands: 11"
    assert bands_num_str in evalscript

    input_units_str = 'units: "reflectance"' if reflectance else 'units: "DN"'
    assert input_units_str in evalscript
    output_sample_type_str = 'sampleType: "FLOAT32"' if reflectance else 'sampleType: "UINT16"'
    assert output_sample_type_str in evalscript
