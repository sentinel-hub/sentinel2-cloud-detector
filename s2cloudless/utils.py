"""
Module with utilities
"""

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
MODEL_BAND_IDS = [0, 1, 3, 4, 7, 8, 9, 10, 11, 12]

DATA_EVALSCRIPT_TEMPLATE = """
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: [{bands}],
      units: "{input_units}"
    }}],
    output: {{
      bands: {band_number},
      sampleType: "{output_sample_type}"
    }}
  }};
}}
{metadata_evalscript}
function evaluatePixel(sample) {{
  return [{sample_bands}];
}}
"""

METADATA_EVALSCRIPT_TEMPLATE = """
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {
    "norm_factor":  inputMetadata.normalizationFactor
  }
}
"""


def get_s2_evalscript(all_bands=False, reflectance=False):
    """Provides an evalscript to download Sentinel-2 data

    :param all_bands: If `True` the evalscript will use all bands. Otherwise it will use only bands needed for cloud
        masking
    :type all_bands: bool
    :param reflectance: If `True` the evalscript will define reflectance values. Otherwise it will define digital
        numbers together with normalization factors to rescale them.
    :type reflectance: bool
    :return: An evalscript
    :rtype: str
    """
    bands = S2_BANDS
    if not all_bands:
        bands = [bands[index] for index in MODEL_BAND_IDS]
    bands = bands + ["dataMask"]

    return DATA_EVALSCRIPT_TEMPLATE.format(
        bands=", ".join(f'"{band}"' for band in bands),
        sample_bands=", ".join(f"sample.{band}" for band in bands),
        band_number=len(bands),
        input_units="reflectance" if reflectance else "DN",
        output_sample_type="FLOAT32" if reflectance else "UINT16",
        metadata_evalscript="" if reflectance else METADATA_EVALSCRIPT_TEMPLATE,
    ).strip("\n ")
