[![Package version](https://badge.fury.io/py/s2cloudless.svg)](https://pypi.org/project/s2cloudless/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/s2cloudless.svg?style=flat-square)](https://pypi.org/project/s2cloudless/)
[![Build status](https://travis-ci.org/sentinel-hub/sentinel2-cloud-detector.svg?branch=master)](https://travis-ci.org/sentinel-hub/sentinel2-cloud-detector)
[![Overall downloads](https://pepy.tech/badge/s2cloudless)](https://pepy.tech/project/s2cloudless)
[![Last month downloads](https://pepy.tech/badge/s2cloudless/month)](https://pepy.tech/project/s2cloudless)

# Sentinel Hub's cloud detector for Sentinel-2 imagery

The **s2cloudless** Python package provides automated cloud detection in
Sentinel-2 imagery. The classification is based on a *single-scene pixel-based cloud detector*
developed by Sentinel Hub's research team and is described in more details
[in this blog](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13).

## Installation

The package requires a Python version >= 3.5. The package is available on
the PyPI package manager and can be installed with

```
$ pip install s2cloudless
```

To install the package manually, clone the repository and
```
$ python setup.py build
$ python setup.py install
```

## Input: Sentinel-2 scenes

The input to cloud detector are Sentinel-2 images. In particular, the cloud detector requires the following 10
Sentinel-2 band reflectances: B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12, which are obtained from raw
reflectance value in the following way: `B_i/10000`.

You don't need to worry about any of this, if you're doing classification of scenes obtained using Sentinel Hub's
WMS or WCS services (i.e. using ours Python library [sentinelhub-py](https://github.com/sentinel-hub/sentinelhub-py)).

## Test

Please test the cloud detector after the installation by performing a classification on a test scene provided with
this package. To execute it do the following:

```Python
>>> import s2cloudless
>>> s2cloudless.test_sentinelhub_cloud_detector()
```

In case your installation is fine and cloud detector works properly you should get the following output:

```
INFO:s2cloudless.test_cloud_detector:Test OK.
Cloud probabilities and cloud masks match templates.
```

## Examples

Jupyter notebook on how to use the cloud detector to produce cloud mask or cloud probability map
can be found in the [examples folder](https://github.com/sentinel-hub/sentinel2-cloud-detector/tree/master/examples).

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>
<br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
