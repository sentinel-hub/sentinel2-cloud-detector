from setuptools import setup, find_packages

__version__ = '1.0.1'

setup(
    name='s2cloudless',
    version=__version__,
    description='Sentinel Hub\'s cloud detector for Sentinel-2 imagery',
    url='https://github.com/sentinel-hub/sentinel2-cloud-detector',
    author='Anze Zupanc',
    author_email='anze.zupanc@sinergise.com',
    license='CC-BY-SA-4.0',
    packages=find_packages('.'),
    package_dir={'': '.'},
    package_data={'s2cloudless': ['models/pixel_s2_cloud_detector_lightGBM_v0.1.joblib.dat',
                                  'TestInputs/input_arrays.npz']},
    install_requires=['numpy', 'scipy', 'scikit-learn', 'scikit-image', 'matplotlib', 'lightgbm'],
    zip_safe=False
)
