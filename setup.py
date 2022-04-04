import io
import os

from setuptools import find_packages, setup


def get_version():
    for line in open(os.path.join(os.path.dirname(__file__), "s2cloudless", "__init__.py")):
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"').strip("'")
            return version


def get_long_description():
    return io.open("README.md", encoding="utf-8").read()


def parse_requirements(file):
    return sorted(
        set(line.partition("#")[0].strip() for line in open(os.path.join(os.path.dirname(__file__), file))) - set("")
    )


setup(
    name="s2cloudless",
    python_requires=">=3.7",
    version=get_version(),
    description="Sentinel Hub's cloud detector for Sentinel-2 imagery",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/sentinel-hub/sentinel2-cloud-detector",
    author="Sinergise EO research team",
    author_email="anze.zupanc@sinergise.com",
    license="CC BY-SA 4.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"s2cloudless": ["models/pixel_s2_cloud_detector_lightGBM_v0.1.txt"]},
    install_requires=parse_requirements("requirements.txt"),
    extras_require={"DEV": parse_requirements("requirements-dev.txt")},
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
)
