"""
Tests for pixel_classifier.py module
"""
import os

import numpy as np
import pytest
from lightgbm import Booster
from numpy.testing import assert_allclose, assert_array_equal

import s2cloudless
from s2cloudless import PixelClassifier
from s2cloudless.cloud_detector import MODEL_FILENAME


@pytest.fixture(name="booster")
def booster_fixture():
    package_path = os.path.dirname(s2cloudless.__file__)
    model_path = os.path.join(package_path, "models", MODEL_FILENAME)
    return Booster(model_file=model_path)


@pytest.mark.parametrize("input_array", [np.ones(5), np.ones((5, 5)), np.ones((5, 5, 5))])
def test_extract_pixels_invalid_input(input_array, booster):
    classifier = PixelClassifier(booster)

    with pytest.raises(ValueError):
        classifier.extract_pixels(input_array)


def test_extract_pixels(booster):
    classifier = PixelClassifier(booster)

    result = classifier.extract_pixels(np.ones((5, 5, 5, 5)))
    assert_array_equal(result, np.ones((5 * 5 * 5, 5)))


def test_image_predict(booster):
    classifier = PixelClassifier(booster)
    array = np.ones((5, 5, 5, 5))

    with pytest.raises(NotImplementedError):
        classifier.image_predict(array)


def test_image_predict_proba(booster):
    classifier = PixelClassifier(booster)
    array = np.random.rand(5, 5, 5, 10)

    result = classifier.image_predict_proba(array)

    assert result.shape == (5, 5, 5, 2)
    assert result.dtype == np.float64
    assert_allclose(np.sum(result, axis=-1), np.ones((5, 5, 5)), rtol=1e-5)
