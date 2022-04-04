"""
Tests for pixel_classifier.py module
"""
import os

import numpy as np
import pytest
from lightgbm import Booster

import s2cloudless
from s2cloudless import PixelClassifier
from s2cloudless.cloud_detector import MODEL_FILENAME


@pytest.fixture(name="booster")
def booster_fixture():
    package_path = os.path.dirname(s2cloudless.__file__)
    model_path = os.path.join(package_path, "models", MODEL_FILENAME)
    return Booster(model_file=model_path)


@pytest.mark.parametrize(
    "input_array,expected_result",
    [
        (np.ones(5), None),
        (np.ones((5, 5)), None),
        (np.ones((5, 5, 5)), None),
        (np.ones((5, 5, 5, 5)), np.ones((5 * 5 * 5, 5))),
    ],
)
def test_extract_pixels(input_array, expected_result, booster):
    classifier = PixelClassifier(booster)

    if expected_result is None:
        with pytest.raises(ValueError):
            classifier.extract_pixels(input_array)
    else:
        result = classifier.extract_pixels(input_array)
        assert np.array_equal(result, expected_result)


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
    assert np.allclose(np.sum(result, axis=-1), np.ones((5, 5, 5)), atol=1e-12)
