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
    """Input array has to be a 4-dimensional."""
    classifier = PixelClassifier(booster)

    with pytest.raises(ValueError):
        classifier.extract_pixels(input_array)


def test_extract_pixels(booster):
    classifier = PixelClassifier(booster)

    result = classifier.extract_pixels(np.ones((5, 5, 5, 5)))
    assert_array_equal(result, np.ones((5 * 5 * 5, 5)))


def test_image_predict_not_implemented(booster):
    classifier = PixelClassifier(booster)
    array = np.ones((5, 5, 5, 5))

    with pytest.raises(NotImplementedError):
        classifier.image_predict(array)


class DummyClassifier:
    """Predict value of second band."""

    @staticmethod
    def predict(X):
        return X[:, 1]

    @staticmethod
    def predict_proba(X):
        return X[:, 1]


ONES = np.ones((4, 8, 3))
ONES_FOUR = np.ones((5, 4, 8, 3))
ONES_THREE = np.ones((5, 4, 8))
ONES_TWO = np.ones((4, 8))


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (ONES_FOUR, ONES_THREE),
        (
            np.concatenate([ONES, ONES * 2, ONES * 3, ONES * 4, ONES * 5]).reshape(ONES_FOUR.shape),
            np.concatenate([ONES_TWO, ONES_TWO * 2, ONES_TWO * 3, ONES_TWO * 4, ONES_TWO * 5]).reshape(
                ONES_THREE.shape
            ),
        ),
        (
            np.concatenate([ONES, ONES * 0.2, ONES * 0.7, ONES * 4, ONES * 0.65]).reshape(ONES_FOUR.shape),
            np.concatenate([ONES_TWO, ONES_TWO * 0.2, ONES_TWO * 0.7, ONES_TWO * 4, ONES_TWO * 0.65]).reshape(
                ONES_THREE.shape
            ),
        ),
        (
            np.moveaxis(np.concatenate([ONES_THREE, ONES_THREE * 5, ONES_THREE * 0.7]).reshape((3, 5, 4, 8)), 0, -1),
            np.ones(ONES_THREE.shape) * 5,
        ),
    ],
)
def test_image_predict(test_input, expected):
    dummy_classifier = DummyClassifier()

    classifier = PixelClassifier(dummy_classifier)
    assert_array_equal(classifier.image_predict(test_input), expected)


def test_image_predict_proba(booster):
    classifier = PixelClassifier(booster)
    array = np.random.rand(5, 5, 5, 10)

    result = classifier.image_predict_proba(array)

    assert result.shape == (5, 5, 5, 2)
    assert result.dtype == np.float64
    assert_allclose(np.sum(result, axis=-1), np.ones((5, 5, 5)), rtol=1e-5)
