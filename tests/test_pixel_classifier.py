import os

import numpy as np
import pytest
from lightgbm import Booster
from numpy.testing import assert_allclose, assert_array_equal

import s2cloudless
from s2cloudless import PixelClassifier
from s2cloudless.cloud_detector import MODEL_FILENAME


@pytest.fixture(name="booster")
def booster_fixture() -> Booster:
    package_path = os.path.dirname(s2cloudless.__file__)
    model_path = os.path.join(package_path, "models", MODEL_FILENAME)
    return Booster(model_file=model_path)


def test_image_predict_not_implemented_for_booster(booster: Booster) -> None:
    classifier = PixelClassifier(booster)
    array = np.ones((5, 5, 5, 5))

    with pytest.raises(NotImplementedError):
        classifier.image_predict(array)


class DummyClassifier:
    """Predict value of second band."""

    @staticmethod
    def predict(X: np.ndarray) -> np.ndarray:
        return X[:, 1]

    @staticmethod
    def predict_proba(X: np.ndarray) -> np.ndarray:
        return X[:, 1]


X_Y = np.ones((4, 8))
T_X_Y = np.ones((5, 4, 8))
X_Y_B = np.ones((4, 8, 3))
T_X_Y_B = np.ones((5, 4, 8, 3))
T_Y_B = np.ones((5, 8, 3))
T_Y = np.ones((5, 8))


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (T_X_Y_B, T_X_Y),
        (
            np.concatenate([X_Y_B, X_Y_B * 2, X_Y_B * 3, X_Y_B * 4, X_Y_B * 5]).reshape(T_X_Y_B.shape),
            np.concatenate([X_Y, X_Y * 2, X_Y * 3, X_Y * 4, X_Y * 5]).reshape(T_X_Y.shape),
        ),
        (
            np.concatenate([X_Y_B, X_Y_B * 0.2, X_Y_B * 0.7, X_Y_B * 0.4, X_Y_B * 0.65]).reshape(T_X_Y_B.shape),
            np.concatenate([X_Y, X_Y * 0.2, X_Y * 0.7, X_Y * 0.4, X_Y * 0.65]).reshape(T_X_Y.shape),
        ),
        (
            np.moveaxis(np.concatenate([T_X_Y, T_X_Y * 5, T_X_Y * 0.7]).reshape((3, 5, 4, 8)), 0, -1),
            np.ones(T_X_Y.shape) * 5,
        ),
        (
            np.moveaxis(np.concatenate([i * T_Y_B for i in range(4)]).reshape((4, 5, 8, 3)), 0, 1),
            np.moveaxis(np.concatenate([i * T_Y for i in range(4)]).reshape((4, 5, 8)), 0, 1),
        ),
        (
            np.array([[[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]]),
            np.array([[[2, 5], [2, 5], [2, 5]]]),
        ),
    ],
)
def test_image_predict(test_input: np.ndarray, expected: np.ndarray) -> None:
    dummy_classifier = DummyClassifier()

    classifier = PixelClassifier(dummy_classifier)
    assert_array_equal(classifier.image_predict(test_input), expected)


def test_image_predict_proba(booster: Booster) -> None:
    classifier = PixelClassifier(booster)
    array = np.random.rand(5, 5, 5, 10)

    result = classifier.image_predict_proba(array)

    assert result.shape == (5, 5, 5, 2)
    assert result.dtype == np.float64
    assert_allclose(np.sum(result, axis=-1), np.ones((5, 5, 5)), rtol=1e-5)
