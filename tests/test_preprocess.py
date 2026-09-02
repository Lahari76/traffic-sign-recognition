import numpy as np
import pytest

from src.preprocess import preprocess_image, prepare_for_prediction


def test_preprocess_image_shape():
    image = np.zeros((100, 120, 3), dtype=np.uint8)

    result = preprocess_image(image)

    assert result.shape == (32, 32, 3)


def test_preprocess_normalization():
    image = np.full((50, 50, 3), 255, dtype=np.uint8)

    result = preprocess_image(image)

    assert result.dtype == np.float32
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0
    assert np.allclose(result, 1.0)


def test_grayscale_image():
    image = np.zeros((50, 50), dtype=np.uint8)

    result = preprocess_image(image)

    assert result.shape == (32, 32, 3)


def test_prediction_batch_dimension():
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    result = prepare_for_prediction(image)

    assert result.shape == (1, 32, 32, 3)


def test_none_image_raises_error():
    with pytest.raises(ValueError):
        preprocess_image(None)


def test_invalid_type_raises_error():
    with pytest.raises(TypeError):
        preprocess_image("not-an-image")
