import cv2
import numpy as np
import pytest
from image_processing_toolkit import filtering


@pytest.fixture
def sample_image():
    # Create a sample input image (e.g., a simple grayscale image)
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:75, 25:75] = 255  # Add a white square in the center
    return image


def test_apply_gaussian_blur(sample_image):
    # Apply Gaussian blur to the sample image
    blurred_image = filtering.apply_gaussian_blur(sample_image)

    # Assert that the output image has the same dimensions as the input image
    assert blurred_image.shape == sample_image.shape

    # Assert that the output image is different from the input image (i.e., blur is applied)
    assert not np.array_equal(blurred_image, sample_image)


def test_apply_median_blur(sample_image):
    # Apply median blur to the sample image
    blurred_image = filtering.apply_median_blur(sample_image)

    # Assert that the output image has the same dimensions as the input image
    assert blurred_image.shape == sample_image.shape

    # Assert that the output image is different from the input image (i.e., blur is applied)
    assert not np.array_equal(blurred_image, sample_image)


def test_apply_bilateral_filter(sample_image):
    # Apply bilateral filter to the sample image
    filtered_image = filtering.apply_bilateral_filter(sample_image)

    # Assert that the output image has the same dimensions as the input image
    assert filtered_image.shape == sample_image.shape

    # Assert that the output image is different from the input image (i.e., filter is applied)
    assert not np.array_equal(filtered_image, sample_image)
