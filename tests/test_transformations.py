import numpy as np
import cv2
import pytest
from image_processing_toolkit import transformations


@pytest.fixture
def sample_image():
    # Create a sample grayscale image for testing
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:75, 25:75] = 255  # Add a white square in the middle

    return image


def test_rotate_image(sample_image):
    # Rotate the sample image by 90 degrees
    rotated_image = transformations.rotate_image(sample_image, 90)

    # Check that the rotated image has the expected shape
    assert rotated_image.shape == (100, 100)


def test_resize_image(sample_image):
    # Resize the sample image to a specified width and height
    resized_image = transformations.resize_image(sample_image, width=50, height=50)

    # Check that the resized image has the expected shape
    assert resized_image.shape == (50, 50)


def test_apply_gaussian_noise(sample_image):
    # Apply Gaussian noise to the sample image
    noisy_image = transformations.apply_gaussian_noise(sample_image, mean=0, std=25)
    # Check that the noisy image has the same shape as the sample image
    assert noisy_image.shape == sample_image.shape


def test_apply_salt_and_pepper_noise(sample_image):
    # Apply salt and pepper noise to the sample image
    noisy_image = transformations.apply_salt_and_pepper_noise(
        sample_image, salt_prob=0.01, pepper_prob=0.01
    )
    # Check that the noisy image has the same shape as the sample image
    assert noisy_image.shape == sample_image.shape
