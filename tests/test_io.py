import cv2
import numpy as np
import pytest
import os
from image_processing_toolkit import io


@pytest.fixture
def sample_image(tmp_path):
    # Create a sample image for testing
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Save the sample image to a temporary file
    image_path = tmp_path / "test_image.png"
    cv2.imwrite(str(image_path), image)

    yield image, image_path  # Provide the image and its path as a tuple

    # Clean up: Delete the temporary image after the tests are done
    os.remove(str(image_path))


def test_load_image(sample_image):
    # Unpack the sample image fixture
    image, image_path = sample_image

    # Read the image from the temporary file
    loaded_image = io.load_image(str(image_path))

    # Assert that the loaded image is not None
    assert loaded_image is not None

    # Assert that the dimensions of the loaded image match the sample image
    assert loaded_image.shape == image.shape


def test_save_image(tmp_path, sample_image):
    # Unpack the sample image fixture
    image, _ = sample_image

    # Define the path to save the image
    save_path = tmp_path / "saved_image.png"

    # Save the sample image to the temporary directory
    io.save_image(image, str(save_path))

    # Read the saved image
    loaded_image = cv2.imread(str(save_path))

    # Assert that the loaded image is not None
    assert loaded_image is not None

    # Assert that the dimensions of the loaded image match the sample image
    assert loaded_image.shape == image.shape

    # Assert that the pixel values of the loaded and sample images are equal
    assert np.array_equal(loaded_image, image)
