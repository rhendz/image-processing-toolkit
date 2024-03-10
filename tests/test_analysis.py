import numpy as np
import cv2
import pytest
from image_processing_toolkit import analysis


@pytest.fixture
def sample_image():
    # Create a sample grayscale image for testing
    image = np.zeros((100, 100), dtype=np.uint8)
    image[25:75, 25:75] = 255  # Add a white square in the middle

    return image


def test_compute_histogram(sample_image):
    # Calculate the histogram using the analysis function
    histogram = analysis.compute_histogram(sample_image)

    # Check that the histogram has the expected shape and sum
    assert histogram.shape == (256,)
    assert (
        np.sum(histogram) == sample_image.size
    )  # Sum of histogram should equal number of pixels


def test_count_objects(sample_image):
    # Binarize the image using a threshold
    _, binary_image = cv2.threshold(sample_image, 127, 255, cv2.THRESH_BINARY)

    # Count objects using the analysis function
    num_objects = analysis.count_objects(binary_image)

    # Check that the number of objects is as expected
    assert num_objects == 1  # The white square should be considered as one object
