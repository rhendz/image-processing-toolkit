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


def test_compute_histogram_gray(sample_image):
    # Calculate the histogram using the analysis function for a grayscale image
    histogram = analysis.compute_histogram(sample_image)

    # Check that the histogram has the expected shape and sum
    assert histogram.shape == (256, 1)
    assert np.sum(histogram) == sample_image.size


def test_compute_histogram_rgb(sample_image):
    # Convert the grayscale sample image to RGB format
    rgb_image = cv2.cvtColor(sample_image, cv2.COLOR_GRAY2RGB)

    # Calculate the histogram using the analysis function for an RGB image
    histograms = analysis.compute_histogram(rgb_image)

    # Check that the histograms list has the expected length and each histogram has the expected shape
    assert len(histograms) == 3
    for hist in histograms:
        assert hist.shape == (256, 1)


def test_count_objects(sample_image):
    # Binarize the image using a threshold
    _, binary_image = cv2.threshold(sample_image, 127, 255, cv2.THRESH_BINARY)

    # Count objects using the analysis function
    num_objects = analysis.count_objects(binary_image)

    # Check that the number of objects is as expected
    assert num_objects == 1  # The white square should be considered as one object
