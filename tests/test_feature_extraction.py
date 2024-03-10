import numpy as np
import pytest
from image_processing_toolkit import feature_extraction


@pytest.fixture
def sample_image():
    # Create a sample grayscale image with known features for testing
    image = np.zeros((100, 100), dtype=np.uint8)

    # Add known features (e.g., edges, corners) to the image
    # For example, add white rectangles for edges and white corners
    image[20:30, 40:60] = 255  # Add a white rectangle for edges
    image[40:50, 20:30] = 255  # Add a white rectangle for edges
    image[60:70, 70:80] = 255  # Add a white rectangle for edges
    image[20:30, 20:30] = 255  # Add white corners
    image[20:30, 70:80] = 255  # Add white corners
    image[70:80, 20:30] = 255  # Add white corners
    image[70:80, 70:80] = 255  # Add white corners

    return image


def test_extract_edges(sample_image):
    # Extract edges from the sample image
    edges = feature_extraction.extract_edges(sample_image)

    # Check that edges are extracted and have the correct shape
    assert isinstance(edges, np.ndarray)
    assert edges.shape == sample_image.shape


def test_detect_corners():
    # Create a sample image with a known corner (a white square at the top-left corner)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:10, :10] = 255

    # Detect corners using the function
    corners, marked_image = feature_extraction.detect_corners(image)

    # Check if corners were detected (any non-zero values in corners array)
    assert np.any(corners)

    # Check if the marked image contains the detected corner
    assert np.any(marked_image[:10, :10])
