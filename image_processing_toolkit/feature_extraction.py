# image_processing_toolit/feature_extraction.py
import cv2
import numpy as np


def extract_edges(image, min_threshold=100, max_threshold=200):
    """
    Extract edges from an image using the Canny edge detection algorithm.

    Parameters:
    - image: Input image (numpy array).
    - min_threshold: Minimum threshold value for edge detection.
    - max_threshold: Maximum threshold value for edge detection.

    Returns:
    - edges: Extracted edges.
    """
    # Convert the image to grayscale (if necessary)
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, min_threshold, max_threshold)

    return edges


def detect_corners(image, threshold=0.01):
    """
    Detect corners in an image using the Harris corner detection method.

    Parameters:
    - image: Input image (numpy array).
    - threshold: Threshold value for corner detection (default is 0.01).

    Returns:
    - corners: Dilated image highlighting corner points.
    - marked_image: Image with detected corners marked.
    """
    # Convert the image to grayscale (if necessary)
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        gray_image = np.float32(image)

    # Detect corners using the Harris corner detection method
    corners = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

    # Dilate the binary image to enhance corner points
    corners = cv2.dilate(corners, None)
    threshold_value = threshold * corners.max()

    # Mark the detected corners on the original image
    marked_image = image.copy()

    for j in range(0, corners.shape[0]):
        for i in range(0, corners.shape[1]):
            if corners[j, i] > threshold_value:
                # image, center pt, radius, color, thickness
                cv2.circle(marked_image, (i, j), 1, (0, 255, 0), 1)

    return corners, marked_image
