# image_processing_toolit/analysis.py

import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_histogram(image, num_bins=256):
    """
    Compute the histogram of an image.

    Args:
        image (numpy.ndarray): Input image.
        num_bins (int, optional): Number of bins for the histogram. Defaults to 256.

    Returns:
        list: List of histograms for each channel.
    """
    if len(image.shape) == 2:
        # Grayscale image
        histogram = cv2.calcHist([image], [0], None, [num_bins], [0, 256])
        return histogram
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image
        histograms = [
            cv2.calcHist([image], [i], None, [num_bins], [0, 256]) for i in range(3)
        ]
        return histograms
    else:
        raise ValueError("Unsupported image format")


def count_objects(image, threshold=128):
    """
    Count the number of objects in a binary image.

    Parameters:
    - image: Binary image (numpy array).
    - threshold: Threshold value for binarization.

    Returns:
    - num_objects: Number of objects in the image.
    """
    # Convert the image to binary (if necessary)
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Binarize the image using the specified threshold
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Count the number of contours (objects)
    num_objects = len(contours)

    return num_objects


def plot_histogram(image):
    """
    Plot the histogram of an image using Matplotlib.

    Args:
        image (numpy.ndarray): Input image.
    """
    if len(image.shape) == 2:
        # Grayscale image
        plt.hist(image.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.7)
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.title("Histogram of Grayscale Image")
        plt.show()
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # RGB image
        colors = ("r", "g", "b")
        for i, color in enumerate(colors):
            plt.plot(compute_histogram(image)[i], color=color)
            plt.xlim([0, 256])
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.title("Histogram of RGB Image")
        plt.show()
    else:
        raise ValueError("Unsupported image format")
