# image_processing_toolit/filtering.py

import cv2
import numpy as np


def apply_gaussian_blur(image, kernel_size=(5, 5), sigmaX=10, sigmaY=10):
    """
    Apply Gaussian blur to an image.

    Parameters:
    - image: Input image (numpy array).
    - kernel_size: Size of the Gaussian kernel (tuple).
    - sigmaX: Standard deviation in the X direction (default is 10).
    - sigmaY: Standard deviation in the Y direction (default is 10).

    Returns:
    - blurred_image: Blurred image.
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigmaX, sigmaY=sigmaY)
    return blurred_image


def apply_median_blur(image, kernel_size=5):
    """
    Apply median blur to an image.

    Parameters:
    - image: Input image (numpy array).
    - kernel_size: Size of the median filter (integer).

    Returns:
    - blurred_image: Blurred image.
    """
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to an image.

    Parameters:
    - image: Input image (numpy array).
    - d: Diameter of each pixel neighborhood (integer).
    - sigma_color: Filter sigma in the color space (float).
    - sigma_space: Filter sigma in the coordinate space (float).

    Returns:
    - filtered_image: Filtered image.
    """
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return filtered_image
