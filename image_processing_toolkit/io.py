# image_processing_toolit/io.py

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(file_path):
    """
    Load an image from a file.

    Parameters:
    - file_path: Path to the image file.

    Returns:
    - image: Loaded image (numpy array).
    """
    image = cv2.imread(file_path)
    return image


def save_image(image, file_path):
    """
    Save an image to a file.

    Parameters:
    - image: Input image (numpy array).
    - file_path: Path to save the image file.
    """

    cv2.imwrite(file_path, image)


def display_image(image):
    """
    Display an image using Matplotlib.

    Parameters:
    - image: Input image (numpy array).
    """
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap="gray")
    else:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB

    plt.axis("off")  # Hide axis
    plt.show()
