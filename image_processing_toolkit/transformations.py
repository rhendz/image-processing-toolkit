# image_processing_toolit/transformations.py
import cv2
import numpy as np


def rotate_image(image, angle):
    """
    Rotate an image by a specified angle.

    Parameters:
    - image: Input image (numpy array).
    - angle: Rotation angle in degrees.

    Returns:
    - rotated_image: Rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image


def resize_image(image, width=None, height=None, interpolation=cv2.INTER_LINEAR):
    """
    Resize an image to a specified width and/or height.

    Parameters:
    - image: Input image (numpy array).
    - width: Width of the resized image (optional).
    - height: Height of the resized image (optional).
    - interpolation: Interpolation method for resizing.

    Returns:
    - resized_image: Resized image.
    """
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(image.shape[0])
        dim = (int(image.shape[1] * r), height)
    else:
        r = width / float(image.shape[1])
        dim = (width, int(image.shape[0] * r))
    resized_image = cv2.resize(image, dim, interpolation=interpolation)
    return resized_image


def apply_gaussian_noise(image, mean=0, std=5):
    """
    Apply Gaussian noise to an image.

    Parameters:
    - image: Input image (numpy array).
    - mean: Mean of the Gaussian distribution (default is 0).
    - std: Standard deviation of the Gaussian distribution (default is 25).

    Returns:
    - noisy_image: Image with added Gaussian noise.
    """
    # Generate Gaussian noise with the same shape as the input image
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)

    # Ensure pixel values are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image


def apply_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Apply salt and pepper noise to an image.

    Parameters:
    - image: Input image (numpy array).
    - salt_prob: Probability of adding salt noise (default is 0.01).
    - pepper_prob: Probability of adding pepper noise (default is 0.01).

    Returns:
    - noisy_image: Image with salt and pepper noise.
    """
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape) < salt_prob
    pepper_mask = np.random.rand(*image.shape) < pepper_prob
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image
