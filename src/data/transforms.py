import cv2
import numpy as np
import tensorflow as tf

from src.helpers.config import get_settings

settings = get_settings()


def apply_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Apply image preprocessing using OpenCV.

    Steps:
        1. Median Blur
        2. CLAHE on LAB color space
    """

    # Median Blur
    image = cv2.medianBlur(image, 3)

    # RGB -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    lab = cv2.merge((l_channel, a_channel, b_channel))

    # LAB -> RGB
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return image


def train_transform(image: np.ndarray) -> tf.Tensor:
    """
    Apply preprocessing and augmentation for training.
    """

    image = apply_preprocessing(image)

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    image = tf.image.resize(
        image,
        (
            settings.IMAGE_SIZE,
            settings.IMAGE_SIZE,
        ),
    )

    image = image / 255.0

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(
        image,
        max_delta=0.2,
    )

    image = tf.image.random_contrast(
        image,
        lower=0.8,
        upper=1.2,
    )

    return image


def validation_transform(image: np.ndarray) -> tf.Tensor:
    """
    Apply preprocessing for validation and testing.
    """

    image = apply_preprocessing(image)

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    image = tf.image.resize(
        image,
        (
            settings.IMAGE_SIZE,
            settings.IMAGE_SIZE,
        ),
    )

    image = image / 255.0

    return image