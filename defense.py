import cv2
import numpy as np
import tensorflow as tf

# Applies Gaussian blur to reduce small perturbations.
def apply_gaussian_blur(image, kernel_size=3):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Adds Gaussian noise to the image to disrupt adversarial perturbations.
def add_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255)

# Applies SimCLR-style augmentations to improve robustness.
def simclr_augmentation(image):
    # Ensure the image is 3D (height, width, channels)
    if len(image.shape) == 2:  # Grayscale or 2D image (height, width)
        image = np.expand_dims(image, axis=-1)  # Add a channel dimension (height, width, 1)
        image = np.repeat(image, 3, axis=-1)  # Duplicate the channel across RGB (height, width, 3)
    
    # Apply augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.5, 1.5)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    return image.numpy()

def apply_defenses(image):
    blurred = apply_gaussian_blur(image)
    noisy = add_gaussian_noise(blurred)
    augmented = simclr_augmentation(noisy)
    return augmented