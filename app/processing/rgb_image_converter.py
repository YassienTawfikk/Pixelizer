import numpy as np


class RGBImageConverter:
    @staticmethod
    def rgb_to_gray(image):
        # Check if the image is in RGB format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a color (RGB) image.")

        # Apply the luminosity method(Z = 0.2126 * R + 0.7152 * G + 0.0722 * B) to convert to grayscale
        gray_image = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
        return gray_image.astype(np.uint8)
