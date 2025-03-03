import numpy as np
import cv2


class Thresholding:
    @staticmethod
    def global_threshold(image, threshold_value):
        # Create an output binary image
        binary_image = np.zeros_like(image)
        binary_image[image > threshold_value] = 255
        return binary_image

    @staticmethod
    def local_threshold(image, block_size):
        # Pad the image to handle borders
        padded_image = np.pad(image, pad_width=block_size // 2, mode='edge')  # pad_width to ensure that the local block is centered around each pixel
        binary_image = np.zeros_like(image)

        # Iterate over each pixel
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Extract the local block
                local_block = padded_image[i:i + block_size, j:j + block_size]
                # Calculate the local mean
                local_mean = np.mean(local_block)
                # Determine the threshold
                threshold = local_mean
                # Apply the threshold
                binary_image[i, j] = 255 if image[i, j] > threshold else 0

        return binary_image
