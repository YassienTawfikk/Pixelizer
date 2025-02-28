import numpy as np

class ImageNormalization:
    @staticmethod
    def normalize_image( gray_image):
        """Normalize the image to the range [0, 255]."""
        # Find the minimum and maximum pixel values
        min_val = np.min(gray_image)
        max_val = np.max(gray_image)

        # Create a new image for the normalized values
        normalized_image = np.zeros_like(gray_image, dtype=np.uint8)

        # Apply normalization formula
        if max_val > min_val:  # Avoid division by zero
            normalized_image = ((gray_image - min_val) / (max_val - min_val)) * 255
            normalized_image = normalized_image.astype(np.uint8)  # Convert to uint8 type

        return normalized_image

    @staticmethod
    def normalize_image_rgb(image):
        """Normalize the RGB image to the range [0, 255]."""
        # Ensure the image is in RGB format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be an RGB image.")

        # Create an empty array for the normalized image
        normalized_image = np.zeros_like(image, dtype=np.uint8)

        # Process each channel individually
        for channel in range(3):
            # Find the minimum and maximum pixel values for the channel
            min_val = np.min(image[:, :, channel])
            max_val = np.max(image[:, :, channel])

            # Apply normalization formula
            if max_val > min_val:  # Avoid division by zero
                normalized_image[:, :, channel] = ((image[:, :, channel] - min_val) / (max_val - min_val)) * 255
                normalized_image[:, :, channel] = normalized_image[:, :, channel].astype(
                    np.uint8)  # Convert to uint8 type

        return normalized_image
