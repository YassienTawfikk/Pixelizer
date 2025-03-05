import numpy as np


class EqualizeHistogram:
    @staticmethod
    def equalizeHist(image):
        # Ensure the image is in grayscale
        if len(image.shape) != 2:
            raise ValueError("Input image must be a grayscale image.")

        # Step 1: the number of repetitions (f) for each pixel value
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

        # Step 2: Calculate the PDF for each pixel
        pdf = hist / np.sum(hist)

        # Step 3: Calculate the CDF for each pixel
        cdf = np.cumsum(pdf)

        # Step 4: Calculate the new pixel values
        new_pixel_values = np.round(cdf * 255).astype(np.uint8)

        # Map the old pixel values to the new pixel values
        equalized_image = new_pixel_values[image]

        return equalized_image

    @staticmethod
    def equalizeHistRGB(image):
        # Ensure the image is in RGB format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be an RGB image.")

        # Initialize an empty array for the equalized image
        equalized_image = np.zeros_like(image)

        # Process each channel individually
        for channel in range(3):
            # Step 1: Calculate the histogram for the channel
            hist, bins = np.histogram(image[:, :, channel].flatten(), bins=256, range=[0, 256])

            # Step 2: Calculate the PDF for the channel
            pdf = hist / np.sum(hist)

            # Step 3: Calculate the CDF for the channel
            cdf = np.cumsum(pdf)

            # Step 4: Calculate the new pixel values
            new_pixel_values = np.round(cdf * 255).astype(np.uint8)

            # Map the old pixel values to the new pixel values
            equalized_image[:, :, channel] = new_pixel_values[image[:, :, channel]]

        return equalized_image
