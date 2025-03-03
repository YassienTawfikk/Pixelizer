import numpy as np


class Denoise:

    @staticmethod
    def apply_average_filter(image, kernel_size=3):
        """Applies an average (mean) filter to grayscale or RGB images manually."""
        pad_size = kernel_size // 2
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

        if len(image.shape) == 3:
            filtered_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                padded_channel = np.pad(image[:, :, c], pad_size, mode='reflect')
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                        filtered_image[i, j, c] = np.sum(region * kernel)
        else:  # Grayscale Image
            filtered_image = np.zeros_like(image, dtype=np.float32)
            padded_image = np.pad(image, pad_size, mode='reflect')
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i + kernel_size, j:j + kernel_size]
                    filtered_image[i, j] = np.sum(region * kernel)

        return np.clip(filtered_image, 0, 255).astype(np.uint8)

    @staticmethod
    def gaussian_kernel(kernel_size, sigma):
        """Generates a 2D Gaussian kernel."""
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        return kernel / np.sum(kernel)

    @staticmethod
    def apply_gaussian_filter(image, kernel_size=3, sigma=1):
        """Applies Gaussian filter manually to grayscale or RGB images."""
        kernel = Denoise.gaussian_kernel(kernel_size, sigma)
        pad_size = kernel_size // 2

        if len(image.shape) == 3:
            filtered_image = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                padded_channel = np.pad(image[:, :, c], pad_size, mode='reflect')
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                        filtered_image[i, j, c] = np.sum(region * kernel)
        else:
            filtered_image = np.zeros_like(image, dtype=np.float32)
            padded_image = np.pad(image, pad_size, mode='reflect')
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i + kernel_size, j:j + kernel_size]
                    filtered_image[i, j] = np.sum(region * kernel)

        return np.clip(filtered_image, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_median_filter(image, kernel_size=3):
        """Applies a median filter manually to grayscale or RGB images."""
        pad_size = kernel_size // 2

        if len(image.shape) == 3:
            filtered_image = np.zeros_like(image, dtype=np.uint8)
            for c in range(image.shape[2]):
                padded_channel = np.pad(image[:, :, c], pad_size, mode='reflect')
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                        filtered_image[i, j, c] = np.median(region)
        else:
            filtered_image = np.zeros_like(image, dtype=np.uint8)
            padded_image = np.pad(image, pad_size, mode='reflect')
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i + kernel_size, j:j + kernel_size]
                    filtered_image[i, j] = np.median(region)

        return filtered_image

    # basically each filter (except for the extra step in gaussian) byakhod el kernel and moves it over each pixel using the nested loop to extract a region.
    # for the mean filter, 3nd kol loop bya5of avg el region
    # for the median, bya5of median el region
    # for the gaussian bya5od el sum w keda keda el kernel normalized (zy badal makhod mean directly, khadt sum w asamt 3al 3adad)
