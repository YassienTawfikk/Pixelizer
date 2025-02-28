import numpy as np

class FourierFilters:

    @staticmethod    
    def get_fft(image):
        dft = np.fft.fft2(image)
        dft_shifted = np.fft.fftshift(dft)
        return dft_shifted

    @staticmethod
    def apply_low_pass(image, radius=30):
        if len(image.shape) == 3:  # If RGB image
            channels = []  # Initialize an empty list
            for ch in range(image.shape[2]):
                channels.append(FourierFilters._apply_low_pass_grayscale(image[:, :, ch], radius))

            return np.stack(channels, axis=2)  # Merge channels back

        else:
            return FourierFilters._apply_low_pass_grayscale(image, radius)
        
    @staticmethod
    def apply_high_pass(image, radius=30):
        if len(image.shape) == 3:  # If RGB image
            channels = []  # Initialize an empty list
            for ch in range(image.shape[2]):
                channels.append(FourierFilters._apply_high_pass_grayscale(image[:, :, ch], radius))

            return np.stack(channels, axis=2)  # Merge channels back

        else:
            return FourierFilters._apply_high_pass_grayscale(image, radius)

    @staticmethod
    def _apply_low_pass_grayscale(image, radius):
        image_fourier = FourierFilters.get_fft(image)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2  # Center

        mask = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                if (i - crow) ** 2 + (j - ccol) ** 2 <= radius ** 2:
                    mask[i, j] = 1

        filtered_dft = image_fourier * mask
        dft_inverse = np.fft.ifftshift(filtered_dft)  # Shift back
        filtered_image = np.fft.ifft2(dft_inverse)  # Inverse Fourier Transform
        
        filtered_image = np.real(filtered_image)  # Keep only real part
        filtered_image = np.clip(filtered_image, 0, 255)  # Clip to valid range
        filtered_image = filtered_image.astype(np.uint8)  # Convert to uint8

        return filtered_image
    
    @staticmethod
    def _apply_high_pass_grayscale(image, radius):
        image_fourier = FourierFilters.get_fft(image)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2  # Center

        mask = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                if (i - crow) ** 2 + (j - ccol) ** 2 <= radius ** 2:
                    mask[i, j] = 0

        filtered_dft = image_fourier * mask
        dft_inverse = np.fft.ifftshift(filtered_dft)  # Shift back
        filtered_image = np.fft.ifft2(dft_inverse)  # Inverse Fourier Transform
        
        filtered_image = np.real(filtered_image)  # Get real part
        filtered_image -= filtered_image.min()    # Shift to start at 0
        filtered_image /= filtered_image.max()    # Scale to range [0,1]
        filtered_image *= 255                     # Scale to range [0,255]
        filtered_image = filtered_image.astype(np.uint8)  # Convert to uint8


        return filtered_image
    

    # APPLY LOW PASS AND APPLY HIGH PASS ARE IDENTICAL. THE ONLY DIFFERENCE IS FEL APPLY FILTER GREYSCALE
    # WHERE MASK=1 LEL LOW PASS W MASK=0 LEL HIGH PASS


    
    