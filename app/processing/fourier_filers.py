import numpy as np

class FourierFilters:

    @staticmethod    
    def get_fft(image):
        dft = np.fft.fft2(image)
        dft_shifted = np.fft.fftshift(dft)
        return dft_shifted

    @staticmethod
    def apply_low_pass(image, radius=30):
        return FourierFilters.__apply_filter(image, radius, 0)
        
    @staticmethod
    def apply_high_pass(image, radius=30):
        return FourierFilters.__apply_filter(image, radius, 1)

    @staticmethod
    def __apply_filter(image, radius=20, mask_value=1):
        if len(image.shape) == 3:  
            channels = [] 
            for ch in range(image.shape[2]):
                channels.append(FourierFilters.__apply_filter_grayscale(image[:, :, ch], radius, mask_value))

            return np.stack(channels, axis=2) 
        else:
            return FourierFilters.__apply_filter_grayscale(image, radius, mask_value)

    @staticmethod
    def __apply_filter_grayscale(image, radius=20, mask_value=1):
        image_fourier = FourierFilters.get_fft(image)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2  # Center

        mask = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                if (i - crow) ** 2 + (j - ccol) ** 2 <= radius ** 2:
                    mask[i, j] = mask_value

        filtered_dft = image_fourier * mask
        dft_inverse = np.fft.ifftshift(filtered_dft)  # Shift back
        filtered_image = np.fft.ifft2(dft_inverse)  # Inverse Fourier Transform
        
        filtered_image = np.real(filtered_image)  # Keep only real part
        filtered_image = np.clip(filtered_image, 0, 255)  # Clip to valid range
        filtered_image = filtered_image.astype(np.uint8)  # Convert to uint8

        return filtered_image
    
    