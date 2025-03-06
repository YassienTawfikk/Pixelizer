import cv2
from app.processing.fourier_filers import FourierFilters


class HybridImageGenerator:
    def __init__(self):
        self.low_frequency_image = None
        self.high_frequency_image = None
        self.hybrid_image = None

    def generate_hybrid_image(self, low_freq_image, high_freq_image, hybrid_image, low_pass_radius=20, high_pass_radius=20):
        """
        Generate a hybrid image by combining the low-frequency components of one image
        with the high-frequency components of another.
        """
        print("Generating hybrid image...")
        # Apply low-pass filter to the low-frequency image
        low_freq_filtered = FourierFilters.apply_low_pass(low_freq_image, low_pass_radius)

        # Apply high-pass filter to the high-frequency image
        high_freq_filtered = FourierFilters.apply_high_pass(high_freq_image, high_pass_radius)

        # Combine the two images
        self.hybrid_image = cv2.addWeighted(low_freq_filtered, 0.5, high_freq_filtered, 0.5, 0)
        return self.hybrid_image
