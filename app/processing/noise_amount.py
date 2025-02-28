import cv2
import numpy as np

class AddingNoise:

    @staticmethod
    def add_uniform_noise(image, noise_amount=0.5):  # noise_amount is from slider (0 to 1)
        high = int(noise_amount * 100)  # Scale slider value to range (0-100)
        noise = np.random.uniform(-high, high, image.shape).astype(np.int16)
        noisy_image = cv2.add(image.astype(np.int16), noise)
        return cv2.convertScaleAbs(noisy_image)

    @staticmethod
    def add_gaussian_noise(image, mean=0.0, sigma=0.5):  
        gaussian = np.random.normal(mean, sigma, image.shape).astype(np.int16)
        noisy_image = cv2.add(image.astype(np.int16), gaussian)
        return cv2.convertScaleAbs(noisy_image)

    @staticmethod
    def add_salt_and_pepper_noise(image, noise_amount=0.5):  
        salt_prob = noise_amount * 0.05  
        pepper_prob = salt_prob
        
        noisy_image = image.copy()
        h, w = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]

        num_salt = int(salt_prob * h * w * channels)
        num_pepper = int(pepper_prob * h * w * channels)

        # Add salt (white) noise
        coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
        if channels > 1:
            noisy_image[tuple(coords) + (np.random.randint(0, channels, num_salt),)] = 255
        else:
            noisy_image[tuple(coords)] = 255

        # Add pepper (black) noise
        coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
        if channels > 1:
            noisy_image[tuple(coords) + (np.random.randint(0, channels, num_pepper),)] = 0
        else:
            noisy_image[tuple(coords)] = 0

        return noisy_image
