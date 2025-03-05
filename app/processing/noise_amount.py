import cv2
import numpy as np


class AddingNoise:

    @staticmethod
    def add_uniform_noise(image, noise_amount=0.5):
        # calculate the maximum value of noise pixel values
        high = int(noise_amount * 100)

        # creates an array of the same size of image with uniformly distributed random values between -high and high
        noise = np.random.uniform(-high, high, image.shape).astype(np.int16)

        # adds these random values onto the image
        noisy_image = cv2.add(image.astype(np.int16), noise)

        # Converts the noisy image back to the uint8 range and clips values to this range.
        return cv2.convertScaleAbs(noisy_image)

    @staticmethod
    def add_gaussian_noise(image, mean=0.0, sigma=0.5):
        # creates an array of the same size of image with normally distributed random values
        #the mean and std determine the gaussian distribution shape (bell curve)
        gaussian = np.random.normal(mean, sigma, image.shape).astype(np.int16)

        # add noise onto the image
        noisy_image = cv2.add(image.astype(np.int16), gaussian)

        return cv2.convertScaleAbs(noisy_image)

    @staticmethod
    def add_salt_and_pepper_noise(image, noise_amount=0.5):
        # determines the probability of salt and pepper to be 5% of the percentage defined by the user.
        salt_prob = noise_amount * 0.05
        pepper_prob = salt_prob

        # takes a copy of the image
        noisy_image = image.copy()
        h, w = image.shape[:2]
        
        # Check if the image is grayscale or RGB
        if len(image.shape) == 2:
            channels = 1 
        else:
            channels = image.shape[2] 

        # calculate number of salt and pepper pixels
        num_salt = int(salt_prob * h * w * channels)
        num_pepper = int(pepper_prob * h * w * channels)

        # Add salt (white) noise
        # initialize arrays that will hold the coordinates of the salt row and col pixels
        salt_rows = []
        salt_cols = []
        for i in range(num_salt):
            # for the amount of salt pixels, append the arrays with random values around the image coordinates
            salt_rows.append(np.random.randint(0, h))
            salt_cols.append(np.random.randint(0, w))
        # join the row and col arrays into one array for efficiency
        salt_coords = [np.array(salt_rows), np.array(salt_cols)]

        if channels > 1:
            # if the image is RGB, create an array of size of salt pixels with random values ranging from 0 to 2 (or num of channels -1)
            salt_channels = np.random.randint(0, channels, num_salt)
            # finally in the noisy image, assign the randomly determined coordinates with value 255 (for white)
            # the salt_channels coordinate determines which channel of the 3 should be set to white
            noisy_image[tuple(salt_coords) + (salt_channels,)] = 255
        else:
            # for a grayscale image, assign 255 to the random coordinates directly
            noisy_image[tuple(salt_coords)] = 255

        # Add pepper (black) noise
        # initialize arrays that will hold the coordinates of the pepper row and col pixels
        pepper_rows = []
        pepper_cols = []
        for i in range(num_pepper):
            # for the amount of pepper pixels, append the arrays with random values around the image coordinates
            pepper_rows.append(np.random.randint(0, h))
            pepper_cols.append(np.random.randint(0, w))
        pepper_coords = [np.array(pepper_rows), np.array(pepper_cols)]

        # finally in the noisy image, assign the randomly determined coordinates with value 0 (for black)
        if channels > 1:
            pepper_channels = np.random.randint(0, channels, num_pepper)
            noisy_image[tuple(pepper_coords) + (pepper_channels,)] = 0
        else:
            noisy_image[tuple(pepper_coords)] = 0

        # return the noisy image
        return noisy_image
