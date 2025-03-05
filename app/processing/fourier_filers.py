import numpy as np


class FourierFilters:

    @staticmethod
    def get_fft(image):
        # computes the fft of the image
        dft = np.fft.fft2(image)
        # shifts the zero frequency component to the center
        dft_shifted = np.fft.fftshift(dft)
        return dft_shifted

    @staticmethod
    def apply_low_pass(image, radius=30):
        # call the apply filter function and pass the mask value as 1 for LPF
        print(image.shape)
        return FourierFilters.__apply_filter(image, radius, 1)

    @staticmethod
    def apply_high_pass(image, radius=30):
        # call the apply filter function and pass the mask value as 0 for HPF
        return FourierFilters.__apply_filter(image, radius, 0)

    def __apply_filter(image, radius=20, mask_value=1):
        # image is RGB
        if len(image.shape) == 3:
            # create an array to hold the values of each channel
            channels = []
            # iterate over the channels
            for ch in range(image.shape[2]):
                # apply the filter on this channel then append it to the array
                channels.append(FourierFilters.__apply_filter_grayscale(image[:, :, ch], radius, mask_value))

            # stack the filtered channels over each other to return to RGB
            return np.stack(channels, axis=2)
        
        # for a greyscale image (one channel) apply grayscale filtering directly
        else:
            return FourierFilters.__apply_filter_grayscale(image, radius, mask_value)

    def __apply_filter_grayscale(image, radius=20, mask_value=1):
        # obtain the fft of the image
        image_fourier = FourierFilters.get_fft(image)
        # find number of image rows and columns
        rows, cols = image.shape
        # find the center coordinates of the image
        crow, ccol = rows // 2, cols // 2  

        # create a mask array thats the same size as the image
        mask = np.zeros((rows, cols), dtype=np.uint8)
        print(image.shape)

        # max_radius = min(rows, cols) / 2
        # radius = (radius / 50) * max_radius  # Scale 0-50% to 0-max_radius

        # initialize to 0 for LPF, and to 1 for HPF
        if mask_value == 0:
            mask = 1 - mask

        for i in range(rows):
            for j in range(cols):
                # determine if coordinate lies within the circle using the equation of the circle
                if (i - crow) ** 2 + (j - ccol) ** 2 <= radius ** 2:
                    # if so fill in the circle values with the mask value
                    mask[i, j] = mask_value

        filtered_dft = image_fourier * mask    # multiply the fft of the image by the mask
        dft_inverse = np.fft.ifftshift(filtered_dft)   #shift the spectrum back to the original format
        filtered_image = np.fft.ifft2(dft_inverse)  #apply inverse fft

        filtered_image = np.real(filtered_image)   #extract only the real values of the filtered image to avoid complex ones
        filtered_image = np.clip(filtered_image, 0, 255)   #clip pixels to the [0-255] range
        filtered_image = filtered_image.astype(np.uint8)   #cast pixel values to uint8 

        return filtered_image
