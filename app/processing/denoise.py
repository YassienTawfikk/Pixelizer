import numpy as np


class Denoise:

    def __convolve(image, kernel, kernel_size, median=False):
        # calculate the padding size
        pad_size=kernel_size//2

        # if the image is RGB
        if len(image.shape) == 3:
            # initialize a zeroed array of the same size of the image
            filtered_image = np.zeros_like(image, dtype=np.float32)

            # iterate for each channel
            for c in range(image.shape[2]):
                # pad the image
                padded_channel = np.pad(image[:, :, c], pad_size, mode='reflect')

                # iterate over the rows and columns
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        # extract a region of the same size of the kernal
                        region = padded_channel[i:i + kernel_size, j:j + kernel_size]

                        # for average and gaussian filtering, return a pixel value of the summation of the kernel multiplied by the region
                        if median==False:
                            filtered_image[i, j, c] = np.sum(region * kernel)
                        #for median filtering return a pixel value that the median of the region 
                        if median==True:
                            filtered_image[i, j, c] = np.median(region)

        # for Grayscale Image, no iteration over the channels                    
        else:  
            # initialize the filtered image array and pad the image
            filtered_image = np.zeros_like(image, dtype=np.float32)
            padded_image = np.pad(image, pad_size, mode='reflect')

            # loop over the rows and columns to convolve the image with the kernel
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    # region extraction
                    region = padded_image[i:i + kernel_size, j:j + kernel_size]
                    
                    # for average and gaussian filtering, return a pixel value of the summation of the kernel multiplied by the region
                    if median==False:
                        filtered_image[i, j] = np.sum(region * kernel)
                    #for median filtering return a pixel value that the median of the region 
                    if median==True:
                        filtered_image[i, j] = np.median(region)

        # clip the filtered image values to [0-255] and cast to uint8
        return np.clip(filtered_image, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_average_filter(image, kernel_size=3):
        # create a kernel of ones and normalize it according to the kernel size
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        # convolve the image with the kernel
        output=Denoise.__convolve(image,kernel,kernel_size)
        return output
  

    @staticmethod
    def gaussian_kernel(kernel_size, sigma):
        """Generates a 2D Gaussian kernel."""
        # Create a 1D array of equally spaced points centered around 0
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)

        # Create two 2D grids (xx and yy) from the 1D array ax,
        # so that xx is the horizontal kernel and yy is the vertical kernel
        xx, yy = np.meshgrid(ax, ax)

        # Compute the Gaussian function for each point in the grid
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))

        # normalize the kernal to that its sum=1 and return it
        return kernel / np.sum(kernel)

    @staticmethod
    def apply_gaussian_filter(image, kernel_size=3, sigma=1):
        # generate a gaussian kernel 
        kernel = Denoise.gaussian_kernel(kernel_size, sigma)
        # convolve the image with the kernel
        output=Denoise.__convolve(image,kernel,kernel_size)
        return output

    @staticmethod
    def apply_median_filter(image, kernel_size=3):
        output=Denoise.__convolve(image,None,kernel_size,True)
        return output
