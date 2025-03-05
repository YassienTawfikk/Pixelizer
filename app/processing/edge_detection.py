import cv2
import numpy as np


class EdgeDetection:

    @staticmethod
    def convolve(image, kernel, roberts=False):
        # determine kernel height and width based on the kernel passed to the function
        kernel_height, kernel_width = kernel.shape

        # determine size of image padding. if type is Roberts padding size is always 1
        if roberts == False:
            pad_h, pad_w = kernel_height // 2, kernel_width // 2
        if roberts == True:
            pad_h, pad_w = 1, 1

        # add constant zero padding around the image
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        # initialize the output image of an array of zeroes with the same shape as image
        output = np.zeros_like(image, dtype=np.float32)

        # loop over the rows and columns of the image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # extract a region of the size of the kernel
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                # return a pixel value that the summation of the mernel values multiplied by the region values
                output[i, j] = np.sum(region * kernel)

        # return the convolved output
        return output

    @staticmethod
    def apply_sobel(image):
        # create sobel type kernels for horizontal and vertical edge detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # get the gradient (convolution) of each kernel over the image
        grad_x = EdgeDetection.convolve(image, sobel_x)
        grad_y = EdgeDetection.convolve(image, sobel_y)

        # compute the magnitude of the gradients and cast to type uint8 to avoid overflow 
        return np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.uint8)

    @staticmethod
    def apply_roberts(image):
        # create roberts type kernels for horizontal and vertical edge detection
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])

        # get the gradient (convolution) of each kernel over the image
        grad_x = EdgeDetection.convolve(image, roberts_x, True)
        grad_y = EdgeDetection.convolve(image, roberts_y, True)

        # compute the magnitude of the gradients and cast to type uint8 to avoid overflow 
        return np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.uint8)

    @staticmethod
    def apply_prewitt(image):
        # create perwitt type kernels for horizontal and vertical edge detection
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # get the gradient (convolution) of each kernel over the image
        grad_x = EdgeDetection.convolve(image, prewitt_x)
        grad_y = EdgeDetection.convolve(image, prewitt_y)

        # compute the magnitude of the gradients and cast to type uint8 to avoid overflow 
        return np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.uint8)

    @staticmethod
    def apply_canny(image, threshold1=100, threshold2=200, apertureSize=3, L2gradient=False):
        # use the built in cv2 function to compute the canny edge detection where aperture size is the kernel size set to 3
        return cv2.Canny(image, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)
