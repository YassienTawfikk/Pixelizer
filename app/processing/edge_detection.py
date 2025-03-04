import cv2
import numpy as np


class EdgeDetection:
    @staticmethod
    def convolve(image, kernel, roberts=False):
        kernel_height, kernel_width = kernel.shape
        if roberts == False:
            pad_h, pad_w = kernel_height // 2, kernel_width // 2
        if roberts == True:
            pad_h, pad_w = 1, 1
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        output = np.zeros_like(image, dtype=np.float32)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                output[i, j] = np.sum(region * kernel)

        return output

    @staticmethod
    def apply_sobel(image):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        grad_x = EdgeDetection.convolve(image, sobel_x, True)
        grad_y = EdgeDetection.convolve(image, sobel_y, True)
        return np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.uint8)

    @staticmethod
    def apply_roberts(image):
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])
        grad_x = EdgeDetection.convolve(image, roberts_x)
        grad_y = EdgeDetection.convolve(image, roberts_y)
        return np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.uint8)

    @staticmethod
    def apply_prewitt(image):
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        grad_x = EdgeDetection.convolve(image, prewitt_x)
        grad_y = EdgeDetection.convolve(image, prewitt_y)
        return np.sqrt(grad_x ** 2 + grad_y ** 2).astype(np.uint8)

    @staticmethod
    def apply_canny(image, threshold1=100, threshold2=200, apertureSize=3, L2gradient=False):
        return cv2.Canny(image, threshold1, threshold2, apertureSize=apertureSize, L2gradient=L2gradient)
