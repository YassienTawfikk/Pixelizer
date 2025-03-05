# Core PyQt5 module
from PyQt5 import QtWidgets

# Core utility and services
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices

# Main GUI design
from app.design.main_layout import Ui_MainWindow
from app.design.metrics_graphs import Ui_PopWindow

# Image processing functionality
import cv2
from app.processing.edge_detection import EdgeDetection
from app.processing.noise_amount import AddingNoise
from app.processing.denoise import Denoise
from app.processing.fourier_filers import FourierFilters
from app.processing.hybrid_images import HybridImageGenerator
from app.processing.histogram_equalization import EqualizeHistogram
from app.processing.image_normalization import ImageNormalization
from app.processing.thresholding import Thresholding
from app.processing.rgb_image_converter import RGBImageConverter


class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()

        self.path = None
        self.path_1 = None
        self.path_2 = None

        self.original_image = None
        self.processed_image = self.original_image

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        self.srv = ImageServices()

        self.hybrid_generator = HybridImageGenerator()

        self.equalize = EqualizeHistogram()
        self.normalize = ImageNormalization()
        self.threshold = Thresholding()
        self.convert = RGBImageConverter()

        # variables for hybrid images
        self.low_frequency_image = None
        self.high_frequency_image = None
        self.hybrid_image = None

        # Connect signals to slots
        self.setupConnections()

    def run(self):
        """Run the application."""
        self.MainWindow.showFullScreen()
        self.app.exec_()

    def setupConnections(self):
        """Connect buttons to their respective methods."""
        self.ui.quit_app_button.clicked.connect(self.closeApp)
        self.ui.upload_button.clicked.connect(self.drawImage)

        self.ui.save_image_button.clicked.connect(lambda: self.srv.save_image(self.processed_image))

        self.ui.clear_image_button.clicked.connect(self.clear_images)
        self.ui.reset_image_button.clicked.connect(self.reset_images)

        self.ui.sobel_edge_detection_button.clicked.connect(lambda: self.edge_detection("Sobel"))
        self.ui.roberts_edge_detection_button.clicked.connect(lambda: self.edge_detection("Roberts"))
        self.ui.prewitt_edge_detection_button.clicked.connect(lambda: self.edge_detection("Prewitt"))
        self.ui.canny_edge_detection_button.clicked.connect(lambda: self.edge_detection("Canny"))

        self.ui.uniform_noise_button.clicked.connect(lambda: self.apply_noise("Uniform"))
        self.ui.gaussian_noise_button.clicked.connect(lambda: self.apply_noise("Gaussian"))
        self.ui.salt_pepper_noise_button.clicked.connect(lambda: self.apply_noise("Salt&Pepper"))

        self.ui.average_filter_button.clicked.connect(lambda: self.remove_noise("Average"))
        self.ui.gaussian_filter_apply_button.clicked.connect(lambda: self.remove_noise("Gaussian"))
        self.ui.median_filter_button.clicked.connect(lambda: self.remove_noise("Median"))

        self.ui.hpf_button.clicked.connect(lambda: self.apply_fourier_filters("High"))
        self.ui.lpf_button.clicked.connect(lambda: self.apply_fourier_filters("Low"))

        # Connect the show_metrics_button to the new method
        self.ui.show_metrics_button.clicked.connect(self.show_metrics)

        self.ui.equalize_image_button.clicked.connect(self.equalize_image)
        self.ui.normalize_image_button.clicked.connect(self.normalize_image)
        self.ui.grayscaling_button.clicked.connect(self.gray_image_converter)
        self.ui.local_threshold_button.clicked.connect(self.local_thresholding)
        self.ui.global_threshold_button.clicked.connect(self.global_thresholding)

        # Connect the hybrid image upload buttons
        self.ui.upload_low_freq_button.clicked.connect(self.upload_low_frequency_image)
        self.ui.upload_high_freq_button.clicked.connect(self.upload_high_frequency_image)
        self.ui.generate_hybrid_image_button.clicked.connect(self.hybrid_generation)

    def show_metrics(self):
        """Show the histogram and metrics popup."""
        if self.path is None:
            print("No image loaded. Please upload an image first.")
            return

        # Initialize the popup window
        self.popup = Ui_PopWindow()
        dialog = QtWidgets.QDialog()
        self.popup.setupUi(dialog)  # Initialize the UI

        # Plot the histogram
        self.popup.plot_histogram_cdf(self.original_image, "original")
        self.popup.plot_histogram_cdf(self.processed_image, "processed")

        # Show the popup
        dialog.exec_()
        # self.popup.show_popup()

    def apply_noise(self, type="Uniform"):
        if self.original_image is None:
            print("No image loaded. Please upload an image first.")
            return

        if type == "Uniform":
            amount = self.ui.uniform_noise_slider.value() / 100
            self.processed_image = AddingNoise.add_uniform_noise(self.original_image, amount)
        elif type == "Gaussian":
            mean = self.ui.mean_gaussian_noise_slider.value()
            stdev = self.ui.stddev_gaussian_noise_slider.value()
            self.processed_image = AddingNoise.add_gaussian_noise(self.original_image, mean, stdev)
        elif type == "Salt&Pepper":
            amount = self.ui.salt_pepper_noise_slider.value() / 100
            self.processed_image = AddingNoise.add_salt_and_pepper_noise(self.original_image, amount)


        self.showProcessed()

    def remove_noise(self, type="Average"):
        if self.original_image is None:
            print("No image loaded. Please upload an image first.")
            return

        sigma = self.ui.gaussian_filter_sigma_spinbox.value()

        if type == "Average":
            self.processed_image = Denoise.apply_average_filter(self.processed_image, self.ui.current_kernal_size)
        elif type == "Gaussian":
            self.processed_image = Denoise.apply_gaussian_filter(self.processed_image, self.ui.current_kernal_size, sigma)
        elif type == "Median":
            self.processed_image = Denoise.apply_median_filter(self.processed_image, self.ui.current_kernal_size)

        self.showProcessed()

    def edge_detection(self, type="Sobel"):
        if self.original_image is None:
            print("No image loaded. Please upload an image first.")
            return  # Prevents crashing

        if len(self.original_image.shape) == 3:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        if type == "Sobel":
            self.processed_image = EdgeDetection.apply_sobel(self.original_image)
        elif type == "Roberts":
            self.processed_image = EdgeDetection.apply_roberts(self.original_image)
        elif type == "Prewitt":
            self.processed_image = EdgeDetection.apply_prewitt(self.original_image)
        elif type == "Canny":
            low = self.ui.edge_detection_low_threshold_spinbox.value()
            high = self.ui.edge_detection_high_threshold_spinbox.value()
            self.processed_image = EdgeDetection.apply_canny(self.original_image, low, high)

        self.showProcessed()

    def apply_fourier_filters(self, type="High"):
        if self.original_image is None:
            print("No image loaded. Please upload an image first.")
            return

        radius = self.ui.raduis_control_slider.value()

        if type == "High":
            self.processed_image = FourierFilters.apply_high_pass(self.original_image, radius)
        elif type == "Low":
            self.processed_image = FourierFilters.apply_low_pass(self.original_image, radius)

        self.showProcessed()


    def upload_low_frequency_image(self):
        self.path_1 = self.srv.upload_image_file()
        if self.path_1:
            self.low_frequency_image = cv2.imread(self.path_1)
            self.srv.clear_image(self.ui.low_frequency_groupbox)
            self.srv.set_image_in_groupbox(self.ui.low_frequency_groupbox, self.low_frequency_image)

    def upload_high_frequency_image(self):
        self.path_2 = self.srv.upload_image_file()
        if self.path_2:
            self.high_frequency_image = cv2.imread(self.path_2)
            self.srv.clear_image(self.ui.high_frequency_groupbox)
            self.srv.set_image_in_groupbox(self.ui.high_frequency_groupbox, self.high_frequency_image)

    def hybrid_generation(self):
        if self.low_frequency_image is None or self.high_frequency_image is None:
            print("Please upload both low-frequency and high-frequency images first.")
            return

        # Generate the hybrid image
        self.hybrid_image = self.hybrid_generator.generate_hybrid_image(
            self.low_frequency_image, self.high_frequency_image, self.hybrid_image
        )
        low_freq_filtered = FourierFilters.apply_low_pass(self.low_frequency_image, 20)
        self.srv.clear_image(self.ui.low_frequency_groupbox)
        self.srv.set_image_in_groupbox(self.ui.low_frequency_groupbox, low_freq_filtered)

        high_freq_filtered = FourierFilters.apply_high_pass(self.high_frequency_image, 5)
        self.srv.clear_image(self.ui.high_frequency_groupbox)
        self.srv.set_image_in_groupbox(self.ui.high_frequency_groupbox, high_freq_filtered)


        # Display the hybrid image in the hybrid_image_groupbox
        if self.hybrid_image is not None:
            self.srv.clear_image(self.ui.hybrid_image_groupbox)
            self.srv.set_image_in_groupbox(self.ui.hybrid_image_groupbox, self.hybrid_image)

    def drawImage(self):
        self.path = self.srv.upload_image_file()
        self.original_image = cv2.imread(self.path)
        self.processed_image = self.original_image

        # If user cancels file selection, path could be None
        if self.path:
            self.srv.clear_image(self.ui.original_groupBox)
            self.srv.clear_image(self.ui.processed_groupBox)
            self.srv.set_image_in_groupbox(self.ui.original_groupBox, self.original_image)
            self.srv.set_image_in_groupbox(self.ui.processed_groupBox, self.processed_image)

    def clear_images(self):
        if self.original_image is None:
            return

        self.srv.clear_image(self.ui.processed_groupBox)
        self.srv.clear_image(self.ui.original_groupBox)

    def reset_images(self):
        if self.original_image is None:
            return

        self.srv.clear_image(self.ui.processed_groupBox)
        self.srv.set_image_in_groupbox(self.ui.processed_groupBox, self.original_image)

    def showProcessed(self):
        if self.processed_image is None:
            print("Error: Processed image is None.")
            return  # Prevents crashing

        self.srv.clear_image(self.ui.processed_groupBox)
        self.srv.set_image_in_groupbox(self.ui.processed_groupBox, self.processed_image)

    def gray_image_converter(self):
        image = self.convert.rgb_to_gray(self.original_image)
        # Update processed image with the equalized image
        self.processed_image = image

        # Show the processed image
        self.showProcessed()
        return self.convert.rgb_to_gray(self.original_image)

    def normalize_image(self):
        """Apply normalization to the original image."""
        if self.original_image is None:
            print("No image available for equalization.")
            return

        img_normalized = self.normalize.normalize_image_rgb(self.original_image)

        # Update processed image with the equalized image
        self.processed_image = img_normalized

        # Show the processed image
        self.showProcessed()

    def equalize_image(self):
        """Apply histogram equalization to the original image."""
        if self.original_image is None:
            print("No image available for equalization.")
            return

        equalized_image = self.equalize.equalizeHistRGB(self.original_image)

        # Update processed image with the equalized image
        self.processed_image = equalized_image

        # Show the processed image
        self.showProcessed()

    def global_thresholding(self):
        """Apply global thresholding to the original image."""
        if self.original_image is None:
            print("No image available for equalization.")
            return

        # Convert to grayscale if the image is in color
        if len(self.original_image.shape) == 3:
            gray_image = self.convert.rgb_to_gray(self.original_image)
        else:
            gray_image = self.original_image

        threshold_value = self.ui.global_threshold_spinbox.value()
        binary_global = self.threshold.global_threshold(gray_image, threshold_value)

        # Update processed image with the equalized image
        self.processed_image = binary_global

        # Show the processed image
        self.showProcessed()

    def local_thresholding(self):
        """Apply local thresholding to the original image."""
        if self.original_image is None:
            print("No processed image available for equalization.")
            return

        # Convert to grayscale if the image is in color
        if len(self.original_image.shape) == 3:
            gray_image = self.convert.rgb_to_gray(self.original_image)
        else:
            gray_image = self.original_image

        # Apply custom local thresholding
        block_size = self.ui.local_block_size_spinbox.value()  # Size of the neighborhood
        binary_local = self.threshold.local_threshold(gray_image, block_size, )

        # Update processed image with the equalized image
        self.processed_image = binary_local

        # Show the processed image
        self.showProcessed()

    def closeApp(self):
        """Close the application."""
        remove_directories()
        self.app.quit()
