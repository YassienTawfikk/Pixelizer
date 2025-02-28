from PyQt5 import QtWidgets
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices
from app.design.main_layout import Ui_MainWindow
from app.processing.edge_detection import EdgeDetection
from app.processing.histogram import ImageHistogram
from app.processing.noise_amount import AddingNoise
from app.processing.denoise import Denoise

import cv2


class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()

        self.path = None
        self.original_image = None
        self.processed_image = None

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        self.srv = ImageServices()

        self.edge = EdgeDetection()
        self.noise = AddingNoise()
        self.denoise = Denoise()

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

        # self.ui.show_metrics_button.clicked.connect(lambda: ImageHistogram.show_histogram_popup(self.path))
        self.ui.show_metrics_button.clicked.connect(self.ui.popup.show_popup)

    def apply_noise(self, type="Uniform"):
        if self.original_image is None:
            print("No image loaded. Please upload an image first.")
            return  # Prevents crashing

        if type == "Uniform":
            amount = self.ui.uniform_noise_slider.value() / 100
            print(amount)
            self.processed_image = self.noise.add_uniform_noise(self.original_image, amount)
        elif type == "Gaussian":
            mean = self.ui.mean_gaussian_noise_slider.value()
            stdev = self.ui.stddev_gaussian_noise_slider.value()
            self.processed_image = self.noise.add_gaussian_noise(self.original_image, mean, stdev)
        elif type == "Salt&Pepper":
            amount = self.ui.salt_pepper_noise_slider.value() / 100
            self.processed_image = self.noise.add_salt_and_pepper_noise(self.original_image, amount)

        self.showProcessed()

    def remove_noise(self, type="Average"):
        if self.original_image is None:
            print("No image loaded. Please upload an image first.")
            return  # Prevents crashing
        
        # kernel_size=self.ui.filter_kernal_size_button.value()
        kernel_size=5
        sigma=self.ui.gaussian_filter_sigma_spinbox.value()

        if type == "Average":
            self.processed_image = self.denoise.apply_average_filter(self.processed_image, kernel_size)
        elif type == "Gaussian":
            self.processed_image = self.denoise.apply_gaussian_filter(self.processed_image, kernel_size, sigma)
        elif type == "Median":
            self.processed_image = self.denoise.apply_median_filter(self.processed_image, kernel_size)

        self.showProcessed()


    def edge_detection(self, type="Sobel"):
        if self.original_image is None:
            print("No image loaded. Please upload an image first.")
            return  # Prevents crashing

        if len(self.original_image.shape) == 3:
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

        if type == "Sobel":
            self.processed_image = self.edge.apply_sobel(self.original_image)
        elif type == "Roberts":
            self.processed_image = self.edge.apply_roberts(self.original_image)
        elif type == "Prewitt":
            self.processed_image = self.edge.apply_prewitt(self.original_image)
        elif type == "Canny":
            low = self.ui.edge_detection_low_threshold_spinbox.value()
            high = self.ui.edge_detection_high_threshold_spinbox.value()
            self.processed_image = self.edge.apply_canny(self.original_image, low, high)

        self.showProcessed()

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

    def reset_images(self):
        self.srv.clear_image(self.ui.processed_groupBox)
        self.srv.clear_image(self.ui.original_groupBox)

    def showProcessed(self):
        if self.processed_image is None:
            print("Error: Processed image is None.")
            return  # Prevents crashing

        self.srv.clear_image(self.ui.processed_groupBox)
        self.srv.set_image_in_groupbox(self.ui.processed_groupBox, self.processed_image)

    def closeApp(self):
        """Close the application."""
        remove_directories()
        self.app.quit()
