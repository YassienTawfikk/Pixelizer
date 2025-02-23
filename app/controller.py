from PyQt5 import QtCore, QtGui, QtWidgets
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices
from app.design.design import Ui_MainWindow
from app.processing.EdgeDetection import EdgeDetection
import cv2


# from app.tests.design_test import Ui_MainWindow


class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()

        self.path=None
        self.original_image=None
        self.processed_image=None

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        self.edge = EdgeDetection()

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
        self.ui.sobel_edge_detection_button.clicked.connect(self.controller_apply_sobel)

    def controller_apply_sobel(self):
        self.processed_image=self.edge.apply_sobel(self.original_image)
        ImageServices.set_image_in_groupbox(self.ui.processed_groupBox, self.path)


    def drawImage(self):
        self.path = ImageServices.upload_image_file()
        self.original_image = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)

        # If user cancels file selection, path could be None
        if self.path:
            ImageServices.clear_image(self.ui.original_groupBox)
            ImageServices.set_image_in_groupbox(self.ui.original_groupBox, self.path)

    def closeApp(self):
        """Close the application."""
        remove_directories()
        self.app.quit()
