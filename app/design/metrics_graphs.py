import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from app.design.tools.gui_utilities import GUIUtilities


class Ui_PopWindow:
    def __init__(self):
        self.util = GUIUtilities()
        self.button_style = """
            QPushButton {
                background-color: rgb(30, 30, 30);
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
                border: 1px solid white;
            }
            QPushButton:hover {
                background-color: rgb(40, 40, 40);
            }
        """

    def setupUi(self, Dialog):
        screen = QtWidgets.QApplication.primaryScreen().size()
        width = int(screen.width() * 0.75)  # 75% of the screen width
        height = int(screen.height() * 0.9)  # 90% of the screen height

        Dialog.setObjectName("Dialog")
        Dialog.resize(width, height)
        Dialog.setWindowFlags(QtCore.Qt.FramelessWindowHint)  # Frameless window
        Dialog.setStyleSheet("background-color: rgb(10, 10, 10);")

        # Calculate center position
        centerX = (screen.width() - width) // 2
        centerY = (screen.height() - height) // 2
        Dialog.move(centerX, centerY)  # Move window to center

        self.layout = QtWidgets.QVBoxLayout(Dialog)  # Set layout on QDialog directly
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)

        group_box_style = "color: white; border: 1px solid white;"

        self.histogram_groupBox, self.histogram_plot_widget = self.util.createGroupBox(
            "Histogram",
            QtCore.QSize(int(width * 0.95), int(height * 0.4)),
            group_box_style,
            isGraph=True
        )
        self.layout.addWidget(self.histogram_groupBox)

        self.distribution_groupBox, self.distribution_plot_widget = self.util.createGroupBox(
            "Distribution Curve",
            QtCore.QSize(int(width * 0.95), int(height * 0.4)),
            group_box_style,
            isGraph=True
        )
        self.layout.addWidget(self.distribution_groupBox)

        self.close_button = self.util.createButton("Close", self.button_style)
        self.close_button.clicked.connect(Dialog.accept)  # Close the dialog on button click

        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.buttonLayout.addStretch(1)
        self.buttonLayout.addWidget(self.close_button)
        self.buttonLayout.addStretch(1)
        self.layout.addLayout(self.buttonLayout)

    def plot_histogram(self, image_path):
        """Plot the histogram of the image."""
        if not image_path:
            print("Error: No image path provided.")
            return

        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Error: Unable to load image.")
            return

        # Calculate the histogram manually using NumPy
        hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))

        # Ensure histogram_plot_widget is properly initialized
        if not hasattr(self, 'histogram_plot_widget') or self.histogram_plot_widget is None:
            print("Error: histogram_plot_widget is not initialized.")
            return

        # Clear previous plot
        self.histogram_plot_widget.ax.clear()

        # Plot histogram
        self.histogram_plot_widget.ax.bar(bin_edges[:-1], hist, width=1, color='blue', edgecolor='black', linewidth=0.5)

        # Set axis labels and title
        self.histogram_plot_widget.ax.set_xlabel("Pixel Intensity")
        self.histogram_plot_widget.ax.set_ylabel("Pixel Count")
        self.histogram_plot_widget.ax.set_title("Histogram of Grayscale Image")

        # Force the canvas to update
        self.histogram_plot_widget.draw()

    def plot_cdf(self, image_path):
        """Plot the CDF of the image."""
        if not image_path:
            print("Error: No image path provided.")
            return

        # Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Error: Unable to load image.")
            return

        # Compute histogram
        hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))

        # Compute CDF
        cdf = hist.cumsum()  # Cumulative sum of histogram
        cdf_normalized = cdf / cdf[-1]  # Normalize to scale between 0 and 1

        print("Plotting CDF...")

        # Ensure distribution_plot_widget is properly initialized
        if not hasattr(self, 'distribution_plot_widget') or self.distribution_plot_widget is None:
            print("Error: distribution_plot_widget is not initialized.")
            return

        # Clear previous plot
        self.distribution_plot_widget.ax.clear()

        # Plot CDF
        self.distribution_plot_widget.ax.plot(bin_edges[:-1], cdf_normalized, color='red', linewidth=2)

        # Set axis labels and title
        self.distribution_plot_widget.ax.set_xlabel("Pixel Intensity")
        self.distribution_plot_widget.ax.set_ylabel("Cumulative Probability")
        self.distribution_plot_widget.ax.set_title("Cumulative Distribution Function (CDF)")

        # Force the canvas to update
        self.distribution_plot_widget.draw()

    def show_popup(self):
        """ Method to show the pop-up window as a modal dialog. """
        self.Dialog = QtWidgets.QDialog()
        self.setupUi(self.Dialog)
        self.Dialog.exec_()
