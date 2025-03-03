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
            QtCore.QSize(int(width * 0.95), int(height * 0.45)),
            group_box_style,
            isGraph=True
        )
        self.layout.addWidget(self.histogram_groupBox)

        self.distribution_groupBox, self.distribution_plot_widget = self.util.createGroupBox(
            "Distribution Curve",
            QtCore.QSize(int(width * 0.95), int(height * 0.45)),
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

    def process_image(self, image_path):
        """Loads a grayscale image and computes its histogram."""
        if not image_path:
            print("Error: No image path provided.")
            return None, None

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Error: Unable to load image.")
            return None, None

        hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))
        return hist, bin_edges

    def plot_graph(self, ax, x, y_hist, y_cdf, title):
        """General function to plot histogram or CDF with enhanced styling."""
        ax.clear()

        ax.bar(x[:-1], y_hist, width=1, color="#F88379", edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.set_title("Histogram and CDF for " + title + " image", fontsize=14, fontweight="bold", color="lightblue")

        ax.plot(x[:-1], y_cdf, color="#FF5733", linestyle="--", linewidth=1, marker="o", markersize=3,
                markerfacecolor="white")

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_facecolor("#202020")
        ax.figure.set_facecolor("#101010")
        ax.tick_params(axis="both", colors="white", labelsize=10)

    def plot_graph_RGB(self, ax, x, y_hist, y_cdf, hist_color, cdf_color, title):
        """General function to plot histogram or CDF with enhanced styling."""

        ax.bar(x[:-1], y_hist, width=1, color=hist_color, edgecolor="black", linewidth=0.5, alpha=0.8)

        ax.plot(x[:-1], y_cdf, color=cdf_color, linestyle="--", linewidth=1, marker="o", markersize=3,
                markerfacecolor="white")

        ax.set_title("Histogram and CDF for " + title + " image", fontsize=14, fontweight="bold", color="lightblue")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_facecolor("#202020")
        ax.figure.set_facecolor("#101010")
        ax.tick_params(axis="both", colors="white", labelsize=10)

    def plot_histogram_cdf(self, image, plot="processed"):
        """Plots CDF using the refactored functions."""

        if not hasattr(self, 'distribution_plot_widget') or self.distribution_plot_widget is None:
            print("Error: distribution_plot_widget is not initialized.")
            return

        if not hasattr(self, 'histogram_plot_widget') or self.histogram_plot_widget is None:
            print("Error: distribution_plot_widget is not initialized.")
            return

        if plot == "processed":
            self.distribution_plot_widget.ax.clear()
        else:
            self.histogram_plot_widget.ax.clear()

        if len(image.shape) == 3:
            hist_colors = ["#FF0000", "#008000", "#0000FF"]
            cdf_colors = ["#FF7F7F", "#90EE90", "#ADD8E6"]
            for i in range(3):
                hist, bins = np.histogram(image[:, :, i].flatten(), bins=256, range=[0, 256])
                pdf = hist / np.sum(hist)
                cdf = np.cumsum(pdf)
                if plot == "processed":
                    self.plot_graph_RGB(self.distribution_plot_widget.ax, bins, hist, cdf * np.max(hist), hist_colors[i], cdf_colors[i], plot)
                    self.distribution_plot_widget.draw()
                else:
                    self.plot_graph_RGB(self.histogram_plot_widget.ax, bins, hist, cdf * np.max(hist), hist_colors[i], cdf_colors[i], plot)
                    self.histogram_plot_widget.draw()

        else:
            hist, bins = np.histogram(image, bins=256, range=(0, 256))
            if hist is None:
                return
            pdf = hist / np.sum(hist)
            cdf = np.cumsum(pdf) * np.max(hist)
            if plot == "processed":
                self.plot_graph(self.distribution_plot_widget.ax, bins, hist, cdf, plot)
                self.distribution_plot_widget.draw()
            else:
                self.plot_graph(self.histogram_plot_widget.ax, bins, hist, cdf, plot)
                self.histogram_plot_widget.draw()

    def show_popup(self):
        """ Method to show the pop-up window as a modal dialog. """
        self.Dialog = QtWidgets.QDialog()
        self.setupUi(self.Dialog)
        self.Dialog.exec_()
