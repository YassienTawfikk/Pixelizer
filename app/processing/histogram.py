# import cv2
# import numpy as np
# from PyQt5 import QtWidgets
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure
#
#
# class ImageHistogram:
#     @staticmethod
#     def show_histogram_popup(image_path):
#         """
#         Displays the histogram of the image in a pop-up window as bars.
#         Uses NumPy to calculate the histogram and Matplotlib to draw it.
#         """
#         # Read the image using OpenCV (in grayscale mode)
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#         # Calculate the histogram manually using NumPy
#         hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))
#
#         # Create a new window to display the histogram
#         histogram_window = QtWidgets.QDialog()
#         histogram_window.setWindowTitle("Image Histogram")
#         histogram_window.setGeometry(100, 100, 800, 600)
#
#         # Create a Matplotlib figure and canvas
#         fig = Figure()
#         canvas = FigureCanvas(fig)
#         ax = fig.add_subplot(111)
#
#         # Plot the histogram as bars using Matplotlib
#         ax.grid(True, linestyle='--', alpha=0.6)
#         ax.bar(bin_edges[:-1], hist, width=1, color='blue', edgecolor='black', linewidth=0.5)  # Use bin_edges[:-1] to match hist size
#         ax.set_facecolor('#f7f7f7')  # Light gray background
#         ax.set_title('Histogram (Bar Chart)')
#         ax.set_xlabel('Pixel Intensity')
#         ax.set_ylabel('Frequency')
#         ax.set_xlim([0, 256])  # Set x-axis limits to match pixel intensity range
#
#         # Add the canvas to the window
#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(canvas)
#         histogram_window.setLayout(layout)
#
#         # Show the window
#         histogram_window.exec_()