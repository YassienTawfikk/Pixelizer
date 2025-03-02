# def generate_hybrid_image(self):
#     """
#     Generates the hybrid image by combining the low-frequency and high-frequency images.
#     """
#     if self.low_frequency_image is not None and self.high_frequency_image is not None:
#         # Ensure both images are the same size
#         self.high_frequency_image = cv2.resize(self.high_frequency_image, (
#             self.low_frequency_image.shape[1], self.low_frequency_image.shape[0]))
#
#         # Apply low-pass filter to the low-frequency image
#         low_pass = cv2.GaussianBlur(self.low_frequency_image, (21, 21), 0)
#
#         # Apply high-pass filter to the high-frequency image
#         high_pass = self.high_frequency_image - cv2.GaussianBlur(self.high_frequency_image, (21, 21), 0)
#
#         # Combine the images
#         self.hybrid_image = cv2.addWeighted(low_pass, 0.5, high_pass, 0.5, 0)
#
#         # Display the hybrid image
#         self.srv.clear_image(self.ui.hybrid_image_groupbox)
#         self.srv.set_image_in_groupbox(self.ui.hybrid_image_groupbox, self.hybrid_image)