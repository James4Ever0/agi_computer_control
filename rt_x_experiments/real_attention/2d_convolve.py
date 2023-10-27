import numpy as np
from scipy.signal import convolve2d

# Define the input image and the kernel
image = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])

# Perform 2D convolution
convolved_image = convolve2d(image, kernel, mode='valid')

# Print the resulting convolved image
print(image.shape, kernel.shape, convolved_image.shape)
# so this might works.
# (3, 3) (3, 3) (1, 1)


################ EXTRA COMPUTATION MIGHT INVOLVED ##################

# import cv2

# # Read the input image
# image = cv2.imread('image.jpg')

# # Define the kernel
# kernel = np.array([[0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]])

# # Apply filter2D
# filtered_image = cv2.filter2D(image, -1, kernel)
