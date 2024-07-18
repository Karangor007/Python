import cv2
import numpy as np

# Load the image
image = cv2.imread('images/image_blur_6.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Define a kernel
kernel = np.ones((3, 3), np.uint8)

# Apply dilation
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Apply erosion
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Apply opening (erosion followed by dilation)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Apply closing (dilation followed by erosion)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Opened Image', opened_image)
cv2.imshow('Closed Image', closed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
