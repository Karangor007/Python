import cv2
import numpy as np

# Load the image
image = cv2.imread('images/image_blur_6.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (9, 9), 15.0)

# Apply Unsharp Masking
unsharp_image = cv2.addWeighted(image, 1.35, gaussian_blur, -0.6, 0)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', unsharp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
