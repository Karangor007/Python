import cv2
import numpy as np

# Load the image
image = cv2.imread('images/image_blur_6.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Apply Prewitt edge detection (manually defined kernels)
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = cv2.filter2D(image, cv2.CV_64F, prewitt_kernel_x)
prewitt_y = cv2.filter2D(image, cv2.CV_64F, prewitt_kernel_y)
prewitt_combined = cv2.magnitude(prewitt_x, prewitt_y)

# Apply Canny edge detection
canny_edges = cv2.Canny(image, 100, 200)

# Display the results
cv2.imshow('Sobel Edge Detection', sobel_combined)
cv2.imshow('Prewitt Edge Detection', prewitt_combined)
cv2.imshow('Canny Edge Detection', canny_edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
