import cv2
from matplotlib import pyplot as plt
import numpy as np

# Read the image in BGR format
nz_img_bgr = cv2.imread('images/image_blur_6.jpg')

# Convert BGR to RGB
nz_img_rgb = cv2.cvtColor(nz_img_bgr, cv2.COLOR_BGR2RGB)

# Create a matrix to adjust brightness
matrix = np.ones(nz_img_rgb.shape, dtype="uint8") * 50
nz_img_brighter = cv2.add(nz_img_bgr, matrix)
nz_img_darker = cv2.subtract(nz_img_bgr, matrix)

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(nz_img_darker); plt.title("Darker")
plt.subplot(132); plt.imshow(nz_img_brighter); plt.title("Brighter")
plt.subplot(133); plt.imshow(nz_img_rgb); plt.title("Original")

plt.show()  # Display the plots
