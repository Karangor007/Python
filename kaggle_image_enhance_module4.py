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

matrix1 = np.ones(nz_img_rgb.shape) * .8
matrix2 = np.ones(nz_img_rgb.shape) * 1.2

img_rgb_darker   = np.uint8(cv2.multiply(np.float64(nz_img_rgb), matrix1))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(nz_img_rgb), matrix2))

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Lower Contrast")
plt.subplot(132); plt.imshow(nz_img_rgb);         plt.title("Original")
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Higher Contrast")


matrix1 = np.ones(nz_img_rgb.shape) * .8
matrix2 = np.ones(nz_img_rgb.shape) * 1.2

img_rgb_lower   = np.uint8(cv2.multiply(np.float64(nz_img_rgb), matrix1))
img_rgb_higher  = np.uint8(np.clip(cv2.multiply(np.float64(nz_img_rgb), matrix2),0,255))

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_lower);  plt.title("Lower Contrast")
plt.subplot(132); plt.imshow(nz_img_rgb);     plt.title("Original")
plt.subplot(133); plt.imshow(img_rgb_higher); plt.title("Higher Contrast")


bw_img = cv2.imread('images/image_blur_6.jpg', cv2.IMREAD_GRAYSCALE)
retval, bw_img_thresh = cv2.threshold(bw_img, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(121); plt.imshow(bw_img, cmap="gray"); plt.title("Original")
plt.subplot(122); plt.imshow(bw_img_thresh, cmap="gray"); plt.title("Thresholded")

bw_img = cv2.imread('images/image_blur_6.jpg', cv2.IMREAD_GRAYSCALE)
bw_img_thresh = cv2.adaptiveThreshold(bw_img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(121); plt.imshow(bw_img, cmap="gray"); plt.title("Original")
plt.subplot(122); plt.imshow(bw_img_thresh, cmap="gray"); plt.title("Thresholded")

plt.show()  # Display the plots
