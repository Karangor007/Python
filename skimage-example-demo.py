import matplotlib.pyplot as plt
from skimage import io, color, filters, img_as_ubyte

# Read an image from a file
image = io.imread('images/ss1.jpg')

# Check if the image has an alpha channel
if image.shape[2] == 4:
    # Convert RGBA to RGB
    image = color.rgba2rgb(image)

# Convert the image to grayscale
gray_image = color.rgb2gray(image)

# Apply a Gaussian filter to the image
blurred_image = filters.gaussian(gray_image, sigma=1)

# Convert the floating-point image to uint8
blurred_image_uint8 = img_as_ubyte(blurred_image)

# Save the result to a file
io.imsave('images/blurred_image.jpg', blurred_image_uint8)

# Display the original and processed images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(blurred_image_uint8, cmap='gray')
ax[1].set_title('Blurred Image')
ax[1].axis('off')

plt.show()
