import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import io, color, restoration

# Read an image from a local folder
image_path = 'images/ss1.jpg'
image = io.imread(image_path)

# Handle images with an alpha channel
if image.shape[2] == 4:
    image = image[:, :, :3]  # Discard the alpha channel

# Convert to grayscale
astro = color.rgb2gray(image)

# Define a PSF (Point Spread Function)
psf = np.ones((5, 5)) / 25

# Blur the image (simulate the optical blur)
astro = conv2(astro, psf, 'same')

# Add noise to the image
rng = np.random.default_rng()
astro_noisy = astro.copy()
astro_noisy += (rng.poisson(lam=25, size=astro.shape) - 10) / 255.0

# Restore the image using Richardson-Lucy algorithm
deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)

# Plotting the results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(astro_noisy)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
ax[2].set_title('Restoration using\nRichardson-Lucy')

fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()
