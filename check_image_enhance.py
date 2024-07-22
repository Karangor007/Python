import cv2
import numpy as np
from skimage import filters, exposure
from matplotlib import pyplot as plt

def is_image_good_for_ocr(image):
    # Calculate contrast
    contrast = image.max() - image.min()
    
    # Calculate sharpness using the Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    
    # Check for noise (simple method using standard deviation)
    noise = image.std()

    # Thresholds for determining if the image is good for OCR
    contrast_threshold = 50  # Example threshold
    sharpness_threshold = 100  # Example threshold
    noise_threshold = 10  # Example threshold

    return (contrast > contrast_threshold and
            laplacian > sharpness_threshold and
            noise < noise_threshold)

def preprocess_image(image):
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Sharpening the image using a kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # Adjust contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    
    # Apply adaptive thresholding to binarize the image
    thresholded = cv2.adaptiveThreshold(adjusted, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    
    return thresholded

def main(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if is_image_good_for_ocr(image):
        print("Image is good for OCR")
        cv2.imwrite(output_path, image)  # Save the original image
    else:
        print("Image needs preprocessing")
        preprocessed_image = preprocess_image(image)
        cv2.imwrite(output_path, preprocessed_image)  # Save the preprocessed image
    
    # Show the images
    plt.figure(figsize=[18, 5])
    plt.subplot(121); plt.imshow(image, cmap='gray'); plt.title("Original")
    plt.subplot(122); plt.imshow(preprocessed_image, cmap='gray'); plt.title("Preprocessed")
    plt.show()

# Example usage
main('images/Media.jpg', 'images/output_image.jpg')
