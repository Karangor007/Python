import cv2
import numpy as np

def enhance_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise using Gaussian Blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sharpen the image using a kernel
    kernel = np.array([[0, -1, 0], [-1, 4.8,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Morphological operations to remove small noises and marks
    kernel = np.ones((3, 3), np.uint8)
    morph_cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
    morph_cleaned = cv2.morphologyEx(morph_cleaned, cv2.MORPH_OPEN, kernel)

    # Edge detection using Canny
    edges = cv2.Canny(morph_cleaned, 50, 100)

    # Combine edges with the cleaned image
    result = cv2.bitwise_or(morph_cleaned, edges)

    # Save the result
    cv2.imwrite('enhanced_image.jpg', result)

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'images/image_blur_6.jpg'
    enhance_image(image_path)
