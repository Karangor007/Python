import cv2
import numpy as np

def enhance_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply a Gaussian Blur to reduce noise and smoothen the image
    gaussian_blur = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply a sharpening kernel
    sharpening_kernel = np.array([[-1, -1, -1], 
                                  [-1,  9, -1], 
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(gaussian_blur, -1, sharpening_kernel)

    # Apply edge detection (Canny)
    edges = cv2.Canny(sharpened, 100, 200)

    # Combine the sharpened image with the edges for enhancement
    enhanced_image = cv2.addWeighted(sharpened, 0.8, edges, 0.2, 0)

    return enhanced_image

# Load and enhance the image
image_path = 'images/image_blur_6.jpg'
enhanced_image = enhance_image(image_path)

# Display the result
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the enhanced image
cv2.imwrite('images/image_blur_6.jpg', enhanced_image)
