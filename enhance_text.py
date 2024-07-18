import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'images/image_blur_6.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Enhance the image for OCR using preprocessing techniques
def preprocess_image(image, clip_limit=2.0, threshold_value=180):
    # Increase contrast and intensity
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Adaptive thresholding to binarize the image
    _, thresh = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Optional: Morphological operations to improve text connectivity
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closing

# Preprocess the image with adjustable parameters
clip_limit = 2.3  # Adjust this to control contrast enhancement
threshold_value = 145  # Adjust this to control text boldness
preprocessed_image = preprocess_image(image, clip_limit, threshold_value)

# Display the preprocessed image
plt.figure(figsize=(8, 6))
plt.imshow(preprocessed_image, cmap='gray')
plt.title('Enhanced Image for OCR')
plt.axis('off')
plt.show()

# Save the preprocessed image
cv2.imwrite('enhanced_image.jpg', preprocessed_image)
