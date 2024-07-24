import cv2
import numpy as np
from matplotlib import pyplot as plt

def is_image_good_for_ocr(image):
    # Calculate contrast
    contrast = image.max() - image.min()
    print("image contrast value")
    print(contrast)

    # Calculate sharpness using the Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    print("image laplacian value")
    print(laplacian)

    # Check for noise (simple method using standard deviation)
    noise = image.std()
    print("image noise value")
    print(noise)
    # Thresholds for determining if the image is good for OCR
    contrast_threshold = 250  # Example threshold
    sharpness_threshold = 4810  # Example threshold
    noise_threshold = 50  # Example threshold

    print("final result")
    print("contrast > contrast_threshold : ",contrast > contrast_threshold)
    print("laplacian > sharpness_threshold :",laplacian > sharpness_threshold)
    print("noise > noise_threshold : ",noise >noise_threshold)

    print(contrast > contrast_threshold and
            laplacian > sharpness_threshold and
            noise > noise_threshold)

    return (contrast > contrast_threshold and
            laplacian > sharpness_threshold and
            noise > noise_threshold)

def deskew_image(image):
    # Convert to binary image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(binary > 0))

    # Find the angle of the rotated bounding box
    angle = cv2.minAreaRect(coords)[-1]
    print("angle")
    print(angle)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = 0

    print("after angle")
    print(angle)
    # Rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed

def preprocess_image(image):
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Sharpening the image using a kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4.35, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # Adjust contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 4    # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    
    # Apply adaptive thresholding to binarize the image
    thresholded = cv2.adaptiveThreshold(adjusted, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 13, 1.5)
    
    return thresholded

def main(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Deskew the image
    deskewed_image = deskew_image(image)

    if is_image_good_for_ocr(deskewed_image):
        print("Image is good for OCR")
        preprocessed_image = deskewed_image
        cv2.imwrite(output_path, deskewed_image)  # Save the original image
    else:
        print("Image needs preprocessing")
        preprocessed_image = preprocess_image(deskewed_image)
        cv2.imwrite(output_path, preprocessed_image)  # Save the preprocessed image
    
    # Show the images
    plt.figure(figsize=[18, 5])
    plt.subplot(131); plt.imshow(image, cmap='gray'); plt.title("Original")
    #plt.subplot(132); plt.imshow(deskewed_image, cmap='gray'); plt.title("Deskewed")
    plt.subplot(132); plt.imshow(preprocessed_image, cmap='gray'); plt.title("Preprocessed")
    plt.show()

# Example usage
main('images/image_blur_6.jpg', 'images/output_image.jpg')
