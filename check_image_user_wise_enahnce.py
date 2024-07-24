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
    print("contrast > contrast_threshold : ", contrast > contrast_threshold)
    print("laplacian > sharpness_threshold :", laplacian > sharpness_threshold)
    print("noise > noise_threshold : ", noise > noise_threshold)

    print(contrast > contrast_threshold and
          laplacian > sharpness_threshold and
          noise > noise_threshold)

    return (contrast > contrast_threshold and
            laplacian > sharpness_threshold and
            noise > noise_threshold)

def deskew_image(image):
    # Convert the image to binary
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = 0
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def preprocess_image(image, alpha, beta, blur_kernel_size, sharpen_kernel_value, block_size, C):
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    
    # Sharpening the image using a kernel
    kernel = np.array([[0, -1, 0],
                       [-1, sharpen_kernel_value, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    # Adjust contrast
    adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    
    # Apply adaptive thresholding to binarize the image
    thresholded = cv2.adaptiveThreshold(adjusted, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block_size, C)
    return thresholded

def main():
    image_path = 'images/invoice_poc_blur.jpg'
    output_path = 'images/output_image.jpg'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    deskewed_image = deskew_image(image)
    
    if is_image_good_for_ocr(deskewed_image):
        print("Image is good for OCR")
        preprocessed_image = deskewed_image
        cv2.imwrite(output_path, deskewed_image)
    else:
        print("Image needs preprocessing")
        
        while True:
            alpha = float(input("Enter contrast adjustment (alpha) (e.g., 1.1): "))
            beta = int(input("Enter brightness adjustment (beta) (e.g., 5): "))
            blur_kernel_size = int(input("Enter Gaussian blur kernel size (odd number, e.g., 5): "))
            sharpen_kernel_value = float(input("Enter sharpen kernel value (e.g., 4.2): "))
            block_size = int(input("Enter adaptive threshold block size (odd number, e.g., 9): "))
            C = int(input("Enter adaptive threshold C value (e.g., 2): "))
            
            preprocessed_image = preprocess_image(deskewed_image, alpha, beta, blur_kernel_size, sharpen_kernel_value, block_size, C)
            cv2.imwrite(output_path, preprocessed_image)
            
            # Show the images
            plt.figure(figsize=[18, 5])
            plt.subplot(131); plt.imshow(image, cmap='gray'); plt.title("Original")
            #plt.subplot(132); plt.imshow(deskewed_image, cmap='gray'); plt.title("Deskewed")
            plt.subplot(133); plt.imshow(preprocessed_image, cmap='gray'); plt.title("Preprocessed")
            plt.show()
            
            retry = input("Do you want to adjust the parameters again? (yes/no): ").strip().lower()
            if retry != 'yes':
                break

# Run the main function
if __name__ == "__main__":
    main()
