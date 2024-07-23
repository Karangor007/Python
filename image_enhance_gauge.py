import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk

def preprocess_image(image, alpha, beta, blur_kernel_size, sharpen_kernel_value, block_size, C):
    # Ensure kernel size and block size are odd
    blur_kernel_size = int(blur_kernel_size)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    block_size = int(block_size)
    if block_size % 2 == 0:
        block_size += 1

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

def update_image():
    alpha = alpha_slider.get()
    beta = beta_slider.get()
    blur_kernel_size = blur_kernel_slider.get()
    sharpen_kernel_value = sharpen_slider.get()
    block_size = block_size_slider.get()
    C = c_slider.get()
    preprocessed_image = preprocess_image(deskewed_image, alpha, beta, blur_kernel_size, sharpen_kernel_value, block_size, C)
    cv2.imwrite(output_path, preprocessed_image)
    
    # Show the images
    plt.figure(figsize=[18, 5])
    plt.subplot(131); plt.imshow(image, cmap='gray'); plt.title("Original")
    plt.subplot(132); plt.imshow(deskewed_image, cmap='gray'); plt.title("Deskewed")
    plt.subplot(133); plt.imshow(preprocessed_image, cmap='gray'); plt.title("Preprocessed")
    plt.show()

def on_submit():
    update_image()

def on_reset():
    alpha_slider.set(1.1)
    beta_slider.set(5)
    blur_kernel_slider.set(5)
    sharpen_slider.set(4.2)
    block_size_slider.set(9)
    c_slider.set(2)

def on_close():
    root.destroy()

def deskew_image(image):
    # Convert the image to binary
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def main():
    global image, deskewed_image, output_path
    global alpha_slider, beta_slider, blur_kernel_slider, sharpen_slider, block_size_slider, c_slider
    global root
    
    image_path = 'images/image_blur_4.jpg'
    output_path = 'images/output_image.jpg'

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    deskewed_image = deskew_image(image)

    # Create a window with sliders
    root = tk.Tk()
    root.title("Image Preprocessing Parameters")

    tk.Label(root, text="Contrast (alpha)").pack()
    alpha_slider = ttk.Scale(root, from_=1.0, to=3.0, orient="horizontal")
    alpha_slider.set(1.1)
    alpha_slider.pack()

    tk.Label(root, text="Brightness (beta)").pack()
    beta_slider = ttk.Scale(root, from_=0, to=100, orient="horizontal")
    beta_slider.set(5)
    beta_slider.pack()

    tk.Label(root, text="Gaussian Blur Kernel Size").pack()
    blur_kernel_slider = ttk.Scale(root, from_=1, to=31, orient="horizontal")
    blur_kernel_slider.set(5)
    blur_kernel_slider.pack()

    tk.Label(root, text="Sharpen Kernel Value").pack()
    sharpen_slider = ttk.Scale(root, from_=1.0, to=10.0, orient="horizontal")
    sharpen_slider.set(4.2)
    sharpen_slider.pack()

    tk.Label(root, text="Adaptive Threshold Block Size").pack()
    block_size_slider = ttk.Scale(root, from_=1, to=31, orient="horizontal")
    block_size_slider.set(9)
    block_size_slider.pack()

    tk.Label(root, text="Adaptive Threshold C Value").pack()
    c_slider = ttk.Scale(root, from_=1, to=10, orient="horizontal")
    c_slider.set(2)
    c_slider.pack()

    # Buttons
    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(side=tk.LEFT, padx=10, pady=10)

    reset_button = tk.Button(root, text="Reset", command=on_reset)
    reset_button.pack(side=tk.LEFT, padx=10, pady=10)

    close_button = tk.Button(root, text="Close", command=on_close)
    close_button.pack(side=tk.LEFT, padx=10, pady=10)

    root.mainloop()

# Run the main function
if __name__ == "__main__":
    main()
