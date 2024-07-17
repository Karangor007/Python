import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def blur(image):
    return cv2.medianBlur(image, 3)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def make_text_bold(image, kernel_size=(2, 2), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)

def make_text_less_bold(image, kernel_size=(2, 2), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(image, kernel, iterations=iterations)

def process_image(boldness):
    # Load the image
    image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not image_path:
        return

    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = get_grayscale(image)

    # Apply blur
    blurred = blur(gray)

    # Apply thresholding
    thresh = thresholding(blurred)

    # Adjust boldness
    if boldness > 0:
        processed_image = make_text_bold(thresh, kernel_size=(2, 2), iterations=boldness)
    else:
        processed_image = make_text_less_bold(thresh, kernel_size=(2, 2), iterations=abs(boldness))

    # Save the resulting image
    result_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if result_path:
        cv2.imwrite(result_path, processed_image)

    # Display the images
    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Get user input for boldness level
    boldness = simpledialog.askinteger("Input", "Enter boldness level (-5 to 5):", minvalue=-15, maxvalue=15)

    if boldness is not None:
        process_image(boldness)

if __name__ == "__main__":
    main()
