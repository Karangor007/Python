import cv2
from PIL import Image
import numpy as np

def check_dpi(image_path):
    """Check the DPI of an image."""
    with Image.open(image_path) as img:
        dpi = img.info.get('dpi', (72, 72))  # Default DPI is 72 if not specified
    return dpi

def increase_dpi(image_path, output_path, dpi=(300, 300)):
    """Increase the DPI of an image and save it."""
    with Image.open(image_path) as img:
        img.save(output_path, dpi=dpi)

def display_images(original_image_path, modified_image_path):
    """Display the original and modified images using OpenCV."""
    original_image = cv2.imread(original_image_path)
    modified_image = cv2.imread(modified_image_path)

    # Stack images horizontally for comparison
    comparison = np.hstack((original_image, modified_image))

    # Display the images
    cv2.imshow('Original and Modified Images', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path, output_path, desired_dpi=(300, 300)):
    """Check and increase the DPI of an image if needed."""
    current_dpi = check_dpi(image_path)
    print(f"Current DPI: {current_dpi}")

    if current_dpi[0] < desired_dpi[0] or current_dpi[1] < desired_dpi[1]:
        print(f"Increasing DPI to: {desired_dpi}")
        increase_dpi(image_path, output_path, dpi=desired_dpi)
        new_dpi = check_dpi(output_path)
        print(f"New DPI: {new_dpi}")
    else:
        print("DPI is already higher than or equal to the desired DPI.")
        # Save the original image as output for display purposes
        cv2.imwrite(output_path, cv2.imread(image_path))

    # Display the original and modified images
    display_images(image_path, output_path)

# Example usage
main('images/image_blur_6.jpg', 'images/output_image_with_higher_dpi.jpg')
