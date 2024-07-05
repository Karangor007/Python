import cv2

def clean_image(image_path):
  """
  This function cleans an image using blurring and adaptive thresholding.

  Args:
      image_path: Path to the image file.

  Returns:
      A cleaned image as a NumPy array.
  """
  # Read the image
  img = cv2.imread(image_path)

  # Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Apply blurring to reduce noise
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  # Apply adaptive thresholding to enhance foreground
  thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

  # You can further customize the cleaning process here
  # ... (e.g., apply morphology operations like erosion or dilation)

  return thresh

# Example usage
image_path = "images/ss1.png"  # Replace with your image path
cleaned_image = clean_image(image_path)

# Display the cleaned image
cv2.imshow("Cleaned Image", cleaned_image)
cv2.waitKey(0)

# Save the cleaned image (optional)
cv2.imwrite("cleaned_image.jpg", cleaned_image)
