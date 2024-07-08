import cv2
import numpy as np

def get_line_contours(thresh_img):
  """
  This function identifies contours corresponding to lines in the image.

  Args:
      thresh_img: The thresholded image (binary image with edges).

  Returns:
      A list of contours representing potential lines in the image.
  """
  # Find contours in the thresholded image
  contours, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  return contours

def select_line_for_removal(contours, image):
  """
  This function allows user selection of the line to remove (optional).

  Args:
      contours: List of contours from the image.
      image: The original image (BGR format).

  Returns:
      The index of the contour corresponding to the line chosen for removal,
      or None if no line is selected.
  """
  # Draw contours on a copy of the image for visualization
  vis_img = image.copy()
  cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)

  # Display the image with contours and prompt user selection (optional)
  # You can implement user interaction here (e.g., click on desired line)
  # to select the appropriate contour index.
  # ...

  # Example: Assuming user selects contour at index 2
  selected_contour_index = 2
  return selected_contour_index

def remove_line_using_mask(image, inpaint_mask, selected_contour_index=None):
  """
  This function removes the line based on the inpaint mask or a selected contour.

  Args:
      image: The original image (BGR format).
      inpaint_mask: The inpainting mask (areas to be inpainted).
      selected_contour_index: Index of the contour to remove (optional).

  Returns:
      The image with the line removed.
  """
  if selected_contour_index is not None:
    # Refine mask based on selected contour
    x, y, w, h = cv2.boundingRect(contours[selected_contour_index])
    inpaint_mask[y:y+h, x:x+w] = 255  # Set contour area to white in mask
  
  # Inpaint the image using the refined mask
  inpaint_result = cv2.inpaint(image, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
  return inpaint_result

# Read the image
image_path = 'images/ss1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform edge detection (Sobel operator)
edges = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, ksize=3)

# Thresholding to binarize the edges
_, thresh = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

# Identify potential line contours
contours = get_line_contours(thresh)

# Optionally, allow user selection of the line to remove
selected_contour_index = select_line_for_removal(contours, image.copy())

# Create inpainting mask and remove the line
inpaint_mask = cv2.bitwise_not(thresh)
inpainted_image = remove_line_using_mask(image, inpaint_mask, selected_contour_index)

# Display results (optional)
# You can comment out this section if you don't want to display the images
cv2.imshow("Original Image", image)
cv2.imshow("Image with Inpainting Mask", inpaint_mask)
cv2.imshow("Image with Line Removed", inpainted_image)
cv2.waitKey(0)

# Save the image with the line removed (optional)
# cv2.imwrite("image_with_line_removed.jpg", inpainted_image)
