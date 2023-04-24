import cv2
import numpy as np

# Load the image
img = cv2.imread('path/to/image.jpg')

# Resize the image to match the input shape of the model
img = cv2.resize(img, (28, 28))

# Convert the image to a numerical array
img_array = np.array(img)
