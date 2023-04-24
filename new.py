from scipy.signal import wiener
import numpy as np
from matplotlib import pyplot as plt
import cv2
# load the image and convert it to grayscale
img = cv2.imread('/home/pc/Desktop/AI/source.jpeg', cv2.IMREAD_GRAYSCALE)

# apply the wiener filter
img_deblurred = wiener(img, (5,5))

# display the original image and the deblurred image
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(img_deblurred, cmap='gray')
plt.title('Deblurred Image')
plt.show()
