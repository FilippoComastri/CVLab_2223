# Write here your solution
# Import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Read image
img = cv2.imread('avengers.png')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# Invert image
img_negative = 255 - img_rgb
# Display Image
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.subplot(1,2,2)
plt.imshow(img_negative)
plt.show()