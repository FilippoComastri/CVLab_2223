import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('grayscale.jpg', cv2.IMREAD_GRAYSCALE)
img_negative = 255 - img
#img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(1,2,2)
plt.imshow(img_negative, cmap='gray', vmin=0, vmax=255)
plt.show()
