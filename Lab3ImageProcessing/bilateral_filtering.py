import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

img = cv.imread('ex/landscape.jpg',cv.IMREAD_GRAYSCALE)
out_img = cv.bilateralFilter(img,9,75,75)
plt.subplot(1,2,1)
plt.title('original img')
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.subplot(1,2,2)
plt.title('bilateral filter')
plt.imshow(out_img,cmap='gray',vmin=0,vmax=255)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
