import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


kernel = np.ones((9,9))/81
img = cv.imread('ex/landscape.jpg',cv.IMREAD_GRAYSCALE)
out_img = cv.filter2D(img,-1,kernel)

plt.subplot(1,2,1)
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.subplot(1,2,2)
plt.imshow(out_img,cmap='gray',vmin=0,vmax=255)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
