import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


kernel_denoising = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
kernel_hp = np.array([[0,1,0],[1,-4,1],[0,1,0]])
img = cv.imread('ex/landscape.jpg',cv.IMREAD_GRAYSCALE)
out_img_den = cv.filter2D(img,-1,kernel_denoising)
out_img_hp = cv.filter2D(img,-1,kernel_hp)
plt.subplot(1,3,1)
plt.title('original img')
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.subplot(1,3,2)
plt.title('denoising filter')
plt.imshow(out_img_den,cmap='gray',vmin=0,vmax=255)
plt.subplot(1,3,3)
plt.title('high pass filter')
plt.imshow(out_img_hp,cmap='gray',vmin=0,vmax=255)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
