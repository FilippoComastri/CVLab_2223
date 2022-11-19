import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time


sigma = 1.5
#finding gaussian kernel and applying it
kernel_size = int(np.ceil(sigma*3)*2+1)
gk = cv.getGaussianKernel(kernel_size,sigma)
print('gk shape',gk.shape)
gk_2D = gk.dot(gk.transpose())
print('gk2D shape',gk_2D.shape)
img = cv.imread('ex/landscape.jpg',cv.IMREAD_GRAYSCALE)
out_img = cv.filter2D(img,-1,gk_2D)
plt.subplot(1,2,1)
plt.title('original img')
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.subplot(1,2,2)
plt.title('gaussian filter')
plt.imshow(out_img,cmap='gray',vmin=0,vmax=255)
plt.show()

cv.waitKey(0)

#using opencv function
img = cv.imread('ex/landscape.jpg',cv.IMREAD_GRAYSCALE)
out_img = cv.GaussianBlur(img,(kernel_size,kernel_size),sigma)
print(out_img.dtype)
plt.subplot(1,2,1)
plt.title('original img')
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.subplot(1,2,2)
plt.title('gaussian filter')
plt.imshow(out_img,cmap='gray',vmin=0,vmax=255)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
