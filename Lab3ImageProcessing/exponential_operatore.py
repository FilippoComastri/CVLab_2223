import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def exp_op(img,r):
    out = ((img/255.)**r)*255.
    return out

img = cv.imread('ex/image.png',cv.IMREAD_GRAYSCALE)
hist, bins = np.histogram(img.flatten(), 256, [0,256])
plt.imshow(img,cmap='gray',vmin=0,vmax=255)
plt.show()

out_img = exp_op(img,0.45)
hist_new, bins_new = np.histogram(out_img.flatten(), 256, [0,256])
plt.subplot(1,2,1)
plt.title('original img')
plt.stem(hist)
plt.subplot(1,2,2)
plt.title('new img')
plt.stem(hist_new)
plt.show()
plt.imshow(out_img,cmap='gray')
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
