import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('es4/pen.jpg')


h = img.shape[0]
w = img.shape[1]

src_pts=np.array([
    [125,155],
    [2533,488],
    [2414,3586],
    [93,3807]],
    dtype=np.float32
)

dest_pts=np.array([
    [0,0],
    [w,0],
    [w,h],
    [0,h]],
    dtype=np.float32
)

p = cv2.getPerspectiveTransform(src_pts,dest_pts)

warped = cv2.warpPerspective(img,p,(w,h))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB))
plt.show()

