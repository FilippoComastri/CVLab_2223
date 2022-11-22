import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('es3/chessboard.jpg')
cat = cv2.imread('es3/stregatto2.jpg')

plt.imshow(cv2.cvtColor(cat,cv2.COLOR_BGR2RGB))
plt.show()

h = img.shape[0]
w = img.shape[1]

h_2 = cat.shape[0]
w_2 = cat.shape[1]

# 4 points where to project the img
_2d_pts=np.array([
    [481,401],
    [2525,504],
    [2573,3467],
    [529,3704]],
    dtype=np.float32)

p1_3d = [0,0]
p2_3d = [0,h_2-1]
p3_3d = [w_2-1,h_2-1]
p4_3d = [w_2-1,0]
_3d_pts = np.array([p1_3d,p2_3d,p3_3d,p4_3d],dtype=np.float32)

# Get perspective transformation
pt = cv2.getPerspectiveTransform(_3d_pts,_2d_pts)

# Warp img
warped = cv2.warpPerspective(cat, pt, (w,h))
plt.imshow(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB))
plt.show()

# White Mask to understand which are black pixels introduced by warping
white_mask = np.ones([h_2,w_2,3],dtype=np.uint8)*255
warped_white_mask = cv2.warpPerspective(white_mask,pt,(w,h))

plt.imshow(cv2.cvtColor(warped_white_mask,cv2.COLOR_BGR2RGB))
plt.show()

# La cornice è rappresentata dai punti neri ([0,0,0])
cornice = np.equal(warped_white_mask,np.array([0,0,0],dtype=np.float32))

# Sostituisco i punti corrispondenti alla cornice con i valori originali
warped[cornice]=img[cornice]

plt.imshow(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB))
plt.show()

