#1: Compute the pixel-wise difference between two pictures : Image1 and Image2. 
#   Compute an output image where each pixel of coordinates (x,y) contains the absolute difference of 
#   the corresponding pixels on Image1 and Image2: Out(x,y) = abs(Image1(x,y) – Image2(x,y)).
#2: Find all pixels with a mean difference (across R,G,B channels) higher than 0 and create a copy 
#   of Image1 obscuring (with value 0) those pixels. Display that image.
#3: Save the new image on disk and check the result.
#
# Test the exercise on Image 1:*"differences1.png"* and Image 2: *"differences2.png"*.

# Write here your solution
# Import libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Read images
img1 = cv2.imread('Differences/differences1.png')
img_rgb1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = cv2.imread('Differences/differences2.png')
img_rgb2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
# Perform Difference
img_diff = abs(img1 - img2)

# Display Image
plt.imshow(img_diff)
plt.show()

#Mean diff > 0
diff_mean = np.mean(img_diff,axis=-1)
out_img = np.copy(img1)
out_img[diff_mean>0]=0
plt.imshow(out_img)
plt.show()

#save on disk
copied_image_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("out_img_diff_mean.jpg", copied_image_bgr)