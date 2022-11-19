import cv2
import numpy as np
from matplotlib import pyplot as plt

SEGMENTS_DIGITS = {
(1, 1, 1, 0, 1, 1, 1): 0,
(0, 0, 1, 0, 0, 1, 0): 1,
(1, 0, 1, 1, 1, 0, 1): 2,
(1, 0, 1, 1, 0, 1, 1): 3,
(0, 1, 1, 1, 0, 1, 0): 4,
(1, 1, 0, 1, 0, 1, 1): 5,
(1, 1, 0, 1, 1, 1, 1): 6,
(1, 0, 1, 0, 0, 1, 0): 7,
(1, 1, 1, 1, 1, 1, 1): 8,
(1, 1, 1, 1, 0, 1, 1): 9
}

SEGMENT_SUM = 255*128*20
#img_name = sys.argv.pop(1)
#img = cv2.imread('Digits/{}'.format(img_name))
img_name = input("Which digit?")
img = cv2.imread('Digits/{}.png'.format(img_name),cv2.IMREAD_GRAYSCALE)
active_segments = [0,0,0,0,0,0,0]

segment0 = img[:20,:]
segment1 = img[:128,:20]
segment2 = img[:128,108:]
segment3 = img[118:138,:]
segment4 = img[128:,:20]
segment5 = img[128:,108:]
segment6 = img[236:,:]

if np.sum(segment0) == SEGMENT_SUM :
    active_segments[0]=1
if np.sum(segment1) == SEGMENT_SUM :
    active_segments[1]=1
if np.sum(segment2) == SEGMENT_SUM :
    active_segments[2]=1
if np.sum(segment3) == SEGMENT_SUM :
    active_segments[3]=1
if np.sum(segment4) == SEGMENT_SUM :
    active_segments[4]=1
if np.sum(segment5) == SEGMENT_SUM :
    active_segments[5]=1
if np.sum(segment6) == SEGMENT_SUM :
    active_segments[6]=1


print('Recognized digit: ',SEGMENTS_DIGITS[tuple(active_segments)])
