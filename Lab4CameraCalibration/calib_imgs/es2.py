import numpy as np
import cv2
from matplotlib import pyplot as plt

pattern_size = (8,5) # number of inner corner, (columns, rows) for OpenCV
square_size = 26 #mm

indices = np.indices(pattern_size, dtype=np.float32)
indices *= square_size
coords_3D = np.transpose(indices, [2, 1, 0])
coords_3D = coords_3D.reshape(-1,2)
pattern_points = np.concatenate([coords_3D, np.zeros([coords_3D.shape[0], 1], dtype=np.float32)], axis=-1)


def processImage(fn):
    print('processing {}'.format(fn))
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    # Check image loaded correctly
    if img is None:
        print("Failed to load", fn)
        return None
    # Finding corners
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    if found:
        # Refining corner position
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        # Visualize detected corners
        #vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        #plt.figure(figsize=(20,10))
        #plt.imshow(vis)
        #plt.show()
    else:
        print('chessboard not found')
        return None
    print('           %s... OK' % fn)
    return (corners.reshape(-1, 2), pattern_points)

corners, pattern_points = processImage('pen.jpg')

# Define the 2D points in homogeneous coordinate
_2d_p1 = [2729, 1500]
_2d_p2 = [2680, 2545]

_2d_p1_homogeneous = [2729., 1500.,1.]
_2d_p2_homogeneous = [2680., 2545.,1.]
#print('2d: ',_2d_point_homogeneous)

# Find Homography
homography = cv2.findHomography(pattern_points[:,:2], corners)[0]

# Get 3d homogeneous coords of the 2 points
_3d_p1_homogeneous = np.matmul(np.linalg.inv(homography), _2d_p1_homogeneous)
_3d_p2_homogeneous = np.matmul(np.linalg.inv(homography), _2d_p2_homogeneous)

# Get euclidean coords
_3d_p1 = _3d_p1_homogeneous / _3d_p1_homogeneous[-1]
_3d_p2 = _3d_p2_homogeneous / _3d_p2_homogeneous[-1]

# Force z=0
_3d_p1[-1]=0
_3d_p2[-1]=0

_3d_p1 = _3d_p1.reshape([1,3])
_3d_p2 = _3d_p2.reshape([1,3])

# Compute euclidean dst
dist = np.linalg.norm(_3d_p1-_3d_p2)

print('Distance = {} mm'.format(dist))

img = cv2.imread('pen.jpg')
cv2.circle(img,_2d_p1,100,(0,0,255),30)
cv2.circle(img,_2d_p2,100,(0,0,255),30)
cv2.line(img,_2d_p1,_2d_p2,(0,255,0),30)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('EX 2')
plt.show()


