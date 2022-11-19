import numpy as np
import cv2
from matplotlib import pyplot as plt

dirname = "calib_imgs/"
img_names = [dirname + str(i) + ".jpg" for i in range(20)]
print(img_names)

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


chessboards = [processImage(fn) for fn in img_names]

obj_points = [] #3D points
img_points = [] #2D points

for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

h, w = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[:2]

# Calibrating Camera
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())
print("Rotation vectors:", rvecs)
print("translation vectors", tvecs)

 
image_index = 0
img = cv2.imread(img_names[image_index])

corners, pattern_points = processImage(img_names[image_index])

# Define the 2D point in homogeneous coordinate in order to multiply with the PPM 

_2d_point = corners[pattern_size[0]*2 + 2] # Corner (2,2) 2D position
_2d_point_homogeneous = np.concatenate([_2d_point, [1]])
print('2d: ',_2d_point_homogeneous)

### USING PPM

#undistort img
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
x, y, w_2, h_2 = roi
img_undistorted = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)


#getting rotation matrix and translation matrix
rotation_matrix = cv2.Rodrigues(rvecs[image_index])[0]
translation_matrix = tvecs[image_index]
#concatenate them to obtain RT matrix
extrinsics_matrix = np.concatenate([rotation_matrix,translation_matrix], axis=1)

ppm = np.matmul(camera_matrix,extrinsics_matrix)

ppm_as_p = np.concatenate([ppm[:,:2],ppm[:,3:]],axis=1)

# Multiply by the inverse of the PPM obtaining the 3d point in homogeneous coordinate
_3d_point_homogeneous = np.matmul(np.linalg.inv(ppm_as_p), _2d_point_homogeneous)

# Divide by the last value to get the 3d coordinate
_3d_point = _3d_point_homogeneous / _3d_point_homogeneous[-1]

# Force z to 0
_3d_point[-1] = 0.

_3d_point = _3d_point.reshape([1,3])

_3d_point = np.array([np.round(_3d_point[0]),np.round(_3d_point[0])],dtype=int)

print('With PPM: ',_3d_point)

#USING HOMOGRAPHY

homography = cv2.findHomography(pattern_points[:,:2], corners)[0]

_3d_point_homogeneous = np.matmul(np.linalg.inv(homography), _2d_point_homogeneous)

_3d_point = _3d_point_homogeneous / _3d_point_homogeneous[-1]
_3d_point[-1]=0
_3d_point = _3d_point.reshape([1,3])
_3d_point = np.array([np.round(_3d_point[0]),np.round(_3d_point[0])],dtype=int)

print('With H: ',_3d_point)




