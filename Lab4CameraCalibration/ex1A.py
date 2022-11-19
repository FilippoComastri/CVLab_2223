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
point_3d = np.array([[52., 52., 0.]], dtype=np.float32)

#Using the function
point_2d,_ = cv2.projectPoints(point_3d,rvecs[image_index],tvecs[image_index],camera_matrix,dist_coefs) 
point_2d[0][0] = np.round(point_2d[0][0])
img1=np.copy(img)
cv2.circle(img1,tuple(np.array(point_2d[0][0],dtype=int)),10,(0,0,255),1)

plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.show()

#Doing it 'manually'
# First undistort image to make the PPM working better removing the lens distorsion.
# If the lens distorsion is small you shoud not see big differences
# We have to do it since the homography is a linear transformation and cannot remap nonlinear deformations

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
x, y, w_2, h_2 = roi
img_undistorted = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)


#getting rotation matrix and translation matrix
rotation_matrix = cv2.Rodrigues(rvecs[image_index])[0]
translation_matrix = tvecs[image_index]
#concatenate them to obtain RT matrix
extrinsics_matrix = np.concatenate([rotation_matrix,translation_matrix], axis=1)

ppm = np.matmul(camera_matrix,extrinsics_matrix)

# Define the 3D point in homogeneous coordinate in order to multiply with the PPM 
_3d_point_homogeneous = np.array([[52.], [52.], [0.], [1.]])

# Multiply by the PPM obtaining the pixel in homogeneous coordinate
_2d_point_homogeneous = np.matmul(ppm, _3d_point_homogeneous)

# Divide by the third value to get the pixel coordinate
_2d_point = _2d_point_homogeneous / _2d_point_homogeneous[-1, 0]

x_2d=int(round(_2d_point[0,0]))
y_2d=int(round(_2d_point[1,0]))

img2=np.copy(img)
cv2.circle(img2,(x_2d,y_2d),10,(0,0,255),1)

plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()





