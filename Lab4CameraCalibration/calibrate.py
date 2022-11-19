import numpy as np
import cv2
from matplotlib import pyplot as plt

print(cv2.__version__)

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
chessboards = [x for x in chessboards if x is not None]

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