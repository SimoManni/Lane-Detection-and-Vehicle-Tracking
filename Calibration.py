import cv2
import numpy as np
import glob


# Define the dimensions of the checkerboard
CHECKERBOARD = (6, 9)

# Stop the iteration when specified accuracy, epsilon, is reached or the specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for 3D points
points_3d = []

# Vector for 2D points
points_2d = []

# 3D points real world coordinates
object_p3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
object_p3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Path containing images
folder_path = 'camera_cal/'

# Pattern for the image filenames
image_pattern = folder_path + 'calibration*.jpg'

# Use glob to get a list of filenames matching the pattern
images = glob.glob(image_pattern)

# Counter for the number of successfully processed images
valid_image_count = 0

for filename in images:
    image = cv2.imread(filename)

    if image is None:
        print(f"Unable to read image: {filename}")
        continue

    gray_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray_color,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # If desired number of corners are found, refine the pixel coordinates and display them on the images of the checkerboard
    if ret:
        valid_image_count += 1
        points_3d.append(object_p3d)
        corners2 = cv2.cornerSubPix(gray_color, corners, (11, 11), (-1, -1), criteria)
        points_2d.append(corners2)

# Check if there are enough valid images for calibration
if valid_image_count < 3:
    print("Insufficient valid images for calibration. Calibration requires at least 3 images.")
else:
    # Perform camera calibration
    ret, K, dist_coeff, _, _ = cv2.calibrateCamera(
        points_3d, points_2d, gray_color.shape[::-1], None, None
    )

    # Displaying estimated parameters
    print("Intrinsics matrix:")
    print(np.array2string(K, precision=4, suppress_small=True))

    print("\nDistortion coefficients:")
    print(np.array2string(dist_coeff, precision=4, suppress_small=True))

with open('calibration_params.txt', 'w') as f:
    f.write('Intrinsics matrix:\n')
    f.write(np.array2string(K, precision=4, suppress_small=True))
    f.write('\n\nDistortion coefficients:\n')
    f.write(np.array2string(dist_coeff, precision=4, suppress_small=True))

