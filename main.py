import numpy as np
import cv2
import os
import pickle
from skimage.feature import hog
from scipy.ndimage import label
import concurrent.futures


## Car class (needed in VDT)
class Car:
    def __init__(self, centroid_x, centroid_y, offset_x, offset_y, H, W):
        self.centroid = (centroid_x, centroid_y)

        self.offset_x_arr = None
        self.offset_y_arr = None

        self.add_offset_x(offset_x)
        self.add_offset_y(offset_y)

        self.detected = True
        self.detected_counter = 0
        self.detected_counter_less = 0

        self.image_shape = (H, W)

    def add_offset_x(self, offset_x, num_avg=10):
        if self.offset_x_arr is None:
            self.offset_x_arr = np.array([offset_x])
        else:
            self.offset_x_arr = np.concatenate((self.offset_x_arr, np.array([offset_x])))

        if len(self.offset_x_arr) > num_avg:
            self.offset_x_arr = self.offset_x_arr[-num_avg:]

        self.offset_x = np.mean(self.offset_x_arr)

    def add_offset_y(self, offset_y, num_avg=10):
        if self.offset_y_arr is None:
            self.offset_y_arr = np.array([offset_y])
        else:
            self.offset_y_arr = np.concatenate((self.offset_y_arr, np.array([offset_y])))

        if len(self.offset_y_arr) > num_avg:
            self.offset_y_arr = self.offset_y_arr[-num_avg:]

        self.offset_y = np.mean(self.offset_y_arr)

    def update_object(self, centroid_x, centroid_y, offset_x, offset_y):
        self.centroid = (centroid_x, centroid_y)
        self.add_offset_x(offset_x)
        self.add_offset_y(offset_y)
        self.detected_counter += 1
        self.detected = True

    def get_box(self):
        # Define a bounding box based on offsets and centroid
        top_left_x = max(0, int(self.centroid[0] - self.offset_x) - 10)
        top_left_y = max(0, int(self.centroid[1] - self.offset_y) - 5)
        bottom_right_x = min(self.image_shape[1], int(self.centroid[0] + self.offset_x) + 10)
        bottom_right_y = min(self.image_shape[0], int(self.centroid[1] + self.offset_y) + 5)

        top_left = (top_left_x, top_left_y)
        bottom_right = (bottom_right_x, bottom_right_y)

        return top_left, bottom_right



## Vehicle Detection and Tracking Pipeline Class
class VehicleDetectionTracking:
    def __init__(self):
        # Import SVC model
        self.import_SVC()

        self.cars = []

        self.first = True
        self.H = None
        self.W = None

        self.heatmap_prev = []

    def import_SVC(self):

        print('Loading Classifier parameters...')

        dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
        self.svc = dist_pickle["svc"]
        self.X_scaler = dist_pickle["scaler"]
        self.orient = dist_pickle["orient"]
        self.pix_per_cell = dist_pickle["pix_per_cell"]
        self.cell_per_block = dist_pickle["cell_per_block"]
        self.spatial_size = dist_pickle["spatial_size"]
        self.hist_bins = dist_pickle["hist_bins"]

        print('Loading is done')

    # Extract features
    def convert_color(self, img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

    def color_hist(self, img, nbins=32):  # bins_range=(0, 256)

        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
        channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
        channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

        return hist_features

    def bin_spatial(self, img, size=(32, 32)):
        color1 = cv2.resize(img[:, :, 0], size).ravel()
        color2 = cv2.resize(img[:, :, 1], size).ravel()
        color3 = cv2.resize(img[:, :, 2], size).ravel()

        return np.hstack((color1, color2, color3))

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(img, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                                      transform_sqrt=False,
                                      visualize=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img, orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           transform_sqrt=False,
                           visualize=vis, feature_vector=feature_vec)
            return features

    def extract_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                             hist_bins=32, orient=9,
                             pix_per_cell=8, cell_per_block=2, hog_channel=0,
                             spatial_feat=True, hist_feat=True, hog_feat=True):
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        if spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            spatial_features = np.ravel(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            hist_features = np.ravel(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = self.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        return np.concatenate((spatial_features, hist_features, hog_features)).reshape(1, -1)

    def thresholding(self, img):

        # Convert the image to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # Extract the S (saturation) channel
        s_channel = hls[:, :, 2]

        # Apply Sobel operator in the x direction to the S channel
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)

        # Take the absolute value of the derivative
        abs_sobelx = np.absolute(sobelx)

        # Scale to 8-bit (0 - 255)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = 150
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        combined_binary[:400, :] = 0

        return combined_binary

    def slide_window(selg, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     xy_window=(50, 50), xy_overlap=(0.25, 0.25), scales=[1.0, 1.5, 2.0], min_pixels=1000):
        windows = []
        for scale in scales:
            # Rescale the image
            scaled_image = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))

            # Define the span of the region to be searched
            x_start = x_start_stop[0] if x_start_stop[0] is not None else 0
            x_stop = x_start_stop[1] if x_start_stop[1] is not None else scaled_image.shape[1]
            y_start = y_start_stop[0] if y_start_stop[0] is not None else 0
            y_stop = y_start_stop[1] if y_start_stop[1] is not None else scaled_image.shape[0]

            # Compute the number of pixels to step in x and y
            x_step = int(xy_window[0] * xy_overlap[0])
            y_step = int(xy_window[1] * xy_overlap[1])

            # Calculate window positions
            start_xs = np.arange(x_start, x_stop - xy_window[0] + 1, x_step)
            start_ys = np.arange(y_start, y_stop - xy_window[1] + 1, y_step)

            # Create meshgrid of window positions
            mesh_xs, mesh_ys = np.meshgrid(start_xs, start_ys)

            # Flatten meshgrid coordinates
            flat_xs = mesh_xs.flatten()
            flat_ys = mesh_ys.flatten()

            # Iterate over window positions
            for startx, starty in zip(flat_xs, flat_ys):
                endx = startx + xy_window[0]
                endy = starty + xy_window[1]

                # Extract window from the image
                window = scaled_image[starty:endy, startx:endx]

                # Check if the number of non-zero pixels exceeds the threshold
                if np.sum(window != 0) > min_pixels * scale:
                    # Calculate centroid of the window and convert to original image coordinates
                    centroid = ((startx + endx) / 2 * scale, (starty + endy) / 2 * scale)
                    windows.append(centroid)

        return windows

    def heatmap(self, img, windows, offsets=(32, 64), color_space='YCrCb', hog_channel='ALL'):

        # Create heat map
        heat_map = np.zeros(img.shape[:2])

        # Concatenate all windows and offsets
        windows_offsets = [(x, y, offset) for x, y in windows for offset in offsets]

        # Resize windows to a fixed size beforehand and extract features
        resized_windows = [cv2.resize(img[max(int(y - offset), 0):min(int(y + offset), img.shape[0]),
                                      max(int(x - offset), 0):min(int(x + offset), img.shape[1])], (64, 64))
                           for x, y, offset in windows_offsets]

        # Extract features for all windows
        features = [self.extract_img_features(window, color_space=color_space, hog_channel=hog_channel) for window in resized_windows]

        # Transform features
        test_features = self.X_scaler.transform(np.concatenate(features))

        # Predict all windows at once
        predictions = self.svc.predict(test_features)

        for (x, y, offset), prediction in zip(windows_offsets, predictions):
            if prediction == 1:
                heat_map[int(y - offset):int(y + offset), int(x - offset):int(x + offset)] += 1

        return heat_map

    def draw_boxes(self, img, heat_map, threshold=2):
        draw_img = np.copy(img)
        heat_map[heat_map <= threshold] = 0
        labels = label(heat_map)

        cars = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()

            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            offset_x = int((np.max(nonzerox) - np.min(nonzerox)) / 2)
            offset_y = int((np.max(nonzeroy) - np.min(nonzeroy)) / 2)

            centroid_x = int((np.min(nonzerox) + np.max(nonzerox)) / 2)
            centroid_y = int((np.min(nonzeroy) + np.max(nonzeroy)) / 2)

            if self.cars != []:
                matched_car, dist = self.match_car((centroid_x, centroid_y))
                if dist < 50:
                    cars.append((centroid_x, centroid_y, offset_x, offset_y, matched_car))
                else:
                    cars.append((centroid_x, centroid_y, offset_x, offset_y, None))
            else:
                cars.append((centroid_x, centroid_y, offset_x, offset_y, None))

        if self.cars != []:
            for car in self.cars:
                car.detected = False

        for car in cars:
            centroid_x, centroid_y, offset_x, offset_y, matched_car = car
            if matched_car is None:
                car = Car(centroid_x, centroid_y, offset_x, offset_y, self.H, self.W)
                car.detected_counter += 1
                self.cars.append(car)

                # Don't represent cars detected the first time to reject outliers

            else:
                self.cars[matched_car].update_object(centroid_x, centroid_y, offset_x, offset_y)

        for car in self.cars:
            if not car.detected:
                if car.detected_counter_less == 0:
                    car.detected_counter_less = car.detected_counter // 4

                car.detected_counter -= car.detected_counter_less


        for car_number in range(len(self.cars)):
            if self.cars[car_number].detected_counter >= 4:
                self.cars[car_number].detected = True

                # Get bounding box and draw it
                top_left, bottom_right = self.cars[car_number].get_box()
                cv2.rectangle(draw_img, top_left, bottom_right, (0, 0, 255), 6)

        self.cars = [car for car in self.cars if car.detected]

        return draw_img

    def match_car(self, centroid):
        dists = []
        x_car = centroid[0]
        y_car = centroid[1]
        for car_number in range(len(self.cars)):
            x_query = self.cars[car_number].centroid[0]
            y_query = self.cars[car_number].centroid[1]
            dist = np.sqrt((x_car - x_query)**2 + (y_car - y_query)**2)
            dists.append(dist)

        return np.argmin(dists), np.min(dists)

    def process_new_frame(self, frame, alpha=0.5, threshold=0.5):
        if self.first:
            self.H, self.W = frame.shape[:2]
            self.heatmap_prev = np.zeros((self.H, self.W))
            self.first = False

        combined_binary = self.thresholding(frame)
        windows = self.slide_window(combined_binary, y_start_stop=[400, 720])
        heatmap_curr = self.heatmap(frame, windows)

        heatmap = (1 - alpha) * self.heatmap_prev + alpha * heatmap_curr
        heatmap_thresh = ((heatmap > threshold) * 255).astype(np.uint8)
        # draw_img = self.draw_boxes(frame, heatmap_thresh)
        self.heatmap_prev = heatmap_curr

        return heatmap_thresh

## Line class
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in meters
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # Counter for number
        self.linenum = 0
    def add_fit(self, fit):
        max_linenum = 15
        if self.linenum > max_linenum:
            if fit is not None:
                self.detected = True
                self.current_fit.append(fit)
                if len(self.current_fit) > max_linenum:
                    self.current_fit = self.current_fit[len(self.current_fit) - max_linenum:]
                self.best_fit = np.average(self.current_fit, axis=0)

            else:
                self.detected = False
                if len(self.current_fit) > 0:
                    self.best_fit = np.average(self.current_fit, axis=0)
        else:
            if fit is not None:
                self.detected = True
                self.current_fit.append(fit)
                if len(self.current_fit) > 1:
                    self.current_fit = self.current_fit[len(self.current_fit) - 1:]
                self.best_fit = np.average(self.current_fit, axis=0)

            else:
                self.detected = False
                if len(self.current_fit) > 0:
                    self.best_fit = np.average(self.current_fit, axis=0)


## Lane Detection Pipline class
class LaneDetection:
    def __init__(self):
        # Camera calibration parameters
        self.K, self.dist_coeff = self.import_calibration_params('calibration_params.txt')
        self.undistorted_frame = None
        # Current left and right fit
        self.left_line = Line()
        self.right_line = Line()


    def import_calibration_params(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Parse intrinsics matrix K
        K_values = []
        for line in lines[1:4]:
            line_values = line.strip().replace('[', '').replace(']', '').split()
            K_values.extend(map(float, line_values))
        K = np.array(K_values).reshape(3, 3)

        # Parse distortion coefficients dist_coeff
        dist_coeff_str = lines[-1].strip().strip('[]')  # Remove square brackets and whitespace from the string
        dist_coeff_values = dist_coeff_str.split()
        dist_coeff = np.array(dist_coeff_values, dtype=float)  # Convert the list of strings to a numpy array

        print("Loaded Intrinsics matrix:")
        for row in K:
            print("[", end=" ")
            for value in row:
                print("{:.2f}".format(value), end=" ")
            print("]")

        # Loaded Distortion coefficients
        print("\nLoaded Distortion coefficients:")
        for value in dist_coeff:
            print("{:.3f}".format(value), end=" ")

        return K, dist_coeff

    def process_new_frame(self, frame):

        # Undistort image
        undistorted_image = np.copy(frame)
        undistorted_image = cv2.undistort(undistorted_image, self.K, self.dist_coeff)
        self.undistorted_frame = undistorted_image

        # Apply thresholding
        thresh_image = self.color_and_gradient_tresh(undistorted_image)

        # Warp image patch
        _, Minv, binary_warped = self.get_top_view(thresh_image)

        if not self.left_line.detected or not self.right_line.detected:
            thresh_image = self.improved_threshold(undistorted_image)
            _, Minv, binary_warped = self.get_top_view(thresh_image)
            # Find the lanes using histogram method
            leftx, lefty, rightx, righty, _ = self.histogram(binary_warped)
            left_fit, right_fit = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        else:
            # Search in the area of previous found lanes
            left_fit, right_fit = self.search_around_poly(binary_warped, self.left_line.best_fit, self.right_line.best_fit)

        left_curverad, right_curverad, dist_from_center = self.measure_curvature_real(binary_warped, left_fit, right_fit)

        # Checking curvature ratios since lanes should be parallel
        if abs(left_curverad) / abs(right_curverad) < 0.3 or abs(left_curverad) / abs(right_curverad) > 3:
            left_fit = None
            right_fit = None

        self.left_line.add_fit(left_fit)
        self.left_line.radius_of_curvature = left_curverad
        self.left_line.linenum += 1

        self.right_line.add_fit(right_fit)
        self.right_line.radius_of_curvature = right_curverad
        self.right_line.linenum += 1

        if self.left_line.best_fit is not None and self.right_line.best_fit is not None:
            image_with_lanes = self.draw_lanes(undistorted_image, Minv)
        else:
            image_with_lanes = undistorted_image

        full_image = self.full_image(image_with_lanes, dist_from_center, binary_warped)
        return full_image

    def color_and_gradient_tresh(self, image, s_thresh=(170, 255), sx_thresh=(20, 100), b_thresh=(150, 255)):

        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sx_binary = np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1


        # Combine binary thresholds
        combined_binary = cv2.bitwise_or(sx_binary, s_binary)

        return combined_binary

    def improved_threshold(self, image):
        # Convert image to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Convert image to HLS color space
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Yellow detection
        adapt_yellow_S = cv2.adaptiveThreshold(hls[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161,-5)
        adapt_yellow_B = cv2.adaptiveThreshold(lab[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161,-5)
        adapt_yellow = adapt_yellow_S & adapt_yellow_B

        # White detection
        adapt_white_R = cv2.adaptiveThreshold(image[:, :, 0], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161,-27)
        adapt_white_L = cv2.adaptiveThreshold(hsv[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161,-27)
        adapt_white = adapt_white_R & adapt_white_L

        # Combine yellow and white detections
        combined_binary = adapt_yellow | adapt_white

        return combined_binary
    def get_top_view(self, image):
        """
            This function takes an image and returns the parameters of the perspective transformation of the road to get a
            top-down view and the corresponding warped image. For more details about implementation see Jupyter notebook.
        """

        src_points = np.float32([
            [-79, 685],  # bottom-left corner
            [550, 450],  # top-left corner
            [729, 450],  # top-right corner
            [1406, 685]  # bottom-right corner
        ])

        # Destination points are to be parallel, taking into account the image size
        dst_points = np.float32([
            [0, 600],  # bottom-left corner
            [0, 0],  # top-left corner
            [500, 0],  # top-right corner
            [500, 600]  # bottom-right corner
        ])

        # Calculate the transformation matrix and it's inverse transformation
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)
        warped = cv2.warpPerspective(image, M, (600, 500))

        return M, M_inv, warped

    def histogram(self, binary_warped):

        # Take a histogram of the bottom half of the image to detect initial position of lanes
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Output image to visualize results later
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 10
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        # Note that x and y are swapped because y corresponds to the row index and x to the column index
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            # Four boundaries of left and right windows
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        # Fit a second order polynomial
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def search_around_poly(self, binary_warped, left_fit, right_fit):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 50

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Area of search based on fit of previous frame
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                       left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                        right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fit, right_fit = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        return left_fit, right_fit

    def measure_curvature_real(self, binary_warped, left_fit, right_fit):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 60 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        ploty = np.arange(-100, 580)
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Fit new polynomials to x, y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve
        # Left lane
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        # Right lane
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Calculating the position of the car assuming camera is mounted at the center of the car
        car_x = binary_warped.shape[1] / 2
        l = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
        r = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
        lane_center = (l + r) / 2
        # Changing to x in world space
        dist_from_center = (car_x - lane_center) * xm_per_pix

        return left_curverad, right_curverad, dist_from_center

    def draw_lanes(self, image, Minv):

        # Get left and right fit
        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        # Create an image to draw the lines on
        height, width, _ = image.shape
        color_warp = np.zeros((height, width, 3), dtype=np.uint8)

        # Compute points from fit
        ploty = np.arange(-100, 580)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.column_stack((left_fitx, ploty, np.ones(len(left_fitx))))
        pts_right = np.column_stack((right_fitx, ploty, np.ones(len(right_fitx))))

        # Multiply points by Minv to transform them back in the image
        pts_left_warped = np.dot(Minv, pts_left.T).T
        pts_left_warped = np.divide(pts_left_warped[:, :2], pts_left_warped[:, 2].reshape(-1, 1))

        pts_right_warped = np.dot(Minv, pts_right.T).T
        pts_right_warped = np.divide(pts_right_warped[:, :2], pts_right_warped[:, 2].reshape(-1, 1))

        # Reshape and concatenate points for fillPoly
        pts_left_warped = pts_left_warped.reshape((-1, 1, 2))
        pts_right_warped = pts_right_warped.reshape((-1, 1, 2))
        pts = np.concatenate((pts_left_warped, pts_right_warped[::-1]), axis=0)

        # Highlight region between lanes on image
        cv2.fillPoly(color_warp, [pts.astype(np.int32)], (0, 255, 0))
        result = cv2.addWeighted(image, 1, color_warp, 0.3, 0)

        # Highlight lanes
        lines_warp = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.polylines(lines_warp, [pts_left_warped.astype(np.int32)], isClosed=False, color=(0, 0, 255), thickness=5)
        cv2.polylines(lines_warp, [pts_right_warped.astype(np.int32)], isClosed=False, color=(255, 0, 0), thickness=5)
        result = cv2.addWeighted(result, 1, lines_warp, 0.5, 0)

        return result

    ## Plotting

    def full_image(self, image_with_lanes, dist_from_center, binary_warped):
        H, W = image_with_lanes.shape[:2]
        full_image = np.copy(image_with_lanes)

        # Black rectangle in top left corner
        x_top_left = 30
        y_top_left = 30
        height_rect = 300
        width_rect = 400
        radius = 20

        # Create a black rectangle as a mask
        black_rect = np.zeros((height_rect, width_rect, 3), dtype=np.uint8)

        # Apply the darkening effect on the full_image
        alpha = 0.5
        full_image[y_top_left:y_top_left + height_rect, x_top_left:x_top_left + width_rect] = cv2.addWeighted(
            full_image[y_top_left:y_top_left + height_rect, x_top_left:x_top_left + width_rect],
            1 - alpha,
            black_rect,
            alpha,
            0
        )


        # Text
        curv_rad = (self.left_line.radius_of_curvature + self.right_line.best_fit[0]) / 2
        rad_thresh = 3000

        if curv_rad > rad_thresh:
            # Straight road
            straight_sign = cv2.imread('outputs/straight.png', cv2.IMREAD_UNCHANGED)
            straight_sign = cv2.resize(straight_sign, (0, 0), fx=0.4, fy=0.4)

            # Coordinates where to overlay image
            roi_top_left = (170, 100)
            h_straight, w_straight = straight_sign.shape[:2]
            # Coordinates of the bottom right corner of the ROI
            roi_bottom_right = (roi_top_left[0] + w_straight, roi_top_left[1] + h_straight)

            # Ensure the ROI stays within the boundaries of the full_image
            roi_bottom_right = (min(roi_bottom_right[0], full_image.shape[1]), min(roi_bottom_right[1], full_image.shape[0]))

            # Define the region of interest in the full_image
            roi = full_image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

            # Overlay the straight_sign onto the region of interest
            overlay_rgb = straight_sign[:, :, :3]
            alpha_mask = straight_sign[:, :, 3]

            # Create a mask for the overlay image using the alpha channel
            alpha_mask = cv2.merge([alpha_mask] * 3)

            # Apply the mask to the overlay image
            masked_overlay = cv2.bitwise_and(overlay_rgb, alpha_mask)

            # Invert the alpha mask
            inverted_alpha_mask = cv2.bitwise_not(alpha_mask)

            # Apply the inverted mask to the ROI
            masked_roi = cv2.bitwise_and(roi, inverted_alpha_mask)

            # Combine the masked overlay and masked ROI
            result_roi = cv2.add(masked_overlay, masked_roi)

            # Replace the region of interest in the full_image with the combined result
            full_image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = result_roi

            # Text
            text1 = 'Straight'

        else:
            if (self.left_line.best_fit[0] < 0) and (self.right_line.best_fit[0] < 0):
                # Left curve
                left_sign = cv2.imread('outputs/left.png', cv2.IMREAD_UNCHANGED)
                left_sign = cv2.resize(left_sign, (0, 0), fx=0.22, fy=0.22)

                # Coordinates where to overlay image
                roi_top_left = (170, 100)
                h_straight, w_straight = left_sign.shape[:2]
                # Coordinates of the bottom right corner of the ROI
                roi_bottom_right = (roi_top_left[0] + w_straight, roi_top_left[1] + h_straight)

                # Ensure the ROI stays within the boundaries of the full_image
                roi_bottom_right = (
                min(roi_bottom_right[0], full_image.shape[1]), min(roi_bottom_right[1], full_image.shape[0]))

                # Define the region of interest in the full_image
                roi = full_image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

                # Overlay the straight_sign onto the region of interest
                overlay_rgb = left_sign[:, :, :3]
                alpha_mask = left_sign[:, :, 3]

                # Create a mask for the overlay image using the alpha channel
                alpha_mask = cv2.merge([alpha_mask] * 3)

                # Apply the mask to the overlay image
                masked_overlay = cv2.bitwise_and(overlay_rgb, alpha_mask)

                # Invert the alpha mask
                inverted_alpha_mask = cv2.bitwise_not(alpha_mask)

                # Apply the inverted mask to the ROI
                masked_roi = cv2.bitwise_and(roi, inverted_alpha_mask)

                # Combine the masked overlay and masked ROI
                result_roi = cv2.add(masked_overlay, masked_roi)

                # Replace the region of interest in the full_image with the combined result
                full_image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = result_roi

                # Text
                text1 = 'Left turn'
                text2 = f'Curvature: {"{:.2f}".format(curv_rad)} m'

            else:
                # Right curve
                right_sign = cv2.imread('outputs/right.png', cv2.IMREAD_UNCHANGED)
                right_sign = cv2.resize(right_sign, (0, 0), fx=0.22, fy=0.22)

                # Coordinates where to overlay image
                roi_top_left = (170, 100)
                h_straight, w_straight = right_sign.shape[:2]
                # Coordinates of the bottom right corner of the ROI
                roi_bottom_right = (roi_top_left[0] + w_straight, roi_top_left[1] + h_straight)

                # Ensure the ROI stays within the boundaries of the full_image
                roi_bottom_right = (
                    min(roi_bottom_right[0], full_image.shape[1]), min(roi_bottom_right[1], full_image.shape[0]))

                # Define the region of interest in the full_image
                roi = full_image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

                # Overlay the straight_sign onto the region of interest
                overlay_rgb = right_sign[:, :, :3]
                alpha_mask = right_sign[:, :, 3]

                # Create a mask for the overlay image using the alpha channel
                alpha_mask = cv2.merge([alpha_mask] * 3)

                # Apply the mask to the overlay image
                masked_overlay = cv2.bitwise_and(overlay_rgb, alpha_mask)

                # Invert the alpha mask
                inverted_alpha_mask = cv2.bitwise_not(alpha_mask)

                # Apply the inverted mask to the ROI
                masked_roi = cv2.bitwise_and(roi, inverted_alpha_mask)

                # Combine the masked overlay and masked ROI
                result_roi = cv2.add(masked_overlay, masked_roi)

                # Replace the region of interest in the full_image with the combined result
                full_image[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]] = result_roi

                # Text
                text1 = 'Right turn'
                text2 = f'Curvature: {"{:.2f}".format(curv_rad)} m'

        # Display text
        text3 = f'Distance from centre: {"{:.2f}".format(dist_from_center)} m'

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size1 = 1
        font_size2 = 0.7
        font_color = (255, 255, 255)  # White color in BGR
        thickness = 1
        position1 = (155, 75)  # (x, y)

        # Write the text on the image
        cv2.putText(full_image, text1, position1, font, font_size1, font_color, thickness)

        if curv_rad < rad_thresh:
            # Turn
            position2 = (50, position1[1] + 180)
            cv2.putText(full_image, text2, position2, font, font_size2, font_color, thickness)
            position3 = (position2[0], position2[1] + 40)
            cv2.putText(full_image, text3, position3, font, font_size2, font_color, thickness)
        else:
            position3 = (50, position1[1]  + 190)
            cv2.putText(full_image, text3, position3, font, font_size2, font_color, thickness)


        # Overimpose binary_warped image
        binary_warped *= 255
        binary_warped = cv2.resize(binary_warped, (0, 0), fx=0.5, fy=0.5)
        binary_warped_rgb = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2RGB)

        # Define the coordinates to place the smaller image in the top right corner
        top_right_x = full_image.shape[1] - binary_warped.shape[1] - 30
        top_right_y = 30

        # Overlay the smaller image onto the full image with slight transparency
        alpha = 0.5
        full_image[top_right_y:top_right_y + binary_warped_rgb.shape[0], top_right_x:top_right_x + binary_warped_rgb.shape[1]] = (
                binary_warped_rgb * alpha + full_image[top_right_y:top_right_y + binary_warped_rgb.shape[0],
                top_right_x:top_right_x + binary_warped_rgb.shape[1]] * alpha
                )

        return full_image


## Main

vdt = VehicleDetectionTracking()
ld = LaneDetection()

# Get video path
current_dir = os.getcwd()
video_path = os.path.join(current_dir, 'test_videos/project_video.mp4')


cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
start_time = 5      # Start time in seconds
end_time = 100      # End time in seconds
total_frames = int(frame_rate * (end_time - start_time))  # Total frames to display for the given duration

# Seek to the start time
cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
frame_count = 0

# Create a VideoWriter object to write the video in memory
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
memory_video = cv2.VideoWriter('outputs/LaneDetectionVehicleTrackingVideo.mp4', fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))


with concurrent.futures.ThreadPoolExecutor() as executor:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= total_frames or cv2.waitKey(1) == ord('q'):
            break

        # Submitting tasks to the ThreadPoolExecutor
        lane_future = executor.submit(ld.process_new_frame, frame)
        heatmap_thresh_future = executor.submit(vdt.process_new_frame, frame)

        # Retrieve results
        lane_image = lane_future.result()
        heatmap_thresh = heatmap_thresh_future.result()

        # Display or process the combined image as needed
        draw_image = vdt.draw_boxes(lane_image, heatmap_thresh)
        cv2.imshow('Detected cars', draw_image)
        memory_video.write(draw_image)

        frame_count += 1

cap.release()
memory_video.release()
cv2.destroyAllWindows()