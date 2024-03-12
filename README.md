# Lane Detection and Vehicle Tracking
This project implements a lane detection and vehicle tracking algorithm using computer vision and machine learning techniques. The lane detection and vehicle tracking part will be analyzed separately. The output of the pipeline can be seen in the video below.

<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/a4d2543b-4f6a-4883-bf74-a47b5f35084f" alt="Lane Video" width="600">
</p>


## Advanced Lane Detection

### Overview
The lane detection pipeline processes video frames to identify lane markings on the road, estimate lane curvature, and determine the vehicle's position relative to the lane center.

<p align="center">
  <img src="https://github.com/SimoManni/Self-Driving-Car-Projects/assets/151052936/470b0d85-5d8a-48ec-bfed-fa88d99a1029" alt="Lane Video" width="600">
</p>


### Pipeline Components

#### 1. Camera Calibration

Camera calibration is a critical step in correcting the distortion caused by the camera lens. In this step, a set of chessboard images with known square sizes is used to compute the camera matrix (K) and distortion coefficients. These parameters are then applied to undistort subsequent frames, ensuring accurate lane detection.

<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/8cfaabdf-ba40-40c8-a920-39272b900b68" alt="Lane Video" width="400">
</p>

#### 2. Thresholding

Thresholding is performed to isolate lane markings by applying color and gradient thresholds to the undistorted frame. By enhancing specific color channels and detecting edges using gradient-based methods like Sobel operators, the pipeline effectively highlights lane features while suppressing noise.

<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/02ae8dc9-7f95-4028-b49d-0df2f3d1e466" alt="Lane Video" width="600">
</p>

#### 3. Bird's Eye View Transformation

The bird's eye view transformation is essential for obtaining a top-down perspective of the road, which simplifies lane detection. This transformation involves defining source and destination points in the original and transformed images, respectively, and applying a perspective transform using OpenCV functions like `cv2.getPerspectiveTransform()`.

<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/79721761-30bb-4732-a5e9-7a29aa51473d" alt="Lane Video" width="600">
</p>


#### 4. Lane Detection

Lane detection is achieved through two main methods:
- **Histogram Based Search**: This method is used initially or when the lanes are lost due to occlusions or sudden changes. It involves computing a histogram of pixel intensities in the bottom half of the bird's eye view image and identifying peak locations as potential lane positions.

<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/5fbb8ec4-5236-4f73-88e0-5a402a3854bc" alt="Lane Video" width="400">
</p>

- **Search Based on Previously Identified Lanes**: Once lanes are detected, a more efficient search is performed around the previously identified lanes using techniques like sliding windows or region of interest (ROI) masking. This approach reduces computational overhead and improves real-time performance.

<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/0e44f32d-433c-4194-9427-ded9d92647d5" alt="Lane Video" width="400">
</p>

#### 5. Curvature Calculation

Curvature calculation estimates the curvature of the lane lines in meters using the detected lane pixels' coordinates. This is done by fitting a second-order polynomial to the lane pixels and applying the curvature formula. Additionally, the offset of the vehicle from the lane center is computed by comparing the midpoint of the detected lanes with the image center.

#### 6. Lane Visualization

The final step involves visualizing the detected lanes on the original undistorted frame. This includes drawing the lane boundaries, filling the lane area, and overlaying curvature and vehicle position information. Visualization aids in validating the pipeline's performance and provides valuable feedback for debugging and parameter tuning.

### Results

The lane detection pipeline demonstrates robust performance in identifying lane markings, estimating lane curvature, and determining the vehicle's position relative to the lane center. Through thorough testing on various road conditions and lighting scenarios, the pipeline consistently produces accurate lane boundaries and curvature measurements.


## Improvements

While the current pipeline performs well under normal conditions, there are opportunities for further enhancement:
- **Robustness to Challenging Conditions**: Enhancing the pipeline's ability to handle challenging conditions such as low light, shadows, adverse weather, and varying road surfaces. This may involve incorporating adaptive thresholding techniques and robust feature extraction algorithms.
- **Real-time Optimization**: Optimizing the pipeline for real-time performance to meet the requirements of embedded systems or autonomous vehicles. This could involve algorithmic optimizations, parallelization, and hardware acceleration using GPUs or specialized processors.
- **Dynamic Region of Interest (ROI)**: Implementing a mechanism to dynamically adjust the ROI based on road geometry and lane configurations. This ensures that the pipeline focuses on relevant regions of the image, improving efficiency and accuracy.
- **Deep Learning Integration**: Exploring the integration of deep learning techniques, such as convolutional neural networks (CNNs), for end-to-end lane detection. Deep learning models can learn complex features directly from raw pixel data, potentially improving performance in challenging scenarios with intricate lane markings or occlusions.



## Vehicle Detection and Tracking Project using SVC

### Overview
This part of the project aims to develop a robust system for detecting and tracking vehicles in video footage captured by an onboard camera. The detection and tracking of vehicles are essential for various applications, including advanced driver assistance systems (ADAS) and autonomous vehicles. The project utilizes a combination of computer vision techniques and machine learning algorithms to accurately identify and track vehicles in real-time.

<div align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/80700b45-fbea-45d0-b6c0-c71046960a0c" alt="Vehicle Detection Tracking GIF" width="600"/>
</div>

### Building the SVC Classifier 
The core of the vehicle detection system lies in the construction of an effective classifier capable of distinguishing between vehicle and non-vehicle regions in the video frames. The classifier is built using a multi-step process:

#### 1. Data Collection and Preprocessing
A diverse dataset containing images of vehicles and non-vehicles is collected from various sources, including online repositories and personal recordings. Each image is preprocessed to ensure consistency in size, color, and orientation.

#### 2. Feature Extraction
Features are extracted from the preprocessed images to represent their unique characteristics. The following features are extracted:
- **Color Histogram**: Histograms of color channels are computed to capture color information.
<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/78378712-016b-41c7-abac-39d8d87e8aa1" alt="image" width="600"/>
</p>

- **Spatial Binning**: Color channels are resized and flattened to create spatially binned feature vectors.
<p align="center">
  <img src=https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/6b35af49-88dd-41cd-8268-14c80b7417d8" alt="image" width="600"/>
</p>

- **Histogram of Oriented Gradients (HOG)**: HOG features are extracted to capture shape and texture information.
<p align="center">
  <img src=https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/bcc54f1e-3bf2-44a3-90a1-fcd496a65c74" alt="image" width="600"/>
</p>

#### 3. Training the Classifier
The extracted features are used to train a Support Vector Classifier (SVC), a popular machine learning algorithm suitable for binary classification tasks. The SVC is trained on labeled data, where vehicle images are labeled as positive examples, and non-vehicle images are labeled as negative examples. The training process involves optimizing the SVC's parameters to maximize classification accuracy and minimize errors.


### Vehicle Detection and Tracking using trained SVC
The Support Vector Classifier (SVC) is a supervised learning algorithm used for classification tasks. It works by finding the hyperplane that best separates different classes in the feature space. The SVC aims to maximize the margin between the classes while minimizing the classification error. It achieves this by transforming the input features into a higher-dimensional space where a hyperplane can be found to separate the classes.

#### Vehicle Detection and Tracking Class

This class implements several key functions for detecting and tracking vehicles within video frames. Let's delve into some of its primary methods:

- #### Thresholding Method

The `thresholding` method performs gradient and color thresholding on an input image to identify potential car points. It first converts the image to HLS color space, extracts the saturation channel, and applies the Sobel operator in the x-direction to detect gradients. Then, it applies thresholds to both the gradient and saturation channel to create binary images. Finally, it combines these binary images and masks out regions in the upper portion of the image, where vehicles are less likely to appear.

<p align="center">
  <img src=https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/bd2a099c-eeb7-4176-b6e9-405017b5d02e" alt="image" width="600"/>
</p>

- #### Sliding Window Method

The slide_window function is crucial for systematically scanning an image for potential vehicle locations. It resizes the image based on specified scales for multi-scale detection and then generates sliding windows. For each window, the function calculates centroids by determining the midpoint coordinates, representing potential vehicle locations. This iterative process covers the image space thoroughly, providing a systematic approach to identify potential vehicle regions.

<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/bff8b593-20b8-4da0-bf21-472a69243fda alt="image" width="600"/>
</p>

- #### Heatmap Generation Method

The `heatmap` method generates a heatmap by processing a set of sliding windows across the image. Each window corresponds to a potential car region, and the method evaluates these regions using a trained classifier to determine the likelihood of containing a vehicle. By accumulating heat from overlapping windows, the method identifies high-confidence regions where vehicles are present.

<p align="center">
  <img src="https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/226963db-9da7-4939-8876-be890bf76551 alt="image" width="600"/>
</p>

- #### Box Drawing Method

The `draw_boxes` method utilizes the heatmap generated by the `heatmap` method to draw bounding boxes around detected vehicles. It employs a label function to identify connected regions with high heat values in the heatmap, indicating potential vehicle locations. These regions are then enclosed with bounding boxes, and the method ensures robust vehicle identification by rejecting outliers and tracking detected vehicles over time. By considering the history of detected vehicles and applying thresholds to the heatmap, the algorithm effectively distinguishes between actual vehicles and noise or irrelevant objects in the scene.

<p align="center">
  <img src=https://github.com/SimoManni/Lane-Detection-and-Vehicle-Tracking/assets/151052936/274873e3-0a2f-4fee-a902-b8c24fbd241f" alt="image" width="600"/>
</p>

In summary, these methods leverage gradient and color thresholding along with heatmap generation and box drawing techniques to robustly identify and track vehicles in video frames, even in challenging scenarios with varying lighting conditions and occlusions.

### Results
The vehicle detection and tracking algorithm demonstrate promising results in identifying and tracking vehicles in the video footage. The combination of color and gradient features, along with the SVC classifier, provides robust performance even under varying lighting and environmental conditions. The system efficiently handles multiple vehicles in the scene and effectively suppresses false positives using heatmap thresholding. However, further optimization and fine-tuning may be required to improve the algorithm's accuracy and real-time performance in complex traffic scenarios.

### Conclusion
In conclusion, the development of an effective vehicle detection and tracking system requires careful consideration of feature selection, classifier design, and algorithm optimization. By leveraging computer vision techniques and machine learning algorithms, it is possible to build a reliable system capable of accurately detecting and tracking vehicles in real-world scenarios.

