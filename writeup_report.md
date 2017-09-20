
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/Feature_example.png
[image4]: ./output_images/HSV_result.png
[image5]: ./output_images/YCrCb_result.png
[image6]: ./output_images/sliding_window.png
[image7]: ./output_images/bboxes_and_heat.png
[image8]: ./output_images/vehicle_detection_pipeline.png
[video1]: ./project_video.mp4

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

All of the code for the project is contained in the Jupyter notebook "P5.ipynb".

### Feature Extraction

#### 1. Explain how (and identify where in your code) you extracted features from the training images.

The code for this step is contained in the #2 to the #8 code cell of the IPython notebook.

I started by loading all the `car` and `notcar` images.  There are totally **8792** car images and **8968** notcar images in the training data set.
Here is an example of twenty random images of the `car` and `notcar` classes:

![alt text][image1]

Then I explored different features by combining the following features.

* Spatial Features

Spatial features are extracted by the function `bin_spatial()`. It is the simplest feature extraction function but also the most useless function.
`spatial_size` is the only one parameter to tune for extracting spatial features. I compared the training results with 32x32 and 16x16 pixel image size, and found that it has little influence on the test accuracy result. With 16x16 pixel image size less time was taken.

* Color Histogram Features

Color histogram features are extracted by the function `color_hist()`. The color histogram features identify the amount of certain color values. Here I chose three color spaces of HSV, YUV, YCrCb for the color histogram features. The only parameter to tune for extracting color histogram features is the number of bins or "buckets" you want to divide the color space into. In this project 32 bins was set.

* HOG Features

HOG features are extracted by the function `get_hog_features()`. These HOG features are the most important features compared to the other two. With these features, the test accuracy increased remarkable.

The scikit-image `hog()` function was used in a single color channel as input, as well as various parameters. These parameters include `orientations`, `pixels_per_cell` and `cells_per_block`. Because the training images size is 64x64, I found that 8 pixels_per_cell and 2 cells_per_block performed well and fit perfectly with the 64x64 pixel training images, so I didn't toy with those settings much. While 9 orientation directions is commonly used, so this parameter was not tuned.

Compared to extracting single color channel, extracting all three channels can improve the test accuracy by about 3-4%.

Here is an example using the HSV color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

In the function `extract_features()`, the combination of the Spatial Features, Color Histogram Features and HOG Features can be tuned by parameters `spatial_feat`, `hist_feat` and `hog_feat`.

#### 2. Explain how you settled on your final choice of feature parameters.

Following is the training approach when I tried various combinations of parameters.

SVM training approach:

| No. | color space | orient | pix per cell | cell per clock | hog channel | spatial size | hist bins | spatial feat | hist feat | hog feat | extract time | test accuracy | traing time |
|:---:|:-----------:|:------:|:------------:|:--------------:|:-----------:|:------------:|:---------:|:------------:|:---------:|:--------:|:-------------:|:-----------:|:-------------|
| #1 | HSV   | 9 | 8 | 2 | 2  | (32, 32) | 32 | False | False | True | 37.41s  | 94.82% | 11.31s|
| #2 | HSV   | 9 | 8 | 2 | ALL| (32, 32) | 32 | False | False | True | 95.08s  | 97.97% | 21.67s|
| #3 | HSV   | 9 | 8 | 2 | 2  | (32, 32) | 32 | True  | True  | True | 108.73s | 97.92% | 20.21s|
| #4 | HSV   | 9 | 8 | 2 | ALL| (32, 32) | 32 | True  | True  | True | 111.76s | 98.99% | 34.33s|
| #5 | HSV   | 9 | 8 | 2 | ALL| (16, 16) | 32 | True  | True  | True | 87.686s | 98.82% | 21.85s|
| #6 | YUV   | 9 | 8 | 2 | 0  | (32, 32) | 32 | False | False | True | 46.96s  | 94.93% | 10.95s|
| #7 | YUV   | 9 | 8 | 2 | ALL| (32, 32) | 32 | False | False | True | 98.08s  | 97.96% | 21.69s|
| #8 | YUV   | 9 | 8 | 2 | ALL| (32, 32) | 32 | True  | True  | True | 141.25s | 98.86% | 33.62s|
| #9 | YCrCb | 9 | 8 | 2 | 0  | (32, 32) | 32 | False | False | True | 33.77s  | 94.82% | 9.94s |
| #10| YCrCb | 9 | 8 | 2 | ALL| (32, 32) | 32 | False | False | True | 97.41s  | 97.97% | 21.67s|
| #11| YCrCb | 9 | 8 | 2 | ALL| (32, 32) | 32 | True  | True  | True | 126.75s | 98.85% | 32.79s|
| #12| YCrCb | 9 | 8 | 2 | ALL| (16, 16) | 32 | True  | True  | True | 135.49s | 98.76% | 20.17s|

Here I tried color space of HSV, YUV and YCrCb. From the results we can find some phenomena:

1. The results of test accuracy using the three color spaces are almost the same, but YCrCb takes the least training time.
2. With all the hog channel can obviously obtain more test accuracy than with only one hog channel.
3. HOG channel has more efficient than color features including spatial feature and color hist feature.
4. Spatial feature has almost no efficient on the result, but cost much time.

From the training approach above, I finally chose the following parameters:
```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

With these parameters, the length of feature vector for each image is **6156**.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected features (and color features if you used them).

With the method above to extract the desired features from an image, it's time to use those features to train a classifier to identify an image is a car or not a car.

Before training, the features extract has been shuffled from the training images. Also, each images's feature value is normalized with a `StandardScaler()` before given to learning.

Here is a feature value of one random image:

![alt text][image3]

It can be seen that in the end of color histogram features, the feature value is significantly high. After the StandardScaler the features combination became to the same order of magnitude.

In the #9 code cell, I trained the classifier with a linear SVM.
From the training data, the features are extracted from 17,760 images, about half containing vehicles and half not containing vehicles. These features are labeled as "0" or "1" (car or notcar) and passed into a Support Vector Machine that linearly separates the data. Train-test data split method is used here to split 20% data into test data.

The final test accuracy is 98.76%!

The classifier with data were stored in the `svc_pickle.p` pickle file.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section titled "Method for Using Classifier to Detect Cars in an Image" of the #11 and #12 code cell, I adapted the method `find_cars()` from the lesson materials. The method combines HOG feature extraction with a sliding window search. But rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image and then these full-image features are sub-sampling according to the size of the window and then fed to the classifier.

The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

The y_start_stop value of sliding window roi is set to [**400**, **650**] to ignore the sky, trees and car header. The scale value for sliding window is **1.5**.

The image below shows the attempt at using find_cars on one of the test images by HSV color space and YCrCb color space, using a single window size:

* with HSV color space features:

![alt text][image4]

* with YCrCb color space features:

![alt text][image5]

It can be seen that YCrCb has better result of the test images.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Finally I searched with a single function that can extract features using hog sub-sampling, using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

Here are all the sliding window results for all the test images:

![alt text][image6]

The optimization of the SVM classifier can be referred in the SVM training approach. Other optimization techniques included changes to window sizing and overlap as described above, and lowering the heat-map threshold to improve accuracy of the detection (higher threshold values tended to underestimate the size of the vehicle).

-------------------

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/Iue9fThCfwU) .

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

##### - Heat-map Detection

The code for heat-map detection is in the cells #13 to #16.

To solving multiple detections and false positives, I applied a heat-map based on the result of sliding window. To generate this heat-map, I simply added "heat" (+=1) in `add_heat()` for all pixels within windows where a positive detection is reported by the SVM classifier.
Then the "hot" parts of the map are where the cars are, and by imposing a threshold `apply_threshold()`, the false positives with less heat can be rejected.

With the `scipy.ndimage.measurements.label()` function to identify individual blobs in the heat-map, the final detection area is set to the extremities of each identified label:

![alt text][image7]

Here are the resulting bounding boxes of the six test images:

![alt text][image8]

##### - Vehicle Detection Tracking

The code for vehicle detection tracking is in the cells #19 to #20.

In order to produce a more stable and robust video output, I create an class called `Vehicle_Detect()`.

This class can save the history bounding boxes into `prev_boxs` from the previous 10 frames of the video. When one vehicle feature is detected, rather than performing the heat-map/threshold/label steps for the current frame's detections, the detections for the past 10 frames are combined and added to the heat-map.

This method is a kind of time filter for the vehicle detection heat-map. The threshold for the heat-map is set to `1 + len(det.prev_boxs)//2`. This adaptive threshold was found to perform well for this project video.


--------------

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Extract HOG feature once and use it for all windows definitely is a more efficient method for doing the sliding window approach. It saves me lot of time for trying parameters such as window size and overlap.
2. Heat-map method will have better effort when keep track of most recent N frames detection results than only use one frame.
3. When one vehicle just appears from behind, or is covered by another vehicle, the detection may be not in time. With more training image of partly vehicles maybe have better performance for it. Prediction will be more robust with data augmentation trick, also the heat-map threshold shall be adapted.
4. For further usage for vehicle detection, determine vehicle location and speed will be more useful in self-driving. So how to extract this information from the bounding box and how to optimization will be questions in future.

