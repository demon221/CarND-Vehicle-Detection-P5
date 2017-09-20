
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
[image8]: ./output_images/output_bboxes.png
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

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

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

In the #9 code cell, I trained the classifier with a linear SVM.
From the training data, the features are extracted from 17,760 images, about half containing vehicles and half not containing vehicles. These features are labeled as "0" or "1" (car or notcar) and passed into a Support Vector Machine that linearly separates the data. Train-test data split method is used here to split 20% data into test data.

The final test accuracy is 98.76%!

The classifier with data were stored in the `svc_pickle.p` pickle file.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the section titled "Method for Using Classifier to Detect Cars in an Image" of the #11 code cell, I adapted the method `find_cars()` from the lesson materials. The method combines HOG feature extraction with a sliding window search. But rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image and then these full-image features are sub-sampling according to the size of the window and then fed to the classifier.

The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

The y_start_stop value of sliding window roi is set to [**400**, **650**] to ignore the sky, trees and car header. The scale value for sliding window is **1.5**.


The image below shows the attempt at using find_cars on one of the test images by HSV color space and YCrCb color space, using a single window size:

HSV:

![alt text][image4]

YCrCb:

![alt text][image5]

It can be seen that YCrCb has better result of the test images. Here are all the sliding window results for all the test images:

![alt text][image6]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

In this step, I basically generate a list of sliding windows, then apply a SVM classifier to the resized window. If the prediction is positive, then add one to a heatmap. To remove false positive, I apply a threshold to the heatmap. Then I use scipy.ndimage.measurements.label to generate a bounding box around the car.

Also, I saved the positions of true detections in each frame of the video. From the true detections I created a heatmap and then thresholded that map to identify vehicle positions:

![alt text][image7]
-------------------

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

