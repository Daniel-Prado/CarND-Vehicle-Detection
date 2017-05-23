# **Vehicle Detection and Tracking**
#### by Daniel Prado Rodriguez
#### Udacity SDC Nanodegree. Feb'17 Cohort.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/fig1_car_nocar.png
[image2]: ./examples/fig2_hog_sample.png
[image3.1]: ./examples/fig3_1_sliding_far.jpg
[image3.2]: ./examples/fig3_2_sliding_mid.jpg
[image3.3]: ./examples/fig3_3_sliding_close.jpg
[image4]: ./examples/fig4_sliding_local.jpg
[image5]: ./examples/fig5_heat_labels_boxes.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

All the code for this project is self-contained in the repo Jupyter notebook `vehicle_detection_notebook.ipynb`.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by loading in all the `vehicle` and `non-vehicle` images, and observing its characteristics. Below I show a random subset of `vehicle` and `non-vehicle` classes of the dataset:

![alt text][image1]

The code for the feature extraction s step is contained in the second section (2. Define Processing Functions) of the IPython notebook that contain all the functions seen in the lessons. In particular, for HOG features see function `get_hog_features`.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

The details of how these parameters were chosen are discussed in the next section.

#### 2. Explain how you settled on your final choice of HOG parameters.

I explored several combinations of parameters that would provide better training results. The preliminary decision was taken to maximize the value of the sklearn accuracy function, to be at least 0.99 and then confirmed with the good performance for class prediction in the test images and video sequence.
Those parameters for HOG were:
* `color space` : tried with all the possible color spaces (RGB, HSV, LUV, HLS, YUV, YCrCb), I also experimented with several channels. In the case were the color space has a ‘luminance’ channel, that one gave the best results.
My final decision was to use YUV.
* `orientations` : I tried with values of 9 and 6. Finally went with 6 as this improved the speed without hindering accuracy much.
* `pixels_per_cell`: I experimented with some other values (16), that seemed to provide greater execution speed but lower accuracy. I kept the default value of (8,8).
* `cells_per_block`: I kept the value of (2,2).

As explained above,the parameters were trained based on what we have seen in the lessons, plus a trial and error approach and a bit of intuition.
As said before, the YUV color space with ALL its channels provided a very good accuracy for the database, together with the other chosen parameters). I found also that reducing the orientations from 9 to 6 did not reduce very much the accuracy while significally reducing the number of features, and hence increasing the classifier speed.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier training is executed in the section "Run the Classifier" using the function `train_classifier`.  As discussed on the lessons, the main characteristics of the model are the following:
* Implemented with SciKit-Learn. Great library for rapid prototyping of machine-learning algorithms.
* Chose to stick to the default Support Vector Machines (SVM) Linear classifier, which provides a good trade-off of accuracy vs. speed.
* The features vector was composed of HOG features, together with Spatial Bins and Color Histograms. The lenght of the feature vector is 4392 features.
* Features are normalized and scaled using the Sklearn StandardScaler function.
* Test Size is 20% of the dataset (and randomly chosen using the Sklearn train_test_split function.
* This setting reaches an Accuracy of 99.0 %

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is perhaps were I have tried to bring most original work of my own to this project.
As we have seen in the lessons, the performance of the `search_windows` function for extracting the HOG features (and other features) from hundreds of windows in every frame is not very efficient.

Of course a python notebook is not an environment for real time applications, but the processing of the video sequence provided can take really long (around 35  min. in my laptop).

To overcome this issue, the Q&A session proposed so get the HOG features in a different way: calculate the features for the entire region of interest in the frame and slicing to get the features in every ‘window’.

However I have tried a different approach. My thought was that probably we don’t need to scan the full region of interest in every frame (30 frames per second). Instead, we could perform a full scan every N frames (if N=15, then twice per second), and in the frames between we can calculate local (small) regions of interest around the vehicles detected in the full scans (or previous local scans).

This enables to track the vehicles movement accurately, reduce the chance of false positives, and dramatically increase performance a tenfold.

We need to mention however that this approach has its drawbacks too. If full scans are very spaced, we introduce a noticeable delay in the vehicle detection. In a real-world system perhaps 2 times/second would not be enough frequency, but it serves for this project.  This risk increases if in a full scan we have some false-negative.

So, to summarize, my project implements two types of scan:

**Full Scans**: full scans divides the region of interest in 3 areas, depending on the relative size of the vehicles vs. distance. Those areas overlap as it can be seen in the following images. The three areas have defined by the parameters shown in the notebook, under section "Design a multi-scale Window Set".
Below you can see the three sliding window areas presented separately. Overlapping value in the x and y direction is 0.75 in all cases.

"Far" set, with window size of 64x64 pixels:

![alt text][image3.1]

"Middle" set, with window size of 128x128 pixels:

![alt text][image3.2]

"Close" set, with window size of 192x192 pixels:

![alt text][image3.3]

**Local Scans** : for every detected vehicle (or labelled object) in the previous frame, a local area is defined. It’s not possible to define a exact size that this area should have because the area of the labelled objects is not very reliable, but basically I have defined such area based on the height of the label, that is more stable than the width. Local areas can also overlap between different vehicles.

The local scan sliding window set is constructed dinamically for each frame in the method `get_local_search_windows` of section "Dynamic Sliding Window Set (Local Scan)"

Below you can see an example of the local area search calculated (in gray) for the next frame, based on the labelled objects areas of the current frame (blue):

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided an acceptable result.  Here are some example images (snapshots extracted from the output video sequence).

###IMAGE
![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To reduce the false positives, I used the technique of calculating a heat map of overlapping slide windows. Note that overlap is applied in both the full and the local search scans.

To go to one step beyond what is shown in the lessons, instead of calculating this heatmap in each frame individually, I accumulate the heatmap over a number of previous frames.
This is implemented in the class `Vehicle` in the Video Pipeline section, and the number of frames over which the heatmap is accumulated is the parameter `smooth_factor = 7` in my final run.

The heatmaps are thresholded to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.
Note that, as explained earlier, these blobs are the basis to designate a local area search in the following frames.


### Pipeline Frame Samples:

Below I present 6 sample frames showing the main 3 steps of the pipeline.
The first column shows the heatmaps constructed as a sum of the windows were a car (or car piece) was detected.
The second column shows the blobs of detected items, based on the output of `scipy.ndimage.measurements.label()`.
The third column presents the final result of drawed boxes around the detected blobs (vehicles)

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As explained before, for the specific aspects of my approach vs. the lessons, the main issue I would face is the fact that I am not scanning the full area of interest in every frame, hence introducing delay in vehicles that could be approaching very fast. To make it more robust, I could still use this technique but with a full scan period much shorter, for example every 5 frames.

To mitigate the computational high load of those full scans I would have implemented the technique of calculating the HOG features for the whole frame instead that for individual windows.

Other ideas that I would like to explore in the future are:
* Exploit the motion flow information of the video sequence :  When our car moves in a highway, the camera captures the motion flow of the environment in front of our vehicle (objects are "approaching" at our vehicle speed). However other moving vehicles in the scene have a different speed, and hence different motion flow information that could help to detect and outline them in every frame. This would not apply to parked or static cars.

* Compare the performance of this Computer Vision approach vs. Convolutional Neural Networks. CNNs can be used not only to classify, has we saw in previous projects, but also to detect vehicles in the image.


