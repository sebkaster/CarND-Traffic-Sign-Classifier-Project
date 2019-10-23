# **Traffic Sign Recognition** 

---

The steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image9]: ./examples/architecture.png "Architecture"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new-test-images/test1.jpg "Traffic Sign 1"
[image5]: ./new-test-images/test2.jpg "Traffic Sign 2"
[image6]: ./new-test-images/test3.jpg "Traffic Sign 3"
[image7]: ./new-test-images/test4.jpg "Traffic Sign 4"
[image8]: ./new-test-images/test5.jpg "Traffic Sign 5"
[image10]: ./new-test-images/test6.jpg "Traffic Sign 6"
[image11]: ./examples/orig_aug.png "Augmentation Original"
[image12]: ./examples/rotate_aug.png "Augmentation Rotation"
[image13]: ./examples/translate_aug.png "Augmentation Translation"
[image14]: ./examples/scale_aug.png "Augmentation Scale"
[image15]: ./examples/warp_aug.png "Augmentation Warping"
[image16]: ./examples/brightness_aug.png "Augmentation Brightness"

---

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 12630
* The size of the validation set is 4410
* The size of test set is 34799
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the training set is distributed among the classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing of Data

As a first step I decided to perform a histogram equalization on all images of the test, validation
and training data set in order to improve the contrast. Since histogram equalization only applies to
the inensity channel I converted the RGB image to YUV colorspace and performed the histogram equalization 
on the Y channel. Finally, the image is converted back to RGB colorspace.

The training data set consists of 12630 labeled images. While this seems to be a great number it is advantageous to have more data samples with different attributes.
Moreover, the training data set is unevenly distributed as can be seen in the bar chart diagram of the last section (see Data Set Summary & Exploration).
In order to extend this data set, image augmentation is performed on images of the training data set. Therefore, the following operations are considered:

* random brightness: The brightness of the image is randomly adjusted. This operation is executed on the V channel of the HSV colorspace.
* random translate: Translation by random factor in the range of [-2,2]
* random scale: Scaling of the image by a random factor in the range of [-2, 2]
* random rotate: Rotation of the image by a random value in the range of [-20, 20].
* random warp: Warp fixed source points to random destination points.

The following images show the augmentation operations on an example image:

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

I generated image augmentation until all 43 classes had 3500 samples.

As a last step, I converted the images to greyscale and normalized them. The normalization is essential since the
 ranges of the distributions of feature values are likely be different for each feature.


#### 2. Final model architecture

My final model consisted of the following layers:

![alt text][image9]


#### 3. Parameters of the Model

For the training I used the Adam optimizer and an decreasing learning rate. Dependent on the current epoch the, the learning rate changes exponentially. 
Furthermore, I used a batch size of 128 and 100 epochs. After each epoch the model is saved if it is better (based on accuracy) than any previous one. By doing so we keep track of the best model we trained during the training process.

Finally, the configuration of the best trained model is used for the prediction on new, unseen images.


#### 4. Description of the approach taken for finding a solution

I started by implementing a basic version of the LeNet architecture presented in the Udacity lectures. Based on this architecture I tried to optimize the performance of the model.
Firstly, I added dropout layers and l2-regularizers for regularizatio in order to avoid overfitting. Furthermore, I played around with adding and removing dense layers. Finally, I decided to remove one dense layer which gave the best performance for me.

Moreover, I used batch normalization to increase the independence of the single layers and improve the training speed.

My final model results were:
* training set accuracy of 97.6 %
* validation set accuracy of 97.1 %
* test set accuracy of 97.3 %

### Test a Model on New Images

#### 1. Five German traffic signs found on the web.

You can find the new images in the folder _new-test-images_.

* _test1.jpg_: Computer animation of the right ahead traffic sign.

* _test2.jpg_: Damaged Ahead only image. 

* _test3.jpg_: Approximately bottum-up view of the go straight or left traffic sign.

* _test4.jpg_: Different style of the children crossing traffic sign.

* _test5.jpg_: Slightly covered image of a no entry traffic sign.

* _test6.jpg_: Speed Limit 120 km/h traffic sign represented by lights. Quiet common on the german Autobahn (highway).

#### 2. Model's predictions on these new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead      		| Turn right ahead   									| 
| Ahead only    			| Ahead only 										|
| Go straight or left					| Go straight or left										|
| Children crossing	      		| Children crossing					 				|
| No entry			| No entry      							|
| Speed limit (120km/h)			| Speed limit (120km/h)	     							|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.3 %. 

#### 3. Performance of Model on New Images
The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is completely sure that this is a turn right ahead traffic sign (~100%).
This shows that the trained model is not only able to classify real world images but also artificially created traffic signs.
 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn right ahead 									| 
| <0.01     				| Keep left										|
| <0.01					| End of no passing by vehicles over 3.5 metric tons											|
| <0.01      			| Vehicles over 3.5 metric tons prohibited					 				|
| <0.01				    | End of all speed and passing limits      							|

The second image is correctly classified as ahead only traffic sign (~100%). This shows that also slightly deformed images are no problem.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only 									| 
| <0.01     				| Go straight or left 										|
| <0.01					| Turn right ahead											|| <0.01      			| Keep left					 				|
| <0.01				    | Yield      							|

The third image is correctly classified as ahead only ahead only sign (~100%). The model is able to classify traffic signs regardless of their perspective.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.67         			| Go straight or left								| 
| 0.16     				| Turn left ahead 										|
| 0.15					| Ahead only											|
| 0.01      			| Bumpy road					 				|
| <0.01				    | Turn right ahead      							|

For the fourth image, the model is relatively sure that this is a children crossing sign (~67%). 
The model suffers from the fact that this traffic sign has another appearance as the one used for training of the model.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.47         			| Children Crossing								| 
| 0.20     				| Road narrows on the right 										|
| 0.10					| General caution											|
| 0.08      			| Beware of ice/snow					 				|
| 0.08				    | Bicycles crossing      							|

For the fourth image, the model is very sure that this is a no entry sign (~93%). The small covered area only has a small impact on the model's prediction quality.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.93         			| No entry							| 
| 0.02     				| No passing for vehicles over 3.5 metric tons 										|
| 0.02					| Keep right											|
| 0.01      			| No passing					 				|
| <0.01				    | Stop      							|

For the fifth image, the model is unsure to which traffic sign this image corresponds to. 
While the highest probability is assigned to the correct class, also other speed limit class get significant probabilities.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.37        			| Speed limit (120km/h)							| 
| 0.18     				| Speed limit (80km/h) 										|
| 0.14					| Speed limit (100km/h)											|
| 0.13      			| Speed limit (50km/h)					 				|
| 0.13				    | Speed limit (60km/h)      							|