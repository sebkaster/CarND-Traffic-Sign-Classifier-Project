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
[image10]: ./new-test-images/test6.jpg "Traffic Sign 5"

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

#### 1.  

I decided to generate additional data because ... 

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...


To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Final model architecture

My final model consisted of the following layers:

![alt text][image9]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the training I used the Adam optimizer and an decreasing learning rate. Depend on the epoch the learning rate changes exponentially. 
Furthermore, I used a batch size of 128 and 100 epochs. After the training is finished 

#### 4. Description of the approach taken for finding a solution

My final model results were:
* training set accuracy of 97.6 %
* validation set accuracy of 97.1 %
* test set accuracy of 97.3 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Five German traffic signs found on the web.

You can find the new images in the folder _new-test-images_.

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

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Turn right ahead 									| 
| <0.01     				| Keep left										|
| <0.01					| End of no passing by vehicles over 3.5 metric tons											|
| <0.01      			| Vehicles over 3.5 metric tons prohibited					 				|
| <0.01				    | End of all speed and passing limits      							|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only 									| 
| <0.01     				| Go straight or left 										|
| <0.01					| Turn right ahead											|
| <0.01      			| Keep left					 				|
| <0.01				    | Yield      							|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.67         			| Go straight or left								| 
| 0.16     				| Turn left ahead 										|
| 0.15					| Ahead only											|
| 0.01      			| Bumpy road					 				|
| <0.01				    | Turn right ahead      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.47         			| Children Crossing								| 
| 0.20     				| Road narrows on the right 										|
| 0.10					| General caution											|
| 0.08      			| Beware of ice/snow					 				|
| 0.08				    | Bicycles crossing      							|


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.93         			| No entry							| 
| 0.02     				| No passing for vehicles over 3.5 metric tons 										|
| 0.02					| Keep right											|
| 0.01      			| No passing					 				|
| <0.01				    | Stop      							|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.37        			| Speed limit (120km/h)							| 
| 0.18     				| Speed limit (80km/h) 										|
| 0.14					| Speed limit (100km/h)											|
| 0.13      			| Speed limit (50km/h)					 				|
| 0.13				    | Speed limit (60km/h)      							|


For the second image ... 
