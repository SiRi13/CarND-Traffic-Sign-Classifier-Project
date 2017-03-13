#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[overview]: ./images/signs_overview.png "Traffic Signs Overview"
[distribution]: ./images/distribution.png "Distribution of Images by Labels"
[augmentation]: ./images/image_augmentation.png "Example Image with different Augmentation Techniques"
[distributionAugmented]: ./images/distribution_augmented.png "Distribution with Augmented Data"
[meanOfImageSets]: ./images/mean_of_sets.png "Mean Values from Image Sets before and after Normalization"
[augmentedOriginal]: ./images/augmented_original_compared.png "Random Augmented Data Compared To Examples From Original Images"
[augmentedNormalizedOriginal]: ./images/augmented_normalized_original.png "Random Normalized Images Compared To Random Original Images"
[normalizedNew]: ./images/normalized_new.png "Normalized New Image Compared To Original New Image And Similar From GTSRB Set"
[predictionExample]: ./images/predicted_0.png "Example Of Predicted Image with Softmax Distribution"
[predictionExampleNeg]: ./images/predicted_3.png "Example Of Wrong Prediction"
[visualizationImages]: ./images/visualization_source_images.png "Test Images For Network Visualization"
[visualizationOutputConv1]: ./images/visualization_conv1.png "Output of Convolutional Layer #1"
[visualizationOutputConv2]: ./images/visualization_conv2.png "Output of Convolutional Layer #2"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my
* [project code](https://github.com/SiRi13/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* [writeup.pdf](https://github.com/SiRi13/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.pdf)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the [third code cell](Traffic_Sign_Classifier.ipynb#basic-summary-1) of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of the original training set is *34799*
* The size of the original test set is *12630*
* The shape of a traffic sign image before processing it is *(32, 32, 3)*
* The number of unique classes/labels in the data set is *43*

To be sure in each set are the same amount of labels and pictures, I compared the length of each image list with their 
corresponding label list.


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

In the [seventh code cell](Traffic_Sign_Classifier.ipynb#thumbnails) of the notebook I print an random image of each class with its corresponding label.

![Overview of sign images][overview]

To see how many images each class contains I plotted bar chart in [cell eight](./Traffic_Sign_Classifier.ipynb#distribution) which shows how the data is distributed across the classes. This clearly shows how the amount of images varies per class in the training data set.

![Distribution of amount of images across the labels][distribution]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the code cell [Normalization](./Traffic_Sign_Classifier.ipynb#normalization).

First step of preprocessing is equalizing the histogram of each image.
This will increase the global contrast for most images and therefore improve accuracy of the network.
The second step is conversion to grayscale which improves in combination with normalization in step three the network speed as well as its accuracy. 

Here is an example of twelve different signs of six classes. The processed image on the left hand side and respectively an similar from the original set on the right hand side.

![Normalized images compared with similar originals][augmentedNormalizedOriginal]

As shown in the following bar chart, the mean value of the train and test set dropped from over 80 to almost 0 on the right hand side after normalizing.

![Mean Value Of Image Sets][meanOfImageSets]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

There was no splitting in test, validation and training data necessary because there were already split sets in seperate files available.

Before balancing the training set I had *34799* images for training, *4410* for validating and *12630* for testing.  
Since I only expanded the training set, validation and test set stayed the same.

![Balanced Training Set][distributionAugmented]

For the training set I created in [code cell sixteen](./Traffic_Sign_Classifier.ipynb#augmentation) *51382* images by rotating, translating and equalizing the Y-value as shown below.

![Augmentation Examples][augmentation]

I decided to augment and balance the training set because more data is always better and to reduce affects of bad distributed data. This could lead to worse recall of similar images with different amounts of data.  
Lets take Class #35 and #36 for example. The original set had about 1000 images of #35 and only a little over 250 of #36.
With those numbers the network sees #35 about four times more often and therefore can recognize it better than #36.
Since they are pretty similar, #35 is Ahead Only and #36 is Straight Or Right, the network will tend to predict #35.
With augmented data the network gets about the same number of images from any class which recudes the risk of tendencies towards images with higher numbers.
Here is an example of an original image and an augmented image:

![Augmented and Original Image][augmentedOriginal]

My final training set has *86181* images which makes about 2000 images per class as shown in [this bar chart](#distributionAugmented).


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 