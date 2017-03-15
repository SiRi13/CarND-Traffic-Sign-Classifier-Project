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
[newRaw]: ./images/new_images_raw.png "New images from Google Maps"
[normalizedNew]: ./images/normalized_new.png "Normalized New Image Compared To Original New Image And Similar From GTSRB Set"
[predictionExample]: ./images/predicted_0.png "Example Of Predicted Image with Softmax Distribution"
[predictionExample2]: ./images/predicted_1.png "Example Of Predicted Image with Softmax Distribution"
[predictionExample3]: ./images/predicted_2.png "Example Of Predicted Image with Softmax Distribution"
[predictionExample4]: ./images/predicted_4.png "Example Of Predicted Image with Softmax Distribution"
[predictionExample5]: ./images/predicted_6.png "Example Of Predicted Image with Softmax Distribution"
[predictionExample6]: ./images/predicted_7.png "Example Of Predicted Image with Softmax Distribution"
[predictionExample7]: ./images/predicted_10.png "Example Of Predicted Image with Softmax Distribution"
[predictionExampleNeg]: ./images/predicted_8.png "Example Of Wrong Prediction"
[predictionExampleNeg2]: ./images/predicted_9.png "Example Of Wrong Prediction"
[predictionExampleNeg3]: ./images/predicted_3.png "Example Of Wrong Prediction"
[predictionExampleNeg4]: ./images/predicted_5.png "Example Of Wrong Prediction"

[visualizationImages]: ./images/visualization_source_images.png "Test Images For Network Visualization"
[visualizationOutputConv1]: ./images/visualization_conv1_1.png "Output of Convolutional Layer #1"
[visualizationOutputConv2]: ./images/visualization_conv1_2.png "Output of Convolutional Layer #2"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my
* [project code](https://github.com/SiRi13/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* [writeup.html](https://github.com/SiRi13/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.html)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell below the header [Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas](./Traffic_Sign_Classifier.ipynb#Provide-a-Basic-Summary-of-the-Data-Set-Using-Python,-Numpy-and/or-Pandas) of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of the original training set is *34799*
* The size of the original test set is *12630*
* The shape of a traffic sign image before processing it is *(32, 32, 3)*
* The number of unique classes/labels in the data set is *43*

To be sure in each set are the same amount of labels and pictures, I compared the length of each image list with their 
corresponding label list.


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

In subchapter [Thumbnails](./Traffic_Sign_Classifier.ipynb#Thumbnails) of the notebook I print an random image of each class with its corresponding label to see how they look like.

![Overview of sign images][overview]

To see how many images each class contains I plotted a bar chart in chapter [Distribution](./Traffic_Sign_Classifier.ipynb#Distribution) which shows how the data is distributed across the classes. This clearly shows how the amount of images varies per class in the training data set.

![Distribution of amount of images across the labels][distribution]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the code cell [Normalization](./Traffic_Sign_Classifier.ipynb#Normalization).

First step of preprocessing is equalizing the histogram of each image.
This will increase the global contrast for most images and therefore improve accuracy of the network.
The second step is conversion to grayscale which improves in combination with the normalization in step three the network speed as well as its accuracy. 

Here are random examples of six classes. The processed image on the left hand side and respectively the original image on the right hand side.
This comparsion makes illustrates it nicely, how the contrast usually improved and unimportant details vanished.

![Normalized images compared with similar originals][augmentedNormalizedOriginal]

As shown in the following bar chart, the mean value of the train and test set dropped from over 80 to almost 0 on the right hand side after normalizing.

![Mean Value Of Image Sets][meanOfImageSets]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

No splitting in test, validation and training sets was necessary because there were already split sets in seperate files available.

Before balancing the training set I had *34799* images for training, *4410* for validating and *12630* for testing.  
Since I only expanded the training set, validation and test set stayed the same.

For the training set I created in chapter [Augmentation](./Traffic_Sign_Classifier.ipynb#Augmentation) about *51300* images by rotating, translating and equalizing the Y-value as shown below.

![Augmentation Examples][augmentation]

I decided to augment and balance the training set because more data is always better and to reduce effects of bad distributed data on the accuracy. This could lead to worse recall of similar images with different amounts of data.  
Lets take Class #35 and #36 for example. The original set had about 1000 images of #35 and only a little over 250 of #36.
With those numbers the network sees #35 about four times more often and therefore can recognize it better than #36.
Since they are pretty similar, #35 is Ahead Only and #36 is Straight Or Right, the network will tend to predict #35.
With augmented data the network gets about the same number of images from any class which recudes the risk of tendencies towards images with higher numbers.  
Here are some examples of an original image and an augmented image:

![Augmented and Original Image][augmentedOriginal]

My final training set has *86181* images which makes about 2000 images per class as shown in this bar chart:

![Balanced Training Set][distributionAugmented]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the code cells in chapter [Model Architecture](./Traffic_Sign_Classifier.ipyng#Model-Architecture) of the ipython notebook. 

My final model consists of the following layers:

|#| Layer | Description | 
|---:|:--- |:--- |
|0| Input | 32x32x1 grayscale image | 
| | 						|												|
|1| Convolution 3x3		| 1x1 stride, valid padding, outputs 30x30x100 	|
| | ReLU					|												|
| | Max pooling			| 2x2 stride, 2x2 kernel, outputs 15x15x100		|
| | 						|												|
|2| Convolution 4x4     	| 1x1 stride, valid padding, outputs 12x12x150 	|
| | ReLU					|												|
| | Max pooling			| 2x2 stride, 2x2 kernel, outputs 6x6x150		|
| | 						|												|
|3| Convolution 4x4	    | 1x1 stride, same padding, outputs 6x6x250 	|
| | ReLU					|												|
| | Max pooling			| 1x1 stride, 1x1 kernel, outputs 6x6x250		|
| | 						|												|
|4| Flatten				| outputs (9000)								|
| | 						|												|
|5| Fully connected		| outputs (200)									|
| | ReLU					|												|
| | Drop-out				| keep probability: 50%							|
| | 						|												|
|6| Fully connected		| outputs (43)									|
| | ReLU					|												|
| | Drop-out				| keep probability: 50%							|
| | 						|												|
|7| Fully connected		| outputs (43)									|
| |						|												|
|8| Softmax				|												|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the cells after [Training](./Traffic_Sign_Classifier.ipynb#Training) of the ipython notebook. 

To train the model, I used *25* epochs with a batch size of *128*. I used the same functions for calculating cross entropy and loss as the _LeNet_ network as well as the same *AdamOptimizer* with a learning rate of *0.001*.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the [Accuracy](./Traffic_Sign_Classifier.ipynb#Training) chapter of the Ipython notebook.

My final model results were:
* validation set accuracy of almost 99% 
* test set accuracy of 96.2%

First approach was the _LeNet_ architecture as mentioned in the introdution video by David Silver which reached about 88% with the test data.  
As the input data is always a good start to improve a network, I implemented the normalization and augmentation of the data.
Since I changed the color and therefore the shape of the images from *32x32x3* to *32x32x1* the network had to be changed to.
While at it I also added one convolutional layer and drop-outs between the fully connected layers
The test accuracy jumped right up to 95%. Since *sermanet* from Pierre Sermanet and Yann LeCun used very high feature counts, I increased the depth as well to *100*, *150* and *250* for the convolutional layers and *200* for the first fully connected.  
This architecture reached almost 99% with the validation set and 96.2% with test data. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the German traffic signs that I found on the web:

![New traffic sign images from Google Maps][newRaw]

Image *#4 No Vehicles* might be hard to predict because the actual traffic sign is smaller and there is an additional sign on it as well.
For curiosity I added three signs which are not contained in the given datasets. I want to see if it predicts similar signs.
For *#10 - Bike Lane* it could recognize *#29 - Bicycles Crossing* since it looks similar but with a different shape.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in chapter [Predict the Sign Type for Each Image](./Traffic_Sign_Classifier.ipynb#Predict-the-Sign-Type-for-Each-Image).

Before I started the in-depth analysis of the predictions, I run the images with the `evaluate` function.
Result was *63.6*%.  
Here are the predicted sign classes I got for running each image separately: 

| Image | Prediction | 
|:----- |:-----------| 
| Ahead Only | Stop sign | 
| Stop | Stop |
| Priority Road | Priority Road |
| No Vehicles | End of No Passing |
| Road Work | Road Work |
| No Parking (NIL) | Priority Road |
| No Entry | No Entry |
| General Caution | General Caution |
| Ped/Bike Lane | Stop |
| Bike Lane | No Vehicles |
| Turn Right Ahead | Turn Right Ahead |

The model was able to correctly guess **seven** of the **eleven** traffic signs, which gives an accuracy of **63.6%**.
Considering that three are not in the original set it actually was much higher and therefore a lot closer to the test accuracy of 96.5%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is below [Predict the Sign Type for Each Image](./Traffic_Sign_Classifier.ipynb#Predict-the-Sign-Type-for-Each-Image) as well.

The model was very certain about its predictions and output almost always 100%

![100% Prediction][predictionExample]

Even wrong predictions were very high and mostly 100%.

![100% Prediction Wrong][predictionExampleNeg]

Only one of wrong predicted images got a wide spread softmax distribution of the probabilities.

![Distributed Probabilties][predictionExampleNeg2]

All other results:

![Stop Result][predictionExample2]
![Priority Road Result][predictionExample3]
![Road Work Result][predictionExample4]
![No Parking Result][predictionExampleNeg3]
![No Entry Result][predictionExample5]
![General Caution Result][predictionExample6]
![Ped/Bike Lane Result][predictionExampleNeg4]
![Turn Right Ahead Only Result][predictionExample7]


### Visualize the Neural Network's State with Test Images

#### 1. Discuss how you used the visual output of your trained network's feature maps to show that it had learned to look for interesting characteristics in traffic sign images

To visualize the state of my trained network with the given function, I used those random signs from the training data set:

![Random Training Sign][visualizationImages]

By running it below header [Visualize Image](./Traffic_Sign_Classifier.ipyng#Visualize-Image) the `outputFeatureMap` function produced following plots.

![Output First Sign][visualizationOutputConv1]
![Output Second Sign][visualizationOutputConv2]

This visual output of the activation of the first convolutional layer *conv1* shows nicely how each neuron gets activated and therefore mirrors the input image.
With more details visible in those feature maps the better are the probabilities of recognizing the image.
