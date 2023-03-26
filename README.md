


  <h3 align="center">Facial expressions recognition</h3>

  <p align="center">
	  Wojciech Czerwiński 147641
	</p>
	  <p align="center">
   Deep Learning (Convolutional Neural Network)
   </p>
    <p align="center">
   <b>course: Intro to Artificial Intelligence @ Poznań University of Technology</b>
	</p>


## Table of contents

- [About the project](#about-the-problem)
- [State of art](#state-of-art)
- [Description of chosen solution](#description-of-chosen-solution)
- [Implementation](#implementation)


## About the project

The chosen problem is to classify facial expressions using a <b>CNN - Convolutional Neural Network</b> - a type of deep learning algorithm architecture based on convolutional layers, particularly used in image data processing. 
- ### Input 
		Images of faces in 48x48 resolution, all images are in greyscale. 
		Images show faces expressing 7 different emotions (classes in the model):
		angry, disgusted, fearful, happy, neutral, sad, surprised
- ### Desired effect
		The purpose of the project is to classify the images given by the user based on a trained model.
		E.g. giving an image of a smiling face (can be any resolution and color palette)
		the model should describe it with the class 'happy'
- ### Motivation 
		Deep learning using neural networks, and in particular CNNs for problems 
		related to image processing, is a future-oriented technology that can significantly 
		improve quality and accelerate development in many areas of everyday life,
		e.g. industry, medicine, aviation.
## State of art
| **Chosen machine learning types** | **short description** | Pros| Cons|
|:---:|:---:|:---:|:---:|
| CNN - convolutional neural networks | Convolution layers extract features from images <br>and then process them on layers that act as classifiers<br>| - effective for object and face recognition<br>- simpler architecture than generic DNN<br>- Convolution layers for efficient feature extraction | -May be less effective in recognising more complex objects<br>- high cost of the calculations performed |
| DNN - deep neural networks | <br>A very large number of hidden layers that process data and are thus able to train the model <br>| - effective for object and face recognition and image generation<br>- The architecture allows for more complex models | - They need a very large amount of data<br>-The learning process is even longer than CNN |
| Learning through feature extraction |<br> From the input data, key features are extracted, which are then used to teach the model<br>| - simplicity, which allows a focus on the key features of the data<br>- effective when features are well described | - less effective than CNN and DNN |

## Description of chosen solution
As presented in the previous sections, <b> convolutional neural networks </b> are specifically designed for image processing. The operation of a CNN relies on a convolution of the input data with a filter to detect patterns in the image.

CNN's architecture consists of several layers:
1.  Convolution layer: this layer consists of filters that are used to detect patterns in the image. The filters move through the image and perform convolution with the input data to detect patterns in different parts of the image.
2.  The max-pooling layer: this layer consists of reducing the size of the input data by applying max-pooling operations (converting several pixels into one, keeping only the largest value).
3.  Hidden layers: the max-pooling layer is followed by hidden layers that process features from the convolution and max-pooling layers.
4.  Output layer: this layer is responsible for classifying the image based on the features extracted by the previous layers.

Images of faces are needed as <b>input data.</b> Datasets in the form of images for training artificial intelligence models are available on the internet. For my problem I used:
- https://www.kaggle.com/datasets/msambare/fer2013

<b> The output of the algorithm </b> for every image is a list of as many values in the range < 0.0 ; 1.0 > as many classes of input data. Each successive value corresponds to a prediction of how much the algorithm estimates that the image belongs to the class with that index. Example:
If the algorithm returns [0.013807331, 0.003256767, 0.9232555, 0.02688416, 0.028897915, 0.0038983417], it means that it classifies the input image as follows:

<b>0.013807331 * 100%</b>, that it's labeled as <b>angry</b>

<b>0.003256767* 100%</b>, that it's labeled as <b>fearful</b>

<b>0.9232555* 100%</b>, that it's labeled as <b>happy</b>

<b>0.02688416* 100%</b>, that it's labeled as <b>neutral</b>

<b>0.028897915* 100%</b>, that it's labeled as <b>sad</b>

<b>0.0038983417* 100%</b>, that it's labeled as <b>surprised</b> 

One class was skipped due to implementation.

In order to apply the CNN architecture, very large computational resources and large amounts of properly prepared data are needed to make the model efficient and as reliable as possible. You also need a well-designed model, which is a very difficult task and depends strongly on the problem to be solved.
## Implementation
In my implementation, I used the TensorFlow library.

### Project stages
- #### Teaching the model
	 Training and test datasets were loaded. The data <b> was not preprocessed</b> which could have influenced the very poor performance of the model. I resigned from preprocessing, due to the fact that the data had already been prepared for the learning process. I then implemented the following neural network based on available materials on the internet:
```python
model = tf.keras.Sequential([

tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48, 48, 1)),

tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),

tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

tf.keras.layers.Dropout(0.25),

tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'),

tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'),

tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

tf.keras.layers.Dropout(0.25),

tf.keras.layers.Flatten(),

tf.keras.layers.Dense(1024, activation='relu'),

tf.keras.layers.Dropout(0.5),

tf.keras.layers.Dense(6, activation='softmax')

])
```

Then the model is compiled and trained. The model in my project achieved very poor results, including strong overfitting. This may have been due to poor quality input images, and inappropriate layers and neural network parameters. To minimally improve the quality of the model, I dropped the 'disgusted' class, which contained significantly fewer images than the other classes. This slightly improved the loss and accuracy rates, whose learning curves I have shown below (for the training and test datasets):

![Figure_1](https://user-images.githubusercontent.com/76266906/213817023-a8522f20-70cd-4faf-968d-5c7350de49cf.png)
![Figure_2](https://user-images.githubusercontent.com/76266906/213817033-297a4ba1-d712-48b5-ba43-43cf0078da59.png)

From some early point, the curve for the test data has remained constant.

Numerical indicators of the loss and accuracy of such a trained model:

```Loss for the train data set: 0.2262008637189865
Loss for the test data set: 1.133596420288086
Accuracy for the train data set: 0.96692955493927
Accuracy for the test data set: 0.6197820901870728
```
As can be read from the indicators above, the accuracy rate for the training data is ~0.97, where for the test data it is ~0.62.

I also evaluated the model using confusion matrix, which looks like this:

- training data set

![macierz1](https://user-images.githubusercontent.com/76266906/213818756-9fe8beab-fbb5-4b70-acc1-7446ec4629b8.png)

- test data set

![macierz2](https://user-images.githubusercontent.com/76266906/213818765-e45614b7-4516-41f1-a7b8-e04671fc1860.png)


The clear dominance of the class with an index of 2 - fearful - can be seen in both the training and test data.
