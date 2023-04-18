# BrainAnomalyDetection
This project aims to classify a brain CT scan as anomaly/not anomaly, given a set of 15.000 training images and 2.500 validation images, respectively their labels, using ML algorithms. My approaches for this task include models such as **Naive Bayes**, **CNN** architectures, and the **ResNet18** architecture.

### 1. Image preprocessing
Firstly, I have rescaled the images from 224 x 224 into the 128 x 128 format and I have also transformed their color channel from RGB to **grayscale**. Henceforth, I normalized the images by dividing their pixel values by 255, to obtain new pixel values between the interval **[0,1]**.

### 2. Data augmentation
While trying different classifiers, especially neural networks, I concluded that there are not enough images for training, so I decided to augment more, to have a bigger training image set. The augmentation rotates, shifts, and changes the shear of images. For this, I have used the ImageDataGenerator from Keras.

### 3. Naive Bayes
My first approach was to use the Naive Bayes classifier, which is based on Bayes' conditional probability theorem. For this model, I did not normalize the images because I chose to transform the pixels from _continuous to discrete values_. Having a desired number of intervals, each pixel's value was transformed into a number that represents the corresponding interval: from values between [0, 255] to *[0, num_of_intervals]*.

### 4. CNN arhitecture
Starting from a little arhitecture with 4 convolutional layers, each one follow by a MaxPooling layer, I have developed my way into *ResNet18*, with which I have obtained my best performance.
