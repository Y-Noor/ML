# Documentation of my ML journey 
## Machine Learning from A - Z course by Kirill Eremenko and Hadelin de Ponteves
### Topics covered so far:
#### Data preprocessing
- Splitting dataset for testing and training
- Feature scaling
- Taking care of missing values
- Encoding categorical data
#### Regression
- Simple linear regression
- Multiple linear regression
- Polynomial regression
- Support Vector Regression
- Decision trees
- Random forest regression
- Evaluation using R^2
#### Classification
- Logistic regression
- K Nearest Neighbor (KNN)
- Support Vector Machine (SVM)
- Kernel SVM
- Naive Bayes
- Decision trees
- Random forest

#### Clustering
- Kmeans
- K++
- Hierarchical Clustering

#### Association Rule learning
- Apriori
- Eclat

#### Reinforcement Learner
- Upper Confidence Bound
- Thompson sampling

#### Natural Language Processing
- Bag of words model

#### Deep learning
- Artificial Neural Network (ANN)
- Convolution Neural Networks (CNN)

#### Dimensionality reduction
- Principle Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

#### Model Selection 
- K-fold cross validation
- Grid search

#### XGBoost

### Tech used
- Python
  - Pandas
  - Numpy
  - Matplotlib
  - ScikitLearn
      - preprocessing
      - imputer
      - tree
      - 
- NLTK
- Tensorflow
- Keras
- Google Colab


# Computer Vision (kudos to CS231n <3)
**Image Classification problem**: is the task of assigning an input image one label from a fixed set of categories.

The *inexhaustive list of challenges* involved from the perspective of a Computer Vision algorithm:
- Viewpoint variation. A single instance of an object can be oriented in many ways with respect to the camera.
- Scale variation. Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image).
- Deformation. Many objects of interest are not rigid bodies and can be deformed in extreme ways.
- Occlusion. The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.
- Illumination conditions. The effects of illumination are drastic on the pixel level.
- Background clutter. The objects of interest may blend into their environment, making them hard to identify.
- Intra-class variation. The classes of interest can often be relatively broad, such as chair. There are many different types of these objects, each with their own appearance.

A good image classification model must be invariant to the cross product of all these variations, while simultaneously retaining sensitivity to the inter-class variations.

![visionprobs](https://github.com/Y-Noor/ML/assets/59338864/5d07a64a-4547-4cf7-b249-aff3267a378c)


The **image classification pipeline**. We’ve seen that the task in Image Classification is to take an array of pixels that represents a single image and assign a label to it. Our complete pipeline can be formalized as follows:

- *Input*: Our input consists of a set of N images, each labeled with one of K different classes. We refer to this data as the training set.
- *Learning*: Our task is to use the training set to learn what every one of the classes looks like. We refer to this step as training a classifier, or learning a model.
- *Evaluation*: In the end, we evaluate the quality of the classifier by asking it to predict labels for a new set of images that it has never seen before. We will then compare the true labels of these images to the ones predicted by the classifier. Intuitively, we’re hoping that a lot of the predictions match up with the true answers (which we call the ground truth).

## Nearest Neighbor Classifier
As our first approach, we will develop what we call a Nearest Neighbor Classifier. This classifier has nothing to do with Convolutional Neural Networks and it is very rarely used in practice, but it will allow us to get an idea about the basic approach to an image classification problem.

Example image classification dataset: [*CIFAR-10*](https://www.cs.toronto.edu/~kriz/cifar.html). This dataset consists of 60,000 tiny images that are 32 pixels high and wide. Each image is labeled with one of 10 classes (for example “airplane, automobile, bird, etc”). These 60,000 images are partitioned into a training set of 50,000 images and a test set of 10,000 images. 

 The nearest neighbor classifier will take a test image, compare it to every single one of the training images, and predict the label of the closest training image.
 A reasonable choice for comparing them might be the L1 distance
 An example of using pixel-wise differences to compare two images with L1 distance. Two images are subtracted elementwise and then all differences are added up to a single number. If two images are identical the result will be zero. But if the images are very different the result will be large.


 **As an evaluation criterion, it is common to use the accuracy, which measures the fraction of predictions that were correct.**
