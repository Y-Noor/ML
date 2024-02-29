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


There are many other ways of computing distances between vectors. Another common choice could be to instead use the L2 distance, which has the geometric interpretation of computing the euclidean distance between two vectors
```
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```
we could leave out the square root operation because square root is a monotonic function.
The choices such as choosing between L1 and L2 norm are called *hyperparameters* and tweaking them to see what works best is fine but must not be done on the test set. Instead grab a subset of test set (validation set) and use it to tune hyperparameters.  

## k - Nearest Neighbor Classifier
The idea is very simple: instead of finding the single closest image in the training set, we will find the top k closest images, and have them vote on the label of the test image. In particular, when k = 1, we recover the Nearest Neighbor classifier. 

**Evaluate on the test set only a single time, at the very end.**

**Cross-validation**. In cases where the size of your training data (and therefore also the validation data) might be small, people sometimes use a more sophisticated technique for hyperparameter tuning called cross-validation.
You can get a better and less noisy estimate of how well a certain value of k works by iterating over different validation sets and averaging the performance across these. For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.

In practice, *people prefer to avoid cross-validation* in favor of having a single validation split, since cross-validation can be computationally expensive. The splits people tend to use is between 50%-90% of the training data for training and rest for validation. However, this depends on multiple factors: For example if the number of hyperparameters is large you may prefer to use bigger validation splits. If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation. Typical number of folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.

### Pros and Cons of Nearest Neighbor classifier.

- One advantage is that it is very simple to implement and understand. Additionally, the classifier takes no time to train, since all that is required is to store and possibly index the training data. 

- However, we pay that computational cost at test time, since classifying a test example requires a comparison to every single training example. This is backwards, since in practice we often care about the test time efficiency much more than the efficiency at training time. In fact, the deep neural networks we will develop later in this class shift this tradeoff to the other extreme: They are very expensive to train, but once the training is finished it is very cheap to classify a new test example. This mode of operation is much more desirable in practice.

As an aside, the computational complexity of the Nearest Neighbor classifier is an active area of research, and several Approximate Nearest Neighbor (ANN) algorithms and libraries exist that can accelerate the nearest neighbor lookup in a dataset (e.g. FLANN). These algorithms allow one to trade off the correctness of the nearest neighbor retrieval with its space/time complexity during retrieval, and usually rely on a pre-processing/indexing stage that involves building a kdtree, or running the k-means algorithm.

The Nearest Neighbor Classifier may sometimes be a good choice in some settings (especially if the data is low-dimensional), but it is rarely appropriate for use in practical image classification settings. One problem is that images are high-dimensional objects (i.e. they often contain many pixels), and distances over high-dimensional spaces can be very counter-intuitive.

Additional reading: 
- [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
- [Recognizing and Learning Object Categories](https://people.csail.mit.edu/torralba/shortCourseRLOC/index.html)

## Parameterized mapping from images to label scores
### Linear classifier
**Image data preprocessing**
## Loss function
### Multiclass Support Vector Machine loss
### Softmax classifier

[Interactive web demo](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/)