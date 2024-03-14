{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Y-Noor/ML/blob/main/softmax.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "id": "149d5b7e",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "149d5b7e",
        "outputId": "d5a8e9b8-8a35-4f17-ac74-2011528a5494"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
            "/content/drive/My Drive/cs231n/assignment1/cs231n/datasets\n",
            "/content/drive/My Drive/cs231n/assignment1\n"
          ]
        }
      ],
      "source": [
        "# This mounts your Google Drive to the Colab VM.\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "# TODO: Enter the foldername in your Drive where you have saved the unzipped\n",
        "# assignment folder, e.g. 'cs231n/assignments/assignment1/'\n",
        "FOLDERNAME = 'cs231n/assignment1/'\n",
        "assert FOLDERNAME is not None, \"[!] Enter the foldername.\"\n",
        "\n",
        "# Now that we've mounted your Drive, this ensures that\n",
        "# the Python interpreter of the Colab VM can load\n",
        "# python files from within it.\n",
        "import sys\n",
        "sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))\n",
        "\n",
        "# This downloads the CIFAR-10 dataset to your Drive\n",
        "# if it doesn't already exist.\n",
        "%cd /content/drive/My\\ Drive/$FOLDERNAME/cs231n/datasets/\n",
        "!bash get_datasets.sh\n",
        "%cd /content/drive/My\\ Drive/$FOLDERNAME"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "5fdfc27e",
      "metadata": {
        "tags": [
          "pdf-title"
        ],
        "id": "5fdfc27e"
      },
      "source": [
        "# Softmax exercise\n",
        "\n",
        "*Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*\n",
        "\n",
        "This exercise is analogous to the SVM exercise. You will:\n",
        "\n",
        "- implement a fully-vectorized **loss function** for the Softmax classifier\n",
        "- implement the fully-vectorized expression for its **analytic gradient**\n",
        "- **check your implementation** with numerical gradient\n",
        "- use a validation set to **tune the learning rate and regularization** strength\n",
        "- **optimize** the loss function with **SGD**\n",
        "- **visualize** the final learned weights\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "id": "ca6cf2d4",
      "metadata": {
        "tags": [
          "pdf-ignore"
        ],
        "id": "ca6cf2d4"
      },
      "outputs": [],
      "source": [
        "import random\n",
        "import numpy as np\n",
        "from cs231n.data_utils import load_CIFAR10\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "%matplotlib inline\n",
        "plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots\n",
        "plt.rcParams['image.interpolation'] = 'nearest'\n",
        "plt.rcParams['image.cmap'] = 'gray'\n",
        "\n",
        "# for auto-reloading extenrnal modules\n",
        "# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython\n",
        "%load_ext autoreload\n",
        "%autoreload 2"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "id": "9e9bbbbe",
      "metadata": {
        "tags": [
          "pdf-ignore"
        ],
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9e9bbbbe",
        "outputId": "d474cc9e-c4a6-4e83-bcca-3093a86ee998"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Train data shape:  (49000, 3073)\n",
            "Train labels shape:  (49000,)\n",
            "Validation data shape:  (1000, 3073)\n",
            "Validation labels shape:  (1000,)\n",
            "Test data shape:  (1000, 3073)\n",
            "Test labels shape:  (1000,)\n",
            "dev data shape:  (500, 3073)\n",
            "dev labels shape:  (500,)\n"
          ]
        }
      ],
      "source": [
        "def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):\n",
        "    \"\"\"\n",
        "    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare\n",
        "    it for the linear classifier. These are the same steps as we used for the\n",
        "    SVM, but condensed to a single function.\n",
        "    \"\"\"\n",
        "    # Load the raw CIFAR-10 data\n",
        "    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'\n",
        "\n",
        "    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)\n",
        "    try:\n",
        "       del X_train, y_train\n",
        "       del X_test, y_test\n",
        "       print('Clear previously loaded data.')\n",
        "    except:\n",
        "       pass\n",
        "\n",
        "    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)\n",
        "\n",
        "    # subsample the data\n",
        "    mask = list(range(num_training, num_training + num_validation))\n",
        "    X_val = X_train[mask]\n",
        "    y_val = y_train[mask]\n",
        "    mask = list(range(num_training))\n",
        "    X_train = X_train[mask]\n",
        "    y_train = y_train[mask]\n",
        "    mask = list(range(num_test))\n",
        "    X_test = X_test[mask]\n",
        "    y_test = y_test[mask]\n",
        "    mask = np.random.choice(num_training, num_dev, replace=False)\n",
        "    X_dev = X_train[mask]\n",
        "    y_dev = y_train[mask]\n",
        "\n",
        "    # Preprocessing: reshape the image data into rows\n",
        "    X_train = np.reshape(X_train, (X_train.shape[0], -1))\n",
        "    X_val = np.reshape(X_val, (X_val.shape[0], -1))\n",
        "    X_test = np.reshape(X_test, (X_test.shape[0], -1))\n",
        "    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))\n",
        "\n",
        "    # Normalize the data: subtract the mean image\n",
        "    mean_image = np.mean(X_train, axis = 0)\n",
        "    X_train -= mean_image\n",
        "    X_val -= mean_image\n",
        "    X_test -= mean_image\n",
        "    X_dev -= mean_image\n",
        "\n",
        "    # add bias dimension and transform into columns\n",
        "    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])\n",
        "    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])\n",
        "    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])\n",
        "    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])\n",
        "\n",
        "    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev\n",
        "\n",
        "\n",
        "# Invoke the above function to get our data.\n",
        "X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()\n",
        "print('Train data shape: ', X_train.shape)\n",
        "print('Train labels shape: ', y_train.shape)\n",
        "print('Validation data shape: ', X_val.shape)\n",
        "print('Validation labels shape: ', y_val.shape)\n",
        "print('Test data shape: ', X_test.shape)\n",
        "print('Test labels shape: ', y_test.shape)\n",
        "print('dev data shape: ', X_dev.shape)\n",
        "print('dev labels shape: ', y_dev.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "d4c3fb04",
      "metadata": {
        "id": "d4c3fb04"
      },
      "source": [
        "## Softmax Classifier\n",
        "\n",
        "Your code for this section will all be written inside `cs231n/classifiers/softmax.py`.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "id": "25f2e5e1",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "25f2e5e1",
        "outputId": "97b3affc-a1bf-4c1a-ac54-063d181c751d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "loss: 2.381073\n",
            "sanity check: 2.302585\n"
          ]
        }
      ],
      "source": [
        "# First implement the naive softmax loss function with nested loops.\n",
        "# Open the file cs231n/classifiers/softmax.py and implement the\n",
        "# softmax_loss_naive function.\n",
        "\n",
        "from cs231n.classifiers.softmax import softmax_loss_naive\n",
        "import time\n",
        "\n",
        "# Generate a random softmax weight matrix and use it to compute the loss.\n",
        "W = np.random.randn(3073, 10) * 0.0001\n",
        "loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)\n",
        "\n",
        "# As a rough sanity check, our loss should be something close to -log(0.1).\n",
        "print('loss: %f' % loss)\n",
        "print('sanity check: %f' % (-np.log(0.1)))"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "15ebc638",
      "metadata": {
        "tags": [
          "pdf-inline"
        ],
        "id": "15ebc638"
      },
      "source": [
        "**Inline Question 1**\n",
        "\n",
        "Why do we expect our loss to be close to -log(0.1)? Explain briefly.**\n",
        "\n",
        "$\\color{blue}{\\textit Your Answer:}$ *Fill this in*\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "id": "a8cb3eb1",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a8cb3eb1",
        "outputId": "570296af-2684-4f6b-b3d6-f8b4299a51de"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n",
            "numerical: 0.000000 analytic: 0.000000, relative error: nan\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/content/drive/My Drive/cs231n/assignment1/cs231n/gradient_check.py:127: RuntimeWarning: invalid value encountered in scalar divide\n",
            "  rel_error = abs(grad_numerical - grad_analytic) / (\n"
          ]
        }
      ],
      "source": [
        "# Complete the implementation of softmax_loss_naive and implement a (naive)\n",
        "# version of the gradient that uses nested loops.\n",
        "loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)\n",
        "\n",
        "# As we did for the SVM, use numeric gradient checking as a debugging tool.\n",
        "# The numeric gradient should be close to the analytic gradient.\n",
        "from cs231n.gradient_check import grad_check_sparse\n",
        "f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]\n",
        "grad_numerical = grad_check_sparse(f, W, grad, 10)\n",
        "\n",
        "# similar to SVM case, do another gradient check with regularization\n",
        "loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)\n",
        "f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]\n",
        "grad_numerical = grad_check_sparse(f, W, grad, 10)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "id": "4a4a81d4",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4a4a81d4",
        "outputId": "06506830-2946-4048-f48b-80b54d64d86d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "naive loss: 0.000000e+00 computed in 0.000160s\n",
            "vectorized loss: 0.000000e+00 computed in 0.000142s\n",
            "Loss difference: 0.000000\n",
            "Gradient difference: 0.000000\n"
          ]
        }
      ],
      "source": [
        "# Now that we have a naive implementation of the softmax loss function and its gradient,\n",
        "# implement a vectorized version in softmax_loss_vectorized.\n",
        "# The two versions should compute the same results, but the vectorized version should be\n",
        "# much faster.\n",
        "tic = time.time()\n",
        "loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)\n",
        "toc = time.time()\n",
        "print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))\n",
        "\n",
        "from cs231n.classifiers.softmax import softmax_loss_vectorized\n",
        "tic = time.time()\n",
        "loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)\n",
        "toc = time.time()\n",
        "print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))\n",
        "\n",
        "# As we did for the SVM, we use the Frobenius norm to compare the two versions\n",
        "# of the gradient.\n",
        "grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')\n",
        "print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))\n",
        "print('Gradient difference: %f' % grad_difference)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "id": "a3453536",
      "metadata": {
        "tags": [
          "code"
        ],
        "test": "tuning",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a3453536",
        "outputId": "06ba7c59-9b3e-40ce-a2d8-953fa3665fd6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "best validation accuracy achieved during cross-validation: -1.000000\n"
          ]
        }
      ],
      "source": [
        "# Use the validation set to tune hyperparameters (regularization strength and\n",
        "# learning rate). You should experiment with different ranges for the learning\n",
        "# rates and regularization strengths; if you are careful you should be able to\n",
        "# get a classification accuracy of over 0.35 on the validation set.\n",
        "\n",
        "from cs231n.classifiers import Softmax\n",
        "results = {}\n",
        "best_val = -1\n",
        "best_softmax = None\n",
        "\n",
        "################################################################################\n",
        "# TODO:                                                                        #\n",
        "# Use the validation set to set the learning rate and regularization strength. #\n",
        "# This should be identical to the validation that you did for the SVM; save    #\n",
        "# the best trained softmax classifer in best_softmax.                          #\n",
        "################################################################################\n",
        "\n",
        "# Provided as a reference. You may or may not want to change these hyperparameters\n",
        "learning_rates = [1e-7, 5e-7]\n",
        "regularization_strengths = [2.5e4, 5e4]\n",
        "\n",
        "# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****\n",
        "\n",
        "pass\n",
        "\n",
        "# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****\n",
        "\n",
        "# Print out results.\n",
        "for lr, reg in sorted(results):\n",
        "    train_accuracy, val_accuracy = results[(lr, reg)]\n",
        "    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (\n",
        "                lr, reg, train_accuracy, val_accuracy))\n",
        "\n",
        "print('best validation accuracy achieved during cross-validation: %f' % best_val)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "id": "8a1f9db3",
      "metadata": {
        "test": "test",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 560
        },
        "id": "8a1f9db3",
        "outputId": "4325c9f5-98a9-4d43-f30c-969147fd23db"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[autoreload of cs231n.classifiers.softmax failed: Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/IPython/extensions/autoreload.py\", line 245, in check\n",
            "    superreload(m, reload, self.old_objects)\n",
            "  File \"/usr/local/lib/python3.10/dist-packages/IPython/extensions/autoreload.py\", line 394, in superreload\n",
            "    module = reload(module)\n",
            "  File \"/usr/lib/python3.10/imp.py\", line 315, in reload\n",
            "    return importlib.reload(module)\n",
            "  File \"/usr/lib/python3.10/importlib/__init__.py\", line 169, in reload\n",
            "    _bootstrap._exec(spec, module)\n",
            "  File \"<frozen importlib._bootstrap>\", line 619, in _exec\n",
            "  File \"<frozen importlib._bootstrap_external>\", line 879, in exec_module\n",
            "  File \"<frozen importlib._bootstrap_external>\", line 1017, in get_code\n",
            "  File \"<frozen importlib._bootstrap_external>\", line 947, in source_to_code\n",
            "  File \"<frozen importlib._bootstrap>\", line 241, in _call_with_frames_removed\n",
            "  File \"/content/drive/My Drive/cs231n/assignment1/cs231n/classifiers/softmax.py\", line 48\n",
            "    dW /= num_train\n",
            "                   ^\n",
            "IndentationError: unindent does not match any outer indentation level\n",
            "]\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "AttributeError",
          "evalue": "'NoneType' object has no attribute 'predict'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-11-f4abb9fb2033>\u001b[0m in \u001b[0;36m<cell line: 3>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# evaluate on test set\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;31m# Evaluate the best softmax on test set\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0my_test_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbest_softmax\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_test\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0mtest_accuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_test\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0my_test_pred\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'softmax on raw pixels final test set accuracy: %f'\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mtest_accuracy\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mAttributeError\u001b[0m: 'NoneType' object has no attribute 'predict'"
          ]
        }
      ],
      "source": [
        "# evaluate on test set\n",
        "# Evaluate the best softmax on test set\n",
        "y_test_pred = best_softmax.predict(X_test)\n",
        "test_accuracy = np.mean(y_test == y_test_pred)\n",
        "print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "86b9b65c",
      "metadata": {
        "tags": [
          "pdf-inline"
        ],
        "id": "86b9b65c"
      },
      "source": [
        "**Inline Question 2** - *True or False*\n",
        "\n",
        "Suppose the overall training loss is defined as the sum of the per-datapoint loss over all training examples. It is possible to add a new datapoint to a training set that would leave the SVM loss unchanged, but this is not the case with the Softmax classifier loss.\n",
        "\n",
        "$\\color{blue}{\\textit Your Answer:}$\n",
        "\n",
        "\n",
        "$\\color{blue}{\\textit Your Explanation:}$\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "009f08b9",
      "metadata": {
        "id": "009f08b9"
      },
      "outputs": [],
      "source": [
        "# Visualize the learned weights for each class\n",
        "w = best_softmax.W[:-1,:] # strip out the bias\n",
        "w = w.reshape(32, 32, 3, 10)\n",
        "\n",
        "w_min, w_max = np.min(w), np.max(w)\n",
        "\n",
        "classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']\n",
        "for i in range(10):\n",
        "    plt.subplot(2, 5, i + 1)\n",
        "\n",
        "    # Rescale the weights to be between 0 and 255\n",
        "    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)\n",
        "    plt.imshow(wimg.astype('uint8'))\n",
        "    plt.axis('off')\n",
        "    plt.title(classes[i])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "3569b36b",
      "metadata": {
        "id": "3569b36b"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "language_info": {
      "name": "python"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}