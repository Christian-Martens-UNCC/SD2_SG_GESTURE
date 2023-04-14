# from torch import nn as nn
# from fn_try_gpu import *

import tensorflow as tf
# This line imports the TensorFlow library, a popular open-source library for machine learning
import keras
# This line imports the Keras library, which is a high-level API for building and training deep learning models. Keras can use TensorFlow as a backend
import numpy as np
# This line imports the NumPy library, which is used for numerical operations in Python, such as working with arrays and matrices

def build_model2_1():
# This line defines a function called build_model2_1() which will build and return a Keras Sequential model object
    model = keras.Sequential([
    # This line initializes a new Keras Sequential model object, which is a linear stack of layers. The list inside the parentheses specifies the layers of the model
        keras.layers.Dense(30, activation='relu'),
        # This line adds a fully connected layer to the model with 30 neurons and the Rectified Linear Unit (ReLU) activation function
        keras.layers.Dense(12)
        # This line adds another fully connected layer to the model with 12 neurons and no activation function specified, which will result in a linear activation function
    ])
    return model
    # This line returns the completed Keras Sequential model object from the build_model2_1() function
