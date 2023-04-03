# from torch import nn as nn
# from fn_try_gpu import *

import tensorflow as tf
import keras
import numpy as np


def build_model2_1():
    model = keras.Sequential([
        keras.layers.Dense(30, activation='relu'),
        keras.layers.Dense(12)
    ])
    return model
