"""
The  main Interactive Learning Project.
"""

# Load the Basic Python Libraries
import os
import csv
import PIL
import pickle
import random
import datetime
import pydot

import keras

# load numpy
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import CS660DataManagement as csDM
import NNExploration

# Load Keras parts
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

from matplotlib.pyplot import imshow
from IPython.display import SVG
from quiver_engine import server

K.set_image_data_format('channels_last')




# For Solo running Just in case.
# (Have no idea why that would ever happen)
if __name__ == "__main__":
    """
    """
    print("GETTING HARDCORE NOW")
    csDM.CS660DataManagementCheck()
    csDM.createAllBasicData()

    print("__main__ Done.")
