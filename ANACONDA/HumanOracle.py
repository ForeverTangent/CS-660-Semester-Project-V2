# First we init stuff.
# Load the Basic Python Libraries
import os
import csv
import PIL
import pickle
import random
import datetime
import copy

# Load my Data Management Module
import CS660DataManagement as csDM

# load numpy
import numpy as np

# Load Keras Stuff
import keras
import keras.backend as K
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

K.set_image_data_format('channels_last')

# Other.  Mostly Graphic stuff for displaying Data in and out of Jupyter.
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pydot
import graphviz
from IPython.display import SVG

# Not using Quiver Yet.
# from quiver_engine import server

# Get Processed Data Directory.
processedDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'PROCESSED' )
combinedDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED' )
pickleDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'PICKLES' )
modelsDirectory = os.path.join( os.getcwd(), os.pardir, 'MODELS' )
modelsStructsDirectory = os.path.join( os.getcwd(), os.pardir, 'MODELS_STRUCTS' )
weightsDirectory = os.path.join( os.getcwd(), os.pardir, 'WEIGHTS' )
resultsDirectory = os.path.join( os.getcwd(), os.pardir, 'RESULTS' )

testImageColorFile = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED', 'ICOLOR', 'TEST.png' )
testImageDepthFile = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED', 'IDEPTH', 'TEST.png' )
testCSVFile = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED', 'PCLOUD', 'TEST.csv' )

tensorLogDataDir = os.path.join( os.getcwd(), os.pardir, 'TENSOR_LOGS' )

imageDirs = ['ICOLOR', 'IDEPTH']
csvDirs = ['PCLOUD']

allDataFlavors = imageDirs + csvDirs

#Load main data file.
searCSVInfoFile = os.path.join( combinedDataDir, 'SEAR_DC_INFO.csv' )

csDM.CS660DataManagementCheck()





# For Solo running Just in case.
# (Have no idea why that would ever happen)
if __name__ == "__main__":
    """
    """
    print("")
    print("If I am running you are doing everything wrong.")
    print("__main__ Done.")