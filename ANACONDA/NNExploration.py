#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 01:10:35 2017

I have to admit, a lot of this file is sort of hackey [not exactly elegant].

Don't hold it against me it got the job done.


@author: staque
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


"""
What needs to happen:
Generate 3 sets of Training Test Elements
Pickles that Data
Generate Arrays of Data Flavors
Pickle the Arrays
Repeat:
    Generate NN Model
    Save Model Description
    Run 3 Data flavors on NN Model.
        While running models Save TensorBoards
        Save results
        Save Weights.

"""

#Globals
# Get Processed Data Directory.
combinedDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED' )
modelsDirectory = os.path.join( os.getcwd(), os.pardir, 'MODELS' )
modelsStructsDirectory = os.path.join( os.getcwd(), os.pardir, 'MODELS_STRUCTS' )
pickleDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'PICKLES' )
tensorLogDataDir = os.path.join( os.getcwd(), os.pardir, 'TENSOR_LOGS' )
weightsDirectory = os.path.join( os.getcwd(), os.pardir, 'WEIGHTS' )

nnResultDirectory = os.path.join( os.getcwd(), os.pardir, 'RESULTS' )
nnTestResultPath = os.path.join( nnResultDirectory, 'NNTestResults.txt' )
nnTestResultPath2 = os.path.join( nnResultDirectory, 'NNTestResults2.txt' )
nnTestResultPath3 = os.path.join( nnResultDirectory, 'NNTestResults3-32To64Nodes-1Layer.txt' )

testImageColorFile = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED', 'ICOLOR', 'TEST.png' )
testImageDepthFile = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED', 'IDEPTH', 'TEST.png' )
testCSVFile = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED', 'PCLOUD', 'TEST.csv' )

imageDirs = ['ICOLOR', 'IDEPTH']
csvDirs = ['PCLOUD']

allDataFlavors = imageDirs + csvDirs

#Load main data file.
# searCSVInfoFile = os.path.join( combinedDataDir, 'SEAR_DC_INFO.csv' )


class NNExploration():
    """
    Lets explore the space.
    """
    theModel = None

    def __init__(self):
        """
        Constructor!

        Let Init variables Here

        """
        print("NNExploration Init()")


    def buildModel(self, numOfNodes=48, numOfLayers=1):
        """
        Builds the basic model.
        Returns:
            A Keras NN Model

        """
        # input image dimensions
        img_rows, img_cols = 480, 640
        input_shape = (img_rows, img_cols, 1)
        num_classes = 4

        print("Building Model with ", numOfNodes, " nodes and ", numOfLayers, " layers.")

        self.theModel = Sequential()

        self.theModel.add(
            Conv2D(5,
                   kernel_size=(5, 5),
                   strides=3,
                   activation='relu',
                   input_shape=input_shape
                   )
        )
        self.theModel.add(
            MaxPooling2D(
                pool_size=(2, 2)
            )
        )

        self.theModel.add(
            Conv2D(
                10,
                kernel_size=(3, 3),
                strides=2,
                activation='relu')
        )
        self.theModel.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=2
            )
        )

        self.theModel.add(Flatten())

        for index in range( numOfLayers ):
            self.theModel.add(Dense(numOfNodes))
            self.theModel.add(BatchNormalization())
            self.theModel.add(Activation('relu'))
            self.theModel.add(Dropout(0.25))

        self.theModel.add(Dense(num_classes, activation='softmax'))

        self.theModel.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['categorical_accuracy']
        )

        self.theModel.summary()

    # server.launch(model)

    def evaluateModel(self, x_test, y_test, num_classes):
        """
        """

        y_test_as_category = to_categorical(y_test, num_classes)

        score = self.theModel.evaluate(x_test, y_test_as_category, verbose=0)
        print('General > Test loss: ', score[0], 'Test accuracy: ', score[1] )

        results = self.theModel.predict_classes(x_test, verbose=1)

        for index in range(len(x_test)):
            if( results[index] == y_test[index] ):
                print("Mathc")
            else:
                print("No Match")


        return score


    def reshape( self, theTrainingSet ):
        """
        This is to add the 'Channels' Dimension for Keras.
        """
        getShape = theTrainingSet.shape
        newShape = getShape + (1,)
        return theTrainingSet.reshape(newShape)


    def saveModelStructure(self, model, modelStructureName ):
        modelStructsFilePath = os.path.join(modelsStructsDirectory, modelStructureName )
        plot_model(self.theModel, to_file=modelStructsFilePath)


    def saveModel(self, model, modelName):
        """
        Saves the Model as JSON
        Args:
            model: the Keras NN Model
            modelName: the Name

        Returns:

        """
        modelFilePath = os.path.join( modelsDirectory, modelName + '.json' )
        model_json = self.theModel.to_json()
        with open( modelFilePath, 'w') as json_file:
            json_file.write(model_json)


    def saveModelWeights(self, model, weightName ):
        """
        Saved the Model Weights
        Args:
            weights: The Weights
            weightName: Weight Names

        Returns:

        """
        weightsFilePath = os.path.join( weightsDirectory, weightName + '.h5' )
        self.theModel.save_weights(weightsFilePath)


    def trainModel(self, trainingName, x_train, y_train, x_test, y_test, num_classes ):
        """
        Trains the model via given data.

        Args:
            trainingName: A name of this train [mainly to track in TensorBoard
            x_train: The X Set for Trainings
            y_train: The Y set for Testing
            x_test:  The X Set for Training/Verification
            y_test:  The Y Set for Testing/Verification

        Returns:

        """

        # Reshape the X sets.
        # Mainly for this project.because Keras/Tensor thinks in Channels.
        # And since we are using Greyscale data, we really don't have a channel.
        # So we have to 'fake' a channel
        #
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        # Convert class vectors to binary class matrices
        y_train_as_category = to_categorical(y_train, num_classes)
        y_test_as_category = to_categorical(y_test, num_classes)

        logFilePath = os.path.join( tensorLogDataDir, trainingName )

        TBoardCallback = keras.callbacks.TensorBoard(
            log_dir=logFilePath,
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )

        self.theModel.fit(x_train,
                  y_train_as_category,
                  batch_size=16,
                  epochs=32,
                  verbose=1,
                  validation_data=(x_test, y_test_as_category),
                  callbacks=[TBoardCallback]
                  )


def loadModel( modelName ):
    """
    Loads the Model.
    """
    theModel = load_model( os.path.join( modelsDirectory, modelName) )
    print('MODEL Loaded.')
    return theModel


def saveModelEverything(theModel, modelName):
    """
    Saved Everything in regards to the model.
    """
    saveModelStructure(theModel, modelName)
    saveModel(theModel, modelName)
    saveModelJSON(theModel, modelName)
    saveModelWeights(theModel, modelName)

    print("Model Everything Saved")


def saveModelStructure(theModel, modelStructureName):
    """
    Saves an image of the Model Structure.
    """
    modelStructsFilePath = os.path.join(modelsStructsDirectory, modelStructureName)
    plot_model(theModel, to_file=modelStructsFilePath)


def saveModelJSON(model, modelName):
    """
    Saves the Model as JSON
    Args:
        model: the Keras NN Model
        modelName: the Name

    Returns:

    """
    modelFilePath = os.path.join(modelsDirectory, modelName + '.json')
    model_json = theModel.to_json()
    with open(modelFilePath, 'w') as json_file:
        json_file.write(model_json)


def saveModel(model, modelName):
    """
    Save the model, in Keras [h5] format.
    """
    theModel.save(os.path.join(modelsDirectory, modelName))


def saveModelWeights(theModel, modelName):
    """
    Saved the Model Weights
    Args:
        weights: The Weights
        weightName: Weight Names

    Returns:

    """
    weightsFilePath = os.path.join(weightsDirectory, modelName + '.h5')
    theModel.save_weights(weightsFilePath)


def recordNNTestResults( somethingToWrite ):

    print(recordNNTestResults)

    if( not os.path.exists( nnTestResultPath ) ):
        print("No file exists")
        with open( nnTestResultPath, 'w') as theFile:
            theFile.write("NNResults\n")
            theFile.write("---------\n")
            theFile.write(somethingToWrite + '\n')
    else:
        print("file exists")
        with open( nnTestResultPath, 'a') as theFile:
            theFile.write( somethingToWrite + '\n' )
            theFile.write( "" )



def testAllNNTypes():
    """
    A quick made function to figure out how many nodes and Layers we needed.

    Returns:

    """
    posibleNodes = [ 32, 64, 128, 256 ]
    posibleLayers = [1, 2, 3, 4]

    img_rows, img_cols = 480, 640
    num_classes = 4

    for index in range(1):
        for numOfNodes in posibleNodes:
            for numOfLayers in posibleLayers:
                for flavor in allDataFlavors:
                    for attempt in range(3):
                        theString = ( index, flavor, numOfNodes, numOfLayers, attempt )
                        theString = 'TRSET_' + \
                                    str(index) + '_' + \
                                    flavor + '_' + \
                                    str(numOfNodes) + '_' + \
                                    str(numOfLayers) + '_' + \
                                    str(attempt)

                        print( theString )

                        nnExplorer = NNExploration()

                        theModel = nnExplorer.buildModel( numOfNodes, numOfLayers)

                        x_train, y_train, x_test, y_test = csDM.load_dataset( index, flavor )

                        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

                        # convert class vectors to binary class matrices
                        y_train_as_category = to_categorical(y_train, num_classes)
                        y_test_as_category = to_categorical(y_test, num_classes)

                        tensorPath = os.path.join( tensorLogDataDir, theString)

                        TBoardCallback = keras.callbacks.TensorBoard(
                            log_dir=tensorPath,
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True
                        )

                        theModel.fit(x_train,
                                     y_train_as_category,
                                     batch_size=16,
                                     epochs=32,
                                     verbose=1,
                                     validation_data=(x_test, y_test_as_category),
                                     callbacks=[TBoardCallback]
                                     )

                        score = theModel.evaluate(x_test, y_test_as_category, verbose=0)

                        testResultsInfo = []
                        testResultsInfo.append( 'TRAINING_' + str(index) )
                        testResultsInfo.append( numOfNodes )
                        testResultsInfo.append( numOfLayers )
                        testResultsInfo.append( flavor )
                        testResultsInfo.append( attempt )
                        testResultsInfo.append( score[0] )
                        testResultsInfo.append( score[1] )

                        with open( nnTestResultPath, 'a', newline='\n') as csvfile:
                            csvWriter = csv.writer(csvfile, delimiter=',')
                            csvWriter.writerow(testResultsInfo)

                with open(nnTestResultPath, 'a') as theFile:
                    theFile.write("\n")




def testAllNNTypes2():
    """
    Another  quick made function to figure out how many nodes and Layers we needed.

    Returns:

    """
    posibleNodes = [ 4, 8, 16 ]
    posibleLayers = [1]

    img_rows, img_cols = 480, 640
    num_classes = 4

    for index in range(1):
        for numOfNodes in posibleNodes:
            for numOfLayers in posibleLayers:
                for flavor in allDataFlavors:
                    for attempt in range(3):
                        theString = ( index, flavor, numOfNodes, numOfLayers, attempt )
                        theString = 'TRSET_' + \
                                    str(index) + '_' + \
                                    flavor + '_' + \
                                    str(numOfNodes) + '_' + \
                                    str(numOfLayers) + '_' + \
                                    str(attempt)

                        print( theString )

                        nnExplorer = NNExploration()

                        theModel = nnExplorer.buildModel( numOfNodes, numOfLayers)

                        x_train, y_train, x_test, y_test = csDM.load_dataset( index, flavor )

                        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

                        # convert class vectors to binary class matrices
                        y_train_as_category = to_categorical(y_train, num_classes)
                        y_test_as_category = to_categorical(y_test, num_classes)

                        tensorPath = os.path.join( tensorLogDataDir, theString)

                        TBoardCallback = keras.callbacks.TensorBoard(
                            log_dir=tensorPath,
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True
                        )

                        theModel.fit(x_train,
                                     y_train_as_category,
                                     batch_size=32,
                                     epochs=16,
                                     verbose=1,
                                     validation_data=(x_test, y_test_as_category),
                                     callbacks=[TBoardCallback]
                                     )

                        score = theModel.evaluate(x_test, y_test_as_category, verbose=0)

                        testResultsInfo = []
                        testResultsInfo.append( 'TRAINING_' + str(index) )
                        testResultsInfo.append( numOfNodes )
                        testResultsInfo.append( numOfLayers )
                        testResultsInfo.append( flavor )
                        testResultsInfo.append( attempt )
                        testResultsInfo.append( score[0] )
                        testResultsInfo.append( score[1] )

                        with open( nnTestResultPath, 'a', newline='\n') as csvfile:
                            csvWriter = csv.writer(csvfile, delimiter=',')
                            csvWriter.writerow(testResultsInfo)

                with open(nnTestResultPath, 'a') as theFile:
                    theFile.write("\n")




def testAllNNTypes3():
    """
    Another  quick made function to figure out how many nodes and Layers we needed.

    Returns:

    """
    posibleNodes = [32, 48, 52, 56, 64]
    posibleLayers = [1]

    img_rows, img_cols = 480, 640
    num_classes = 4

    for index in range(1):
        for numOfNodes in posibleNodes:
            for numOfLayers in posibleLayers:
                for flavor in allDataFlavors:
                    for attempt in range(3):
                        theString = (index, flavor, numOfNodes, numOfLayers, attempt)
                        theString = 'TRSET_' + \
                                    str(index) + '_' + \
                                    flavor + '_' + \
                                    str(numOfNodes) + '_' + \
                                    str(numOfLayers) + '_' + \
                                    str(attempt)

                        print(theString)

                        nnExplorer = NNExploration()

                        theModel = nnExplorer.buildModel(numOfNodes, numOfLayers)

                        x_train, y_train, x_test, y_test = csDM.load_dataset(index, flavor)

                        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

                        # convert class vectors to binary class matrices
                        y_train_as_category = to_categorical(y_train, num_classes)
                        y_test_as_category = to_categorical(y_test, num_classes)

                        tensorPath = os.path.join(tensorLogDataDir, theString)

                        TBoardCallback = keras.callbacks.TensorBoard(
                            log_dir=tensorPath,
                            histogram_freq=0,
                            write_graph=True,
                            write_images=True
                        )

                        theModel.fit(x_train,
                                     y_train_as_category,
                                     batch_size=16,
                                     epochs=24,
                                     verbose=1,
                                     validation_data=(x_test, y_test_as_category),
                                     callbacks=[TBoardCallback]
                                     )

                        score = theModel.evaluate(x_test, y_test_as_category, verbose=0)

                        testResultsInfo = []
                        testResultsInfo.append('TRAINING_' + str(index))
                        testResultsInfo.append(numOfNodes)
                        testResultsInfo.append(numOfLayers)
                        testResultsInfo.append(flavor)
                        testResultsInfo.append(attempt)
                        testResultsInfo.append(score[0])
                        testResultsInfo.append(score[1])

                        with open(nnTestResultPath3, 'a', newline='\n') as csvfile:
                            csvWriter = csv.writer(csvfile, delimiter=',')
                            csvWriter.writerow(testResultsInfo)

                with open(nnTestResultPath3, 'a') as theFile:
                    theFile.write("\n")



# For Solo running Just in case.
# (Have no idea why that would ever happen)
if __name__ == "__main__":
    """
    """
    csDM.CS660DataManagementCheck()
    csDM.createAllBasicData()

    nnExplorer = NNExploration()

    # testAllNNTypes3()

    print("__main__ Done.")