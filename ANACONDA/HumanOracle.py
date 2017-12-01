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


def getElementsAddedToTrainingSetFileName():
    """
    """
    theFileName = 'JUPYTER_ADDED_ELEMENTS_RECORD' + csDM.getDateTimeAsString() + '.txt'
    return theFileName


def getTrainingPredictionResultsFileName():
    """
    """
    theFileName = 'JUPYTER_TRAIN_PREDICTION_RESULTS' + csDM.getDateTimeAsString() + '.txt'
    return theFileName


def recordElementsAddedToTrainingSet(currentFileName, nameOfClassAdded, elementsList):
    """
    Record the Training and Predictions Results.
    """
    trainingPredictionResultsPath = os.path.join(resultsDirectory, currentFileName)

    classAndElements = []
    classAndElements.append(nameOfClassAdded)
    classAndElements = classAndElements + elementsList

    print(classAndElements)

    with open(trainingPredictionResultsPath, 'a', newline='\n') as csvfile:
        csvWriter = csv.writer(
            csvfile,
            delimiter=','
        )
        csvWriter.writerow(classAndElements)


def recordTrainingPredictionResults(currentFileName, scoringResults, lowestScoringClassName):
    """
    Record the Training and Predictions Results.
    """
    trainingPredictionResultsPath = os.path.join(resultsDirectory, currentFileName)
    allScoringInfo = scoringResults['SCORE'] + scoringResults['SCORELIST']
    allScoringInfo.append(lowestScoringClassName)

    print(allScoringInfo)

    if (not os.path.exists(trainingPredictionResultsPath)):
        headers = ['TestLoss', 'TestAccuracy', 'NA', 'UP', 'DOWN', 'HOLE']
        with open(trainingPredictionResultsPath, 'w', newline='\n') as csvfile:
            csvWriter = csv.writer(
                csvfile,
                delimiter=','
            )
            csvWriter.writerow(headers)
            csvWriter.writerow(allScoringInfo)

    else:
        print(allScoringInfo)
        with open(trainingPredictionResultsPath, 'a', newline='\n') as csvfile:
            csvWriter = csv.writer(
                csvfile,
                delimiter=','
            )
            csvWriter.writerow(allScoringInfo)


def reportOracle(status):
    """
    """
    oracleReportPath = os.path.join(resultsDirectory, 'JUPYTER_ORACLE_REPORT.txt')
    fileToWrite = open(oracleReportPath, 'a')
    fileToWrite.write(status + '\n')
    fileToWrite.close()


def getLowestScoringCategory(scoringListAsPecents):
    """
    Get the index of the lowest scoring category.
    """

    lowestCategoryPercent = 1.0
    currentLowestIndex = 0

    for index in range(len(scoringListAsPecents)):
        currentCategoryPercent = scoringListAsPecents[index]
        if (currentCategoryPercent < lowestCategoryPercent):
            currentLowestIndex = index
            lowestCategoryPercent = currentCategoryPercent

    return currentLowestIndex


def addNewSamplesToTrainingSet(trainingList, newSamples):
    """
    Merge the new samples with the Training set.
    """
    return trainingList + newSamples


def getSamplesFromAClass(classType, trainingList, allTestListsCombined, numberOfSamples=20):
    """
    Get multiple samples from a class.
    """
    newSamples = []

    for index in range(numberOfSamples):
        newSample = getASampleOfClass(classType, trainingList, allTestListsCombined)
        newSamples.append(newSample)

    return newSamples


def getASampleOfClass(classType, trainingList, allTestListsCombined):
    """
    Get a Sample from a Specific class that isn't being trained on yet.
    """

    classTypeToRetreive = ''

    if (classType == 'UP'):
        classTypeToRetreive = 'upList'
    elif (classType == 'DOWN'):
        classTypeToRetreive = 'downList'
    elif (classType == 'NA'):
        classTypeToRetreive = 'naList'
    elif (classType == 'HOLE'):
        classTypeToRetreive = 'holeList'

    random.seed(datetime.datetime.utcnow())

    dataClassList = csDM.getDictOfClassLists()

    trainingListAsSet = set(trainingList)
    allTestListsCombinedAsSet = set(allTestListsCombined)

    allElementsOfClass = dataClassList[classTypeToRetreive]

    selectedSample = random.choice(allElementsOfClass)

    while ((selectedSample in trainingListAsSet) or (selectedSample in allTestListsCombinedAsSet)):
        selectedSample = random.choice(allElementsOfClass)

    return selectedSample


def getAllTestLists():
    """
    Gets all the test lists.
    Returns:
        A Dict of the test lists.
    """
    allTestLists = []
    allTestListCombined = []

    for index in range(5):
        allTestLists.append(csDM.getTestList(index))
        allTestListCombined = allTestListCombined + csDM.getTestList(index)

    return {'TestLists': allTestLists, 'CombinedTestLists': allTestListCombined}






# For Solo running Just in case.
# (Have no idea why that would ever happen)
if __name__ == "__main__":
    """
    """
    print("__main__ Done.")