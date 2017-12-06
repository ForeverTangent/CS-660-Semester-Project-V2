"""
Contains our 'Human Oracle functions.
"""

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
import NNExploration as nnEx

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
resultsDirectory = os.path.join(os.getcwd(), os.pardir, 'RESULTS')


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

numberOfEachClassForAnalysis = {'UP': 456, 'DOWN': 302, 'NA': 308, 'HOLE': 336}



def getElementsAddedToTrainingListFileName( dataFlavor, onDataSetIndex, usingTestListIndex, attempt=0 ):
    """
    """
    theFileName = 'PYCHARM_ADDED_ELEMENTS_RECORD_' + \
                  dataFlavor + '_' + \
                  'DL_' + str(onDataSetIndex) + '_' + \
                  'TL_' + str(usingTestListIndex) + '_' +  \
                  'AT_' + str(attempt) + '.txt'

    return theFileName


def getTrainingPredictionResultsFileName( dataFlavor, onDataSetIndex, usingTestListIndex, attempt=0 ):
    """
    """
    theFileName = 'PyCHARM_TRAIN_PREDICTION_RESULTS_' + \
                  dataFlavor + '_' + \
                  'DL_' + str(onDataSetIndex) + '_' + \
                  'TL_' + str(usingTestListIndex) + '_' + \
                  'AT_' + str(attempt) + '.txt'

    return theFileName


def recordElementsAddedToTrainingList(currentFileName, nameOfClassAdded, elementsList):
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
    oracleReportPath = os.path.join(resultsDirectory, 'PyCHARM_ORACLE_REPORT.txt')
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


def addNewSamplesToTrainingList(trainingList, newSamples):
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


def getRandomSamples(trainingList, allTestListsCombined, numberOfSamples=20):
    """
    Get multiple samples from a class.
    """
    newSamples = []

    for index in range(numberOfSamples):
        newSample = getARandomSample(trainingList, allTestListsCombined)
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

    trainingListAsList = set(trainingList)
    allTestListsCombinedAsList = set(allTestListsCombined)

    allElementsOfClass = dataClassList[classTypeToRetreive]

    selectedSample = random.choice(allElementsOfClass)

    while ((selectedSample in trainingListAsList) or (selectedSample in allTestListsCombinedAsList)):
        selectedSample = random.choice(allElementsOfClass)

    return selectedSample


def getARandomSample(trainingList, allTestListsCombined):
    """
    Get a Sample from a Specific class that isn't being trained on yet.
    """

    random.seed(datetime.datetime.utcnow())

    allSamplesList = csDM.getAllSamples()

    trainingListAsList = set(trainingList)
    allTestListsCombinedAsList = set(allTestListsCombined)

    selectedSample = random.choice(allSamplesList)

    while ((selectedSample in trainingListAsList) or (selectedSample in allTestListsCombinedAsList)):
        selectedSample = random.choice(allSamplesList)

    return selectedSample


def getANormalTrainValidationTestSplit( allSamplesList ):
    """
    Get a Sample from a Specific class that isn't being trained on yet.
    """

    random.seed(datetime.datetime.utcnow())

    #Shuffle 7 times.
    random.shuffle(allSamplesList)

    trainingList = allSamplesList[:1400]
    testLists = allSamplesList[1400:1800]

    return trainingList, testLists


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


def humanOracleFullExperiment():
    """
    Full Test Run of
    Returns:

    """

    # Randomize the Training and Test sets we use.
    # random.seed(datetime.datetime.utcnow())
    # aList = [0,1,2,3,4]
    # trainingListIndexes = random.sample( aList, 3 )
    # testingListIndexes = random.sample( aList, 3 )

    trainingListIndexes = [0, 1, 2, 3, 4]
    testingListIndexes = [0, 1, 2, 3, 4]


    # Every Flavor, 3 Attempts, Training, Test
    for flavor in allDataFlavors:
        for attempt in range(1):
            for trainingElement in trainingListIndexes:
                for testElement in testingListIndexes:
                    print( 'Runnning', flavor, 'TR:', trainingElement, 'TE', testElement, 'AT:', attempt )
                    # singleHumanOraclePass( flavor, trainingElement, testElement, 15, attempt )
                    singleHumanOraclePass(flavor, trainingElement, testElement, 15, attempt)


def noHumanOracleFullExperiment():
    """
    Full Test Run of
    Returns:

    """

    # Randomize the Training and Test sets we use.
    # random.seed(datetime.datetime.utcnow())
    # aList = [0,1,2,3,4]
    # trainingListIndexes = random.sample( aList, 3 )
    # testingListIndexes = random.sample( aList, 3 )

    trainingListIndexes = [0, 1, 2, 3, 4]
    testingListIndexes = [0, 1, 2, 3, 4]


    # Every Flavor, 3 Attempts, Training, Test
    for flavor in allDataFlavors:
        for attempt in range(1):
            for trainingElement in trainingListIndexes:
                for testElement in testingListIndexes:
                    print( 'Runnning', flavor, 'TR:', trainingElement, 'TE', testElement, 'AT:', attempt )
                    # singleHumanOraclePass( flavor, trainingElement, testElement, 15, attempt )
                    singleNoOraclePass(flavor, trainingElement, testElement, 15, attempt)



def classicExperiment():
    """
    Full Test Run of
    Returns:

    """

    # Randomize the Training and Test sets we use.
    # random.seed(datetime.datetime.utcnow())
    # aList = [0,1,2,3,4]
    # trainingListIndexes = random.sample( aList, 3 )
    # testingListIndexes = random.sample( aList, 3 )

    # Every Flavor, 3 Attempts, Training, Test
    for flavor in allDataFlavors:
        for attempt in range(1):
            print( 'Runnning', flavor, 'CLASSIC', 'AT:', attempt )
            classicNNTraining( flavor, attempt )




def singleHumanOraclePass(dataFlavor='IDEPTH', onTrainingListIndex=0, usingTestListIndex=2, numberOfAges=15, attempt=0):
    """
    Single Human Oracle function
    PARAMETERS:
        dataFlavor = String
        onTrainingListIndex = Int
        usingTestListIndex = Int
        numberOfAges = Int

    """

    print("Starting Human Oracle")

    # First get the file name we need to record data.
    trainingPredictionResultsFileName = getTrainingPredictionResultsFileName(dataFlavor, onTrainingListIndex, usingTestListIndex, attempt)
    elementsAddedToTrainingListFileName = getElementsAddedToTrainingListFileName(dataFlavor, onTrainingListIndex, usingTestListIndex, attempt)

    # Get Elements to train and test on
    X_train, Y_train, X_test, Y_test, X_trainList, Y_TestList = csDM.loadTrainingAndTestDatasetAndLists(str(onTrainingListIndex), dataFlavor)

    thisModelName = 'PyCHARMHumanOracleTraining_' + str(dataFlavor) + '_DS_' + str(onTrainingListIndex) + '_TS_' + str(usingTestListIndex)

    # This is to ensure our test set always stay independent.
    allTestListsCombined = getAllTestLists()

    # Get Independent List for Testing
    # Techincally the above X_test, Y_test, should be indepedent, according to the Keras documentation.
    # But I am using a second set just because.
    X_test_Z, Y_test_Z = csDM.loadTestOnlyDataset( str(usingTestListIndex), dataFlavor )

    # Build Model
    theModel = nnEx.buildModel()

    print("Training Model")

    # Train the Model
    nnEx.trainModel( thisModelName, theModel, X_train, Y_train, X_test, Y_test, 4)

    # Evaluate the Model [with indie data]
    scoringResults = nnEx.evaluateModel(theModel, X_test_Z, Y_test_Z, 4)

    # Now the real Fun starts.
    # Get Lowest scoring class.
    lowestScoringClassName = csDM.getClassFromNumeral(getLowestScoringCategory(scoringResults['SCORELIST']))

    # Record the Init Results of the first eval and the first lowest scoring category.
    recordTrainingPredictionResults(trainingPredictionResultsFileName, scoringResults, lowestScoringClassName)

    for index in range(numberOfAges):
        # Get new samples from the lowest scoring class.
        newSamples = getSamplesFromAClass(lowestScoringClassName, X_trainList, allTestListsCombined, 20)

        # Record What we added.
        recordElementsAddedToTrainingList(elementsAddedToTrainingListFileName, lowestScoringClassName, newSamples)

        X_trainList = X_trainList + newSamples

        # Add samples to training set
        # NOTE: We need to turn the 'newSamples' into NPArrays.  It is these new NP Arrays we add to
        # X_train, Y_train.  Out Original List of the what is in the training sets stay intact.

        # So first get the NPArrays of the new Samples
        dictOfLearningAndVerificationNPArrays = csDM.createNPArraysFor(newSamples, dataFlavor)

        # Then we add them to the training set.

        #         print(type(X_train))
        #         print(type(dictOfLearningAndVerificationNPArrays['LEARNING']))

        #         print( X_train.shape )
        #         print( dictOfLearningAndVerificationNPArrays['LEARNING'].shape )

        X_train = np.concatenate((X_train, dictOfLearningAndVerificationNPArrays['LEARNING']), axis=0)
        Y_train = np.concatenate((Y_train, dictOfLearningAndVerificationNPArrays['VERIFICATION']), axis=0)

        # And Then we train the model again.
        stringOfTheAge = thisModelName + '_AGE_' + str(index)
        nnEx.trainModel(stringOfTheAge, theModel, X_train, Y_train, X_test, Y_test, 4, 6)

        # Evaluate the Model [with indie data]
        scoringResults = nnEx.evaluateModel(theModel, X_test_Z, Y_test_Z, 4)

        # Now the real Fun starts.
        # Get Lowest scoring class.
        lowestScoringClassName = csDM.getClassFromNumeral(getLowestScoringCategory(scoringResults['SCORELIST']))

        # Record the Init Results of the first eval and the first lowest scoring category.
        recordTrainingPredictionResults(trainingPredictionResultsFileName, scoringResults, lowestScoringClassName)


    nnEx.saveModelEverything( theModel, thisModelName)



def singleNoOraclePass(dataFlavor='IDEPTH', onTrainingListIndex=0, usingTestListIndex=2, numberOfAges=15, attempt=0):
    """
    Single No Oracle Function. Use to compare to Oracle one.
    PARAMETERS:
        dataFlavor = String
        onTrainingListIndex = Int
        usingTestListIndex = Int
        numberOfAges = Int

    """

    print("Starting Human Oracle")

    # First get the file name we need to record data.
    trainingPredictionResultsFileName = getTrainingPredictionResultsFileName(dataFlavor, onTrainingListIndex, usingTestListIndex, attempt)
    elementsAddedToTrainingListFileName = getElementsAddedToTrainingListFileName(dataFlavor, onTrainingListIndex, usingTestListIndex, attempt)

    # Get Elements to train and test on
    X_train, Y_train, X_test, Y_test, X_trainList, Y_TestList = csDM.loadTrainingAndTestDatasetAndLists(str(onTrainingListIndex), dataFlavor)

    thisModelName = 'PyCHARMNoOracleTraining_' + str(dataFlavor) + '_DS_' + str(onTrainingListIndex) + '_TS_' + str(usingTestListIndex)

    # This is to ensure our test set always stay independent.
    allTestListsCombined = getAllTestLists()

    # Get Independent List for Testing
    # Techincally the above X_test, Y_test, should be indepedent, according to the Keras documentation.
    # But I am using a second set just because.
    X_test_Z, Y_test_Z = csDM.loadTestOnlyDataset(str(usingTestListIndex), dataFlavor)

    # Build Model
    theModel = nnEx.buildModel()

    # Train the Model
    nnEx.trainModel( thisModelName, theModel, X_train, Y_train, X_test, Y_test, 4)

    # Evaluate the Model [with indie data]
    scoringResults = nnEx.evaluateModel(theModel, X_test_Z, Y_test_Z, 4)

    # Now the real Fun starts.
    # Get Lowest scoring class.
    lowestScoringClassName = csDM.getClassFromNumeral(getLowestScoringCategory(scoringResults['SCORELIST']))

    # Record the Init Results of the first eval and the first lowest scoring category.
    recordTrainingPredictionResults(trainingPredictionResultsFileName, scoringResults, lowestScoringClassName)

    for index in range(numberOfAges):
        # Get new samples from the lowest scoring class.
        newSamples = getRandomSamples( X_trainList, allTestListsCombined, 20)

        X_trainList = X_trainList + newSamples

        # Add samples to training set
        # NOTE: We need to turn the 'newSamples' into NPArrays.  It is these new NP Arrays we add to
        # X_train, Y_train.  Out Original List of the what is in the training sets stay intact.

        # So first get the NPArrays of the new Samples
        dictOfLearningAndVerificationNPArrays = csDM.createNPArraysFor(newSamples, dataFlavor)

        # Then we add them to the training set.

        #         print(type(X_train))
        #         print(type(dictOfLearningAndVerificationNPArrays['LEARNING']))

        #         print( X_train.shape )
        #         print( dictOfLearningAndVerificationNPArrays['LEARNING'].shape )

        X_train = np.concatenate((X_train, dictOfLearningAndVerificationNPArrays['LEARNING']), axis=0)
        Y_train = np.concatenate((Y_train, dictOfLearningAndVerificationNPArrays['VERIFICATION']), axis=0)

        # And Then we train the model again.
        stringOfTheAge = thisModelName + '_AGE_' + str(index)
        nnEx.trainModel(stringOfTheAge, theModel, X_train, Y_train, X_test, Y_test, 4, 6)

        # Evaluate the Model [with indie data]
        scoringResults = nnEx.evaluateModel(theModel, X_test_Z, Y_test_Z, 4)

        # Now the real Fun starts.
        # Get Lowest scoring class.
        lowestScoringClassName = csDM.getClassFromNumeral(getLowestScoringCategory(scoringResults['SCORELIST']))

        # Record the Init Results of the first eval and the first lowest scoring category.
        recordTrainingPredictionResults(trainingPredictionResultsFileName, scoringResults, lowestScoringClassName)


    # nnEx.saveModelEverything( theModel, thisModelName)



def classicNNTraining(dataFlavor='IDEPTH', attempt=0):
    """
    Single No Oracle Function. Use to compare to Oracle one.
    PARAMETERS:
        dataFlavor = String
        onTrainingListIndex = Int
        usingTestListIndex = Int
        numberOfAges = Int

    """

    print("Starting Human Oracle")

    # First get the file name we need to record data.
    trainingPredictionResultsFileName = getTrainingPredictionResultsFileName(dataFlavor, 0, 0, attempt)

    # Get Elements to train and test on

    allSamplesList = csDM.getAllSamples()

    trainingList, testLists = getANormalTrainValidationTestSplit( allSamplesList )

    trainingListLV = csDM.createNPArraysFor(trainingList, dataFlavor)
    testListsLV = csDM.createNPArraysFor(testLists, dataFlavor)

    X_train = trainingListLV['LEARNING']
    Y_train = trainingListLV['VERIFICATION']
    X_test = testListsLV['LEARNING']
    Y_test = testListsLV['VERIFICATION']

    thisModelName = 'PyCHARMNoOracleTraining_' + str(dataFlavor) + '_DS_' + str(0) + '_TS_' + str(0) + '_AT_' + str(attempt)

    # Build Model
    theModel = nnEx.buildModel()

    # Train the Model
    nnEx.trainModel( thisModelName, theModel, X_train, Y_train, X_test, Y_test, 4, numOfEpochs=24)

    # Evaluate the Model [with indie data]
    scoringResults = nnEx.evaluateModel(theModel, X_test, Y_test, 4)

    print("")
    print( scoringResults['SCORE'] )
    print("")

    # Now the real Fun starts.
    # Get Lowest scoring class.
    lowestScoringClassName = csDM.getClassFromNumeral(getLowestScoringCategory(scoringResults['SCORELIST']))

    # Record the Init Results of the first eval and the first lowest scoring category.
    recordTrainingPredictionResults(trainingPredictionResultsFileName, scoringResults, lowestScoringClassName)





# For Solo running Just in case.
# (Have no idea why that would ever happen)
if __name__ == "__main__":
    """
    """

    # X_train, Y_train, X_test, Y_test = csDM.loadTrainingAndTestDataset('0', 'IDEPTH')
    #
    # X_test_Z, Y_test_Z = csDM.loadTestOnlyDataset('2', 'IDEPTH')

    csDM.reportStats()

    classicExperiment()

    print("__main__ Done.")