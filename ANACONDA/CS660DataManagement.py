# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import os
import csv
import random
import datetime

import numpy as np

import PIL
import pickle


combinedDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED' )
pickleDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'PICKLES' )
searCSVInfoFile = os.path.join( combinedDataDir, 'SEAR_DC_INFO.csv' )

imageDirs = ['ICOLOR', 'IDEPTH']
csvDirs = ['PCLOUD']

allDataFlavors = imageDirs + csvDirs


def CS660DataManagementCheck():
    """
    Checks that this module loaded correctly.
    """
    print("CS660DataManagementCheck Imported")
    
    
def createLearningAndVerificationPickle( index, dataFlavor ):
    """
    Create the Pickles for the Learning and Verification NPArray of a dataFlavor
    
    PARAMETERS:
    index: Int of the case to make.
    dataFlavor: String of the data flavor
    """
    
    trainingLists = unpickleTrainingLists()
    testLists = unpickleTestLists()
    
    theArrays = getLearningAndVerificationArraysForTrainingOrTestList( trainingLists[index], dataFlavor )

    print( "Training INDEX: " + str(index) )
    print( theArrays['LEARNING'].shape )
    print( theArrays['VERIFICATION'].shape )

    pickleLearningSet( theArrays['LEARNING'], 'TRAINING', dataFlavor, index )
    pickleVerificationSet( theArrays['VERIFICATION'], 'TRAINING', dataFlavor, index )


    theArrays = getLearningAndVerificationArraysForTrainingOrTestList( testLists[index], dataFlavor )

    print( "Training INDEX: " + str(index) )
    print( theArrays['LEARNING'].shape )
    print( theArrays['VERIFICATION'].shape )

    pickleLearningSet( theArrays['LEARNING'], 'TEST', dataFlavor, index )
    pickleVerificationSet( theArrays['VERIFICATION'], 'TEST', dataFlavor, index )
    

def createLearningAndVerificationPickles():
    """
    Creates all the initial Learning and Verification Pickles.
    """
    for index in range(10):
        for dataFlavor in allDataFlavors:
            
            createLearningAndVerificationPickle( index, dataFlavor )


def generateKeysForTrainingList( trainingSize ):
    """
    Generates the Training set of a certain size.
    
    PARAMETERS:
    trainingSize: Int of size you need.
    """
    
    # Seed with the current time
    random.seed( datetime.datetime.utcnow() )

    # Get the Class lists
    dictOfLists = getCSVDictAsClassesLists()
    upList = dictOfLists['upList']
    downList = dictOfLists['downList']
    naList = dictOfLists['naList']
    holeList = dictOfLists['holeList']

    # Shuffle each list. 7 for good measure.
    for index in range(7):
        random.shuffle(upList)
        random.shuffle(downList)
        random.shuffle(naList)
        random.shuffle(holeList)
    
        
    # Get the training elements.  
    # The first 'trainingSize' samples in the shuffled list.
    # First we see how many of each class we need.
    subTrainingSize = int(trainingSize / 4)
    
    upListTrain = upList[:subTrainingSize]
    downListTrain = downList[:subTrainingSize]
    naListTrain = naList[:subTrainingSize]
    holeListTrain = holeList[:subTrainingSize]
    
    # Then we put all the classes together
    returnTrainingList = upListTrain + downListTrain + naListTrain + holeListTrain
    returnTrainingList.sort()
    
    return returnTrainingList


def generateKeysForTestingList( testSize ):
    """
    Generates the Test set of a certain size.
    
    PARAMETERS:
    testSize: Int of size you need.
    """
    
    # Seed with the current time
    random.seed( datetime.datetime.utcnow() )

    # Get the Class lists
    dictOfLists = getCSVDictAsClassesLists()
    upList = dictOfLists['upList']
    downList = dictOfLists['downList']
    naList = dictOfLists['naList']
    holeList = dictOfLists['holeList']

    # Shuffle each list. 7 for good measure.
    for index in range(7):
        random.shuffle(upList)
        random.shuffle(downList)
        random.shuffle(naList)
        random.shuffle(holeList)
        
    
    # Get the test elements.  The last 'testSize' samples in the shuffled list.
    # First we see how many of each class we need.
    subTestSize = int(testSize / 4)
    
    # Then we find the startIndex of the last 'testSize' samples in each class list.
    startIndex = len(upList)-subTestSize
    upListTest = upList[startIndex:]
    
    startIndex = len(downList)-subTestSize
    downListTest = downList[startIndex:]
    
    startIndex = len(naList)-subTestSize
    naListTest = naList[startIndex:]
    
    startIndex = len(holeList)-subTestSize
    holeListTest = holeList[startIndex:]  
    
    #Then we put all the classes together
    returnTestList = upListTest + downListTest + naListTest + holeListTest
    returnTestList.sort()
    
    return returnTestList


def generateKeysForTrainingAndTestingList( trainingSize, testSize ):
    """
    Generates a tuple of the Training and Test set of a certain size.
    
    PARAMETERS:
    trainingSize: Int of size you need.
    testSize: Int of size you need.
    """
    # Then we do a sanity check to make sure the Training and Test set are disjoint.
    returnTrainingList = generateKeysForTrainingList( trainingSize )
    returnTestList = generateKeysForTestingList( testSize )
    
    returnTrainingListAsSet = set( returnTrainingList )
    returnTestListAsSet = set( returnTestList )
    
    if returnTrainingListAsSet.isdisjoint(returnTestListAsSet):
        dictToReturn = {'TrainingList': returnTrainingList, 'TestList': returnTestList}
    else:
        dictToReturn =  { 'TrainingList': [], 'TestList': [] }
    
    return dictToReturn


def generateAndPickleTrainAndTestSetFor( index ):
    """
    Generation of Training and Test Set and pickling.
    
    PARAMETERS:
    index: Int of the index
    """
    theLists = generateKeysForTrainingAndTestingList(100,20)

    # Pick a NORM or FLOP for each TrainingTest element.
    theLists['TrainingList'] = pickNormOrFlopForTrainingTestingLists( theLists['TrainingList'] )
    theLists['TestList'] = pickNormOrFlopForTrainingTestingLists( theLists['TestList'] )

    pickleTrainAndTestList( theLists['TrainingList'], theLists['TestList'], index )


def generateAndPickleTrainAndTestSets():
    """
    Combo Generation of Training and Test Set and pickling.
    """
    # Generate 10 training lists.
    for index in range(10):
        generateAndPickleTrainAndTestSetFor( index )
    
    print("Sets Generated and Pickled")
    
    

def getDataDictOfTestList( theList ):
    """
    Gets a Dict of the CSV Info Data for the generated training/test set.

    PARAMETERS:
    theList = A list of the keys of the train/test set.
    """
    
    dataDict = getDictFromDataCSVFile()
    
    returnDict = {}
    
    for element in theList:
        
        asString = str(element)
        minusLast = asString[:-1]
        asIntAgain = int(minusLast)
        
        returnDict[asIntAgain] = dataDict[asIntAgain]
    
    return returnDict
    

def getCSVDictAsClassesLists():
    """
    Get the CSV Data Classes as lists.
    """
    upList = []
    downList = []
    naList = []
    holeList = []
    
    csvDataDict = getDictFromDataCSVFile()

    for key in getListOfDataCSVFileKeys():
        if (csvDataDict[key]['CLASS'] == 'UP'):
            upList.append( key )
        if (csvDataDict[key]['CLASS'] == 'DOWN'):
            downList.append( key )
        if (csvDataDict[key]['CLASS'] == 'NA'):
            naList.append( key )
        if (csvDataDict[key]['CLASS'] == 'HOLE'):
            holeList.append( key )
                
    return {'upList': upList, 'downList': downList, 'naList': naList, 'holeList' : holeList}


def getDepthCSVFileAsNPArray( theCSVFile ):
    """
    Turn Depth CSV data File into as Numpy Array
    
    PARAMETER:
    Name of CSV file.
    
    RETURNS:
    Numpy Array
    """
    CSVDataList = []
    
    with open(theCSVFile) as csvfile:
        CSVReader = csv.reader(csvfile)
        for row in CSVReader:
            CSVDataList = CSVDataList + row

    CSVDataFloatList = []
    
    for element in CSVDataList:
        CSVDataFloatList.append( float(element) )
 
    return np.array(CSVDataFloatList).reshape(480, 640)/10000.0




def getDictFromDataCSVFile():
    """
    Reads the Data CSV File.
    
    Returns:
    A Dictionary
    """
    dataCSVFileDict = {}

    with open(searCSVInfoFile, newline='') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            data =  {'ANGLE': float(row[1]), 'CLASS': row[2], 'TYPE': row[3]}
            dataCSVFileDict[ int(row[0]) ] = data

    return dataCSVFileDict


    
def getImageFileAsNPArray( theImageFile ):
    """
    Turn a PNG image as Numpy Array
    
    PARAMETER:
    image: An image
    
    RETURNS:
    Numpy Array
    """
    # The 'L' allow convertion to grayscale.
    theImage = PIL.Image.open(theImageFile).convert("L")
    return np.array(theImage)/255.0


def getListOfDataCSVFileKeys():
    """
    Get the keys of the CSV Data File.
    
    Returns:
    A List
    """

    return list( getDictFromDataCSVFile().keys() )



def getLearningSetAsNPArray( theTrainingList, targetDataDirectory, fileExtention ):
    """
    Gets the list of training elements as an NP Array
    """
    
    print( targetDataDirectory )
    
    trainingDataNPArray = None
    
    trainingArray = np.empty([len(theTrainingList), 480, 640], dtype = float)
    
    for element in theTrainingList:
        fullFilePath = os.path.join( targetDataDirectory, ( str(element) + '.' + fileExtention ) )
        
        if (fileExtention == 'png'):
            trainingDataNPArray = getImageFileAsNPArray( fullFilePath )
        elif (fileExtention == 'csv'):
            trainingDataNPArray = getDepthCSVFileAsNPArray(fullFilePath)
        
        trIndex = theTrainingList.index(element)
        trainingArray[trIndex] = trainingDataNPArray
        
    return trainingArray


def getLearningAndVerificationArraysForTrainingOrTestList( theList, dataFlavor ):
    """
    Get the Learning and Verification NPArrays
    
    PARAMETERS:
    typeOfList: String, 'TRAIN' or 'TEST"
    IDNum: Int of ID
    """
    
    print( dataFlavor in allDataFlavors )
    
    
    if( dataFlavor in allDataFlavors):        
        theArrays = getNPArraysFor( theList, dataFlavor )
        return theArrays
    
    else:
        return None
    


def getNPArraysFor( theList, dataFlavor ):
    """
    Gets the NPArrays for a given list of Elements.
    
    PARAMETERS:
        theList: A List of element Keys
        dataFlavor: String of What type of Data we are looking at
            ['ICOLOR', 'IDEPTH', 'PCLOUD']
    """
    
    dataFlavorInfo = getSubDirectoryAndEXTForDataFlavor( dataFlavor )
    
    subDir = dataFlavorInfo['SubDir']
    extention = dataFlavorInfo['EXT']
    
    # Get the Learning NPArray
    returnLearningNPArray = getLearningSetAsNPArray( 
        theList, 
        subDir, 
        extention 
    )
    
    returnVerificationNPArray = getVerificationSetNPArray( theList )

    return { 'LEARNING' : returnLearningNPArray, 'VERIFICATION' : returnVerificationNPArray }


def getNumeralValueOf( theClass ):
    
    if (theClass=="NA"):
        return 0
    elif (theClass=="UP"):
        return 1
    elif (theClass=="DOWN"):
        return 2
    elif (theClass=="HOLE"):
        return 3
    else:
        return None


def getVerificationSetNPArray( theList ):
    """
    Gets the list of testing elements as an NP Array
    """
    
    theDictOfTestElements = getDataDictOfTestList( theList )
    
    returnNPArray = np.empty([len( theList ),1], dtype = float) 
    
    for element in theList:
        
        theIndexOfElm = theList.index(element)
        
        asString = str(element)
        asStringMinusOne = asString[:-1]
        asInt = int(asStringMinusOne)
        
        elementData = theDictOfTestElements[asInt]
        classData = elementData['CLASS']
        classAsInt = getNumeralValueOf(classData)
        
        returnNPArray[theIndexOfElm] = classAsInt
        
    return returnNPArray



def getSubDirectoryAndEXTForDataFlavor( dataFlavor ):
    """
    Gets the SubDirection and Extention for the Data we are analysing.
    
    PARAMETERS:
    dataFlavor: String of the dataFlavor -> 'ICLOUD','IDEPTH,'PCLOUD'
    
    RETURNS: A Dict
    """
    subCombinedDataDir = None
    fileExtention = None
    
    if (dataFlavor in imageDirs):
        subCombinedDataDir = os.path.join( combinedDataDir, dataFlavor)
        fileExtention = 'png'
        pass
    elif (dataFlavor in csvDirs):
        subCombinedDataDir = os.path.join( combinedDataDir, dataFlavor)
        fileExtention = 'csv'
        pass
    else:
        print("Data Flavor not specified. Returning.")
        return None
    
    return { 'SubDir' : subCombinedDataDir, 'EXT': fileExtention }
        
    

def load_dataset( nameID, dataFlavor ):
    print("FILL IN")
    xTrainLearningSet = None
    yTrainVerificationSet = None
    xTestLearningSet = None
    yTestVerificationSet = None
    return xTrainLearningSet, yTrainVerificationSet, xTestLearningSet, yTestVerificationSet


def pickNormOrFlopForTrainingTestingLists( theList ):
    """
    Add '1' or '0' to pick a norm or flop version of picture.
    
    PARAMETERS:
    theList: The list
    """
    random.seed( datetime.datetime.utcnow() )
    returnList = []
    
    for element in theList:
        elementAsStringWithFlip = str( element ) + random.choice( ['0','1'] )
        returnList.append( int(elementAsStringWithFlip) )
        
    return returnList


def pickleTrainAndTestList( theTrainingList, theTestList, nameID ):
    """
    Pickle the Training and Test Lists
    
    PARAMETERS:
    theTrainingList: List, of the indexes of the training set
    theTestList: List, of the indexes of the test set
    nameID: String of the set ID.
    """
    # Just in case an Int get passed in.
    nameID = str(nameID)
    
    trainingPickleFileName = os.path.join( pickleDataDir, ('SET_' + nameID + '_TRAINING.pkl') )
    testPickleFileName = os.path.join( pickleDataDir, ('SET_' + nameID + '_TEST.pkl') )
    
    pickle.dump( theTrainingList, open( trainingPickleFileName, 'wb' ) )
    pickle.dump( theTestList, open( testPickleFileName, 'wb' ) )
    
      
def pickleLearningSet( learningNPArray, dataType, ofDataFlavor, withNameID ):
    """
    Pickles the NP Array
    
    PARAMETERS:
    theNPArray: The Numpy Array of the Training Set
    nameID: Int of the ID number.
    """
    stringOfFileName = dataType + '_' + str(withNameID) + '_LEARNING_' + ofDataFlavor + '_npArray.pkl'
    
    fullPathFileName = os.path.join( 
        pickleDataDir,
        stringOfFileName
    )

    print( fullPathFileName )
    
    pickle.dump( learningNPArray , open( fullPathFileName, 'wb') )

    
def pickleVerificationSet( verificationNPArray, dataType, dataFlavor, withNameID ):
    """
    Pickles the NP Array
    
    PARAMETERS:
    theNPArray: The Numpy Array of the Training Set
    nameID: Int of the ID number.
    """
    stringOfFileName = dataType + '_' + str(withNameID) + '_VERIFICATION_' + dataFlavor + '_npArray.pkl'
    
    fullPathFileName = os.path.join( 
        pickleDataDir,
        stringOfFileName
    )

    print( fullPathFileName )
    
    pickle.dump( verificationNPArray , open( fullPathFileName, 'wb') )

    
def pickleLearningAndVerificationNPArrays( learningNPArray, verificationNPArray, dataFlavor, nameID ):
    """
    Pickle both the Arrays
    
    PARAMETERS:
    learningNPArray: The learning NPArray
    verificationNPArray: The verification NPArray
    
    """
    pickleLearningSet( 
        learningNPArray,
        dataFlavor,
        nameID
    )
    
    pickleVerificationSet(
        verificationNPArray, 
        dataFlavor, 
        nameID
    )
    
    
def reportStats():
    """
    Quick function to see how many of every class we have.
    """
    UPs = 0
    DOWNs = 0
    NAs = 0
    HOLEs = 0

    dataDict = getDictFromDataCSVFile()
    
    for key in getListOfDataCSVFileKeys():
        if (dataDict[key]['CLASS'] == 'UP'):
            UPs = UPs + 1
        if (dataDict[key]['CLASS'] == 'DOWN'):
            DOWNs = DOWNs + 1
        if (dataDict[key]['CLASS'] == 'NA'):
            NAs = NAs + 1
        if (dataDict[key]['CLASS'] == 'HOLE'):
            HOLEs = HOLEs + 1
            
    print( "Number of Everything" )
    print( "UPs: " + str(UPs) )
    print( "DOWNs: " + str(DOWNs) )
    print( "NAs: " + str(NAs) )
    print( "HOLEs: " + str(HOLEs) )
    
    
    
def unpickleTrainAndTestList( nameID ):
    """
    Unpickle the Training and Test Lists

    PARAMETERS:
    nameID: String of the set ID.
    """
    # Just in case an Int get passed in.
    nameID = str(nameID)
    
    trainingPickleFileName = os.path.join( pickleDataDir, ('SET_' + nameID + '_TRAINING.pkl') )
    testPickleFileName = os.path.join( pickleDataDir, ('SET_' + nameID + '_TEST.pkl') )
    
    trainingSet = pickle.load( open( trainingPickleFileName, 'rb' ) )
    testSet = pickle.load( open( testPickleFileName, 'rb' ) )

    return  { 'TrainingList': trainingSet, 'TestList': testSet }


def unpickleAllTrainAndTestLists():
    """
    Unpickle the Training and Test List.
    """
    allTrainingLists = []
    allTestLists = []
    
    for index in range(10):
        bothTrainTestList = unpickleTrainAndTestList(index)
        
        allTrainingLists.append( bothTrainTestList['TrainingList'] )
        allTestLists.append( bothTrainTestList['TestList'] )

    return  { 'TrainingLists': allTrainingLists, 'TestLists': allTestLists }


def unpickleTrainingLists():
    """
    Unpickle the Training List.
    """
    allLists = unpickleAllTrainAndTestLists()
    return allLists['TrainingLists']
    

def unpickleTestLists():
    """
    Unpickle the Test List.
    """
    allLists = unpickleAllTrainAndTestLists()
    return allLists['TestLists']


# For Solo running Just in case.
# (Have no idea why that would ever happen)
if __name__ == "__main__":
    CS660DataManagementCheck()