# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import os
import csv
import random
import datetime

import numpy as np
import pickle

from PIL import Image

combinedDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'COMBINED' )
pickleDataDir = os.path.join( os.getcwd(), os.pardir, 'DATA', 'PICKLES' )
searCSVInfoFile = os.path.join(combinedDataDir, 'SEAR_DC_INFO.csv')

imageDirs = ['ICOLOR', 'IDEPTH']
csvDirs = ['PCLOUD']

allDataFlavors = imageDirs + csvDirs


def CS660DataManagementCheck():
    """
    Checks that this module loaded correctly.
    """
    print("CS660DataManagementCheck Imported")
    


def createAllBasicData():
    """
    This is core function to build all basic data.
    Only run this once
    Returns:

    """
    pickleCheck = os.path.join(pickleDataDir, 'PICKLES_EXIST.pkl')

    if (not os.path.exists( pickleCheck )):

        numberOfSetsToCreate = 5

        # print(searDCInfoCSVFileAsDict)

        dictOfClassLists =  getDictOfClassLists()

        print("")

        print("TEST CHECK")

        print("")


        for key in dictOfClassLists.keys():
            print(key, len(dictOfClassLists[key]))

        print("")

        allTestLists = []

        #  We want to geneate 5 sets of Training and Test List.
        for index in range(numberOfSetsToCreate):

            #First we generate Test Set, to remove them from the overall set.
            dictOfTestSetAndRemainingDict = getKeysForATestingList( dictOfClassLists, 40 )

            pickleTestList( dictOfTestSetAndRemainingDict['TestList'], index )
            allTestLists.append(dictOfTestSetAndRemainingDict['TestList'])
            dictOfClassLists = dictOfTestSetAndRemainingDict['RemainingDict']

            for key in dictOfClassLists.keys():
                print(key, len(dictOfClassLists[key]))

            print("")


        print("TRAINING CHECK")
        print("")

        # Now Create the Training Lists from the remaining Elements.
        # Because we have already removed the test keys we know they are disjoint.

        allTrainingLists = []

        for index in range(numberOfSetsToCreate):

            # First we generate Test Set, to remove them from the overall set.
            dictOfTrainingSetAndRemainingDict = getKeysForTrainingList(dictOfClassLists, 200)

            pickleTrainingList( dictOfTrainingSetAndRemainingDict['TrainingList'], index )
            allTrainingLists.append( dictOfTrainingSetAndRemainingDict['TrainingList'] )
            dictOfClassLists = dictOfTrainingSetAndRemainingDict['RemainingDict']

            for key in dictOfClassLists.keys():
                print(key, len(dictOfClassLists[key]))

            print("")


        for index in range(len(allTestLists)):
            for flavor in allDataFlavors:
                print( "TEST", index, flavor )

                dictOfLVArrays = createNPArraysFor( allTestLists[index], flavor )

                print( dictOfLVArrays['LEARNING'].shape, dictOfLVArrays['VERIFICATION'].shape)

                pickleLearningSet( dictOfLVArrays['LEARNING'], 'TEST', flavor, index)
                pickleVerificationSet( dictOfLVArrays['VERIFICATION'], 'TEST', flavor, index)


        for index in range(len(allTrainingLists)):
            for flavor in allDataFlavors:
                print( "TRAINING", index, flavor )

                dictOfLVArrays = createNPArraysFor( allTrainingLists[index], flavor )

                print( dictOfLVArrays['LEARNING'].shape, dictOfLVArrays['VERIFICATION'].shape)

                pickleLearningSet( dictOfLVArrays['LEARNING'], 'TRAINING', flavor, index)
                pickleVerificationSet( dictOfLVArrays['VERIFICATION'], 'TRAINING', flavor, index)


        pickleExistFile = os.path.join(pickleDataDir, 'PICKLES_EXIST.pkl')
        fo = open( pickleExistFile, 'w' )
        fo.write( 'Pickles Exists. Don\'t make anything.' )
        fo.close()

        print("")
        print('Pickles Made, good to go.')
        print("")
    else:
        print("")
        print('Pickles Exist, don\'t have to do anything.')
        print("")


def createNPArraysFor( theList, dataFlavor ):
    """
    Gets the NPArrays for a given list of Elements.

    PARAMETERS:
        theList: A List of element Keys
        dataFlavor: String of What type of Data we are looking at
            ['ICOLOR', 'IDEPTH', 'PCLOUD']
    """

    dataFlavorInfo = getSubDirectoryAndEXTForDataFlavor(dataFlavor)

    subDir = dataFlavorInfo['SubDir']
    extention = dataFlavorInfo['EXT']

    # Get the Learning NPArray
    returnLearningNPArray = getLearningSetAsNPArray(
        theList,
        subDir,
        extention
    )

    returnVerificationNPArray = getVerificationSetNPArray(theList)

    return {'LEARNING': returnLearningNPArray, 'VERIFICATION': returnVerificationNPArray}


def getKeysForTrainingList( dictOfLists, trainingSize ):
    """
    Generates the Training set of a certain size.
    
    PARAMETERS:
    trainingSize: Int of size you need.
    """
    
    # Seed with the current time
    random.seed( datetime.datetime.utcnow() )

    # Get the Class lists
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
    
    return { 'TrainingList' : returnTrainingList , 'RemainingDict' : dictOfLists }


def getKeysForATestingList( dictOfLists, testSize ):
    """
    Generates the Test set of a certain size.
    
    PARAMETERS:
    dictOfLists: A Dictionary Containing all the class lists.
    testSize: Int of size you need.
    """
    
    # Seed with the current time
    random.seed( datetime.datetime.utcnow() )

    # Get the Class lists
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
    # We slice off those last elements and put them into a list.
    # we then replace the individual class lists in the dict with the remaining [First part] of Elements
    # in the dict of class list.
    startIndex = len(upList)-subTestSize
    upListLastFewElements = upList[startIndex:]
    dictOfLists['upList'] = upList[:startIndex]

    startIndex = len(downList)-subTestSize
    downListLastFewElements = downList[startIndex:]
    dictOfLists['downList'] = downList[:startIndex]
    
    startIndex = len(naList)-subTestSize
    naListLastFewElements = naList[startIndex:]
    dictOfLists['naList'] = naList[:startIndex]
    
    startIndex = len(holeList)-subTestSize
    holeListLastFewElements = holeList[startIndex:]
    dictOfLists['holeList'] = holeList[:startIndex]


    # Now we put all the seperate test lasses together
    testElementsMergedList = upListLastFewElements + \
                             downListLastFewElements + \
                             naListLastFewElements + \
                             holeListLastFewElements

    testElementsMergedList.sort()

    return { 'TestList' : testElementsMergedList , 'RemainingDict' : dictOfLists }


def getDataDictOfTestList( theList ):
    """
    Gets a Dict of the CSV Info Data for the generated training/test set.

    PARAMETERS:
    theList = A list of the keys of the train/test set.
    """

    dataDict = getDictFromSEARDCInfoCSVFile()

    returnDict = {}

    for element in theList:

        asString = str(element)
        minusLast = asString[:-1]
        asIntAgain = int(minusLast)

        returnDict[asIntAgain] = dataDict[asIntAgain]

    return returnDict
    

def getDictOfClassLists():
    """
    Get the CSV Data Classes as lists.
    """

    searDCInfoCSVDict = getDictFromSEARDCInfoCSVFile()

    upList = []
    downList = []
    naList = []
    holeList = []
    
    searDCInfoCSVDict = getDictFromSEARDCInfoCSVFile()

    for key in getListOfDataCSVFileKeys():
        if (searDCInfoCSVDict[key]['CLASS'] == 'UP'):
            oneAndZeroVersion = getNormAndFlopVersionFor(key)
            upList.append( oneAndZeroVersion[0] )
            upList.append( oneAndZeroVersion[1] )
        if (searDCInfoCSVDict[key]['CLASS'] == 'DOWN'):
            oneAndZeroVersion = getNormAndFlopVersionFor(key)
            downList.append( oneAndZeroVersion[0] )
            downList.append( oneAndZeroVersion[1] )
        if (searDCInfoCSVDict[key]['CLASS'] == 'NA'):
            oneAndZeroVersion = getNormAndFlopVersionFor(key)
            naList.append( oneAndZeroVersion[0] )
            naList.append( oneAndZeroVersion[1] )
        if (searDCInfoCSVDict[key]['CLASS'] == 'HOLE'):
            oneAndZeroVersion = getNormAndFlopVersionFor(key)
            holeList.append( oneAndZeroVersion[0] )
            holeList.append( oneAndZeroVersion[1])

    return { 'upList': upList, 'downList': downList, 'naList': naList, 'holeList' : holeList }


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




def getDictFromSEARDCInfoCSVFile():
    """
    Reads the Data CSV File.
    
    Returns:
    A Dictionary
    """
    searDCInfoCSVFileAsDict = {}

    with open( searCSVInfoFile, newline='') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            data =  {'ANGLE': float(row[1]), 'CLASS': row[2], 'TYPE': row[3]}
            searDCInfoCSVFileAsDict[ int(row[0]) ] = data

    return searDCInfoCSVFileAsDict


    
def getImageFileAsNPArray( theImageFile ):
    """
    Turn a PNG image as Numpy Array
    
    PARAMETER:
    image: An image
    
    RETURNS:
    Numpy Array
    """
    # The 'L' allow convertion to grayscale.
    theImage = Image.open(theImageFile).convert("L")

    # theImage = PIL.Image.open(theImageFile).convert("L")
    return np.array(theImage)/255.0


def getNormAndFlopVersionFor( aKey ):
    """
    Generate the Norm and Flop keys for aKey.
    Args:
        aKey: A Data Key [Int]

    Returns:
        A List of both versions.
    """
    asAString = str(aKey)
    addAZero = asAString + '0'
    addAOne = asAString + '1'

    return ( int(addAZero), int(addAOne) )


def getListOfDataCSVFileKeys():
    """
    Get the keys of the CSV Data File.
    
    Returns:
    A List
    """

    return list( getDictFromSEARDCInfoCSVFile().keys() )



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


def getNumeralValueOf( theClass ):
    """
    Get Numberal Class of a class
    Args:
        theClass: the Class as String

    Returns:
        An Int
    """

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


def getClassFromNumeral( value ):
    """
    Get Numberal Class of a class
    Args:
        theClass: the Class as String

    Returns:
        An Int
    """

    if (value==0):
        return 'NA'
    elif (value==1):
        return 'UP'
    elif (value==2):
        return 'DOWN'
    elif (value==3):
        return 'HOLE'
    else:
        return None




def getTrainingList( nameID ):
    """
    Unpickle the Training and Test, Learning and Verification Arrays for a session.

    PARAMETERS:
    nameID: String of the set ID.
    dataFlavor: String of dataFlavor ['ICOLOR', 'IDEPTH', 'PCLOUD']
    """
    # Just in case an Int get passed in.
    nameID = str(nameID)

    theLists = unpickleTrainAndTestList( nameID )

    return theLists['TrainingList']



def getTestList( nameID ):
    """
    Unpickle the Training and Test, Learning and Verification Arrays for a session.

    PARAMETERS:
    nameID: String of the set ID.
    dataFlavor: String of dataFlavor ['ICOLOR', 'IDEPTH', 'PCLOUD']
    """
    # Just in case an Int get passed in.
    nameID = str(nameID)

    theLists = unpickleTrainAndTestList( nameID )

    return theLists['TestList']



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
    """
    Unpickle the Training and Test, Learning and Verification Arrays for a session.

    PARAMETERS:
    nameID: String of the set ID.
    dataFlavor: String of dataFlavor ['ICOLOR', 'IDEPTH', 'PCLOUD']
    """
    # Just in case an Int get passed in.
    nameID = str(nameID)

    trainingLearningPickleFileName = os.path.join(
            pickleDataDir,
            ('TRAINING_' + nameID + '_LEARNING_' + dataFlavor + '_npArray.pkl')
            )
    trainingVerificationPickleFileName = os.path.join(
            pickleDataDir,
            ('TRAINING_' + nameID + '_VERIFICATION_' + dataFlavor + '_npArray.pkl')
            )
    testLearningPickleFileName = os.path.join(
            pickleDataDir, ('TEST_' + nameID + '_LEARNING_' + dataFlavor + '_npArray.pkl')
            )
    testVerificationPickleFileName = os.path.join(
            pickleDataDir, ('TEST_' + nameID + '_VERIFICATION_' + dataFlavor + '_npArray.pkl')
            )


    trainingLearningSet = pickle.load( open( trainingLearningPickleFileName, 'rb' ) )
    trainingVerificationSet = pickle.load( open( trainingVerificationPickleFileName, 'rb' ) )
    testLearningSet = pickle.load( open( testLearningPickleFileName, 'rb' ) )
    testVerificationSet = pickle.load( open( testVerificationPickleFileName, 'rb' ) )

    return  (
            trainingLearningSet,
            trainingVerificationSet,
            testLearningSet,
            testVerificationSet
            )


def load_datasetAndGetBaseLists( nameID, dataFlavor ):
    """
    Unpickle the Training and Test, Learning and Verification Arrays for a session.

    PARAMETERS:
    nameID: String of the set ID.
    dataFlavor: String of dataFlavor ['ICOLOR', 'IDEPTH', 'PCLOUD']
    """
    # Just in case an Int get passed in.
    nameID = str(nameID)

    trainingLearningPickleFileName = os.path.join(
        pickleDataDir,
        ('TRAINING_' + nameID + '_LEARNING_' + dataFlavor + '_npArray.pkl')
    )
    trainingVerificationPickleFileName = os.path.join(
        pickleDataDir,
        ('TRAINING_' + nameID + '_VERIFICATION_' + dataFlavor + '_npArray.pkl')
    )

    testLearningPickleFileName = os.path.join(
        pickleDataDir, ('TEST_' + nameID + '_LEARNING_' + dataFlavor + '_npArray.pkl')
    )
    testVerificationPickleFileName = os.path.join(
        pickleDataDir, ('TEST_' + nameID + '_VERIFICATION_' + dataFlavor + '_npArray.pkl')
    )

    theLists = unpickleTrainAndTestList( nameID )

    trainingLearningSet = pickle.load(open(trainingLearningPickleFileName, 'rb'))
    trainingVerificationSet = pickle.load(open(trainingVerificationPickleFileName, 'rb'))
    testLearningSet = pickle.load(open(testLearningPickleFileName, 'rb'))
    testVerificationSet = pickle.load(open(testVerificationPickleFileName, 'rb'))

    return (
        trainingLearningSet,
        trainingVerificationSet,
        theLists['TrainingList'],
        testLearningSet,
        testVerificationSet,
        theLists['TestList'],
    )


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


def pickleTrainingList( theTrainingList, nameID):
    """

    Args:
        theTestList:  The Test Set to Pickle
        nameID: Its IS

    Returns:

    """

    trainingPickleFileName = os.path.join(pickleDataDir, ('TRAINING_LIST_' + str(nameID) + '.pkl'))
    pickle.dump( theTrainingList, open(trainingPickleFileName, 'wb' ))


def pickleTestList( theTestList, nameID ):
    """

    Args:
        theTestList:  The Test Set to Pickle
        nameID: Its IS

    Returns:

    """

    testPickleFileName = os.path.join(pickleDataDir, ( 'TEST_LIST_' + str(nameID) + '.pkl'))
    pickle.dump( theTestList, open(testPickleFileName, 'wb' ))


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

      
def pickleLearningSet( learningNPArray, trainingOrTestType, ofDataFlavor, withNameID ):
    """
    Pickles the NP Array
    
    PARAMETERS:
    theNPArray: The Numpy Array of the Training Set
    nameID: Int of the ID number.
    """
    stringOfFileName = trainingOrTestType + '_' + str(withNameID) + '_LEARNING_' + ofDataFlavor + '_npArray.pkl'
    
    fullPathFileName = os.path.join( 
        pickleDataDir,
        stringOfFileName
    )

    print( fullPathFileName )
    
    pickle.dump( learningNPArray , open( fullPathFileName, 'wb') )

    
def pickleVerificationSet( verificationNPArray, trainingOrTestType, dataFlavor, withNameID ):
    """
    Pickles the NP Array
    
    PARAMETERS:
    theNPArray: The Numpy Array of the Training Set
    nameID: Int of the ID number.
    """
    stringOfFileName = trainingOrTestType + '_' + str(withNameID) + '_VERIFICATION_' + dataFlavor + '_npArray.pkl'
    
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

    dataDict = getDictFromSEARDCInfoCSVFile()
    
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
    
    
# def reshape( theTrainingSet ):
#     """
#     This is to add the 'Channels' Dimension for Keras.
#     """
#     getShape = theTrainingSet.shape
#     newShape = getShape + (1,)
#     return theTrainingSet.reshape(newShape)
    
    
def unpickleTrainAndTestList( nameID ):
    """
    Unpickle the Training and Test Lists

    PARAMETERS:
    nameID: String of the set ID.
    """
    # Just in case an Int get passed in.
    nameID = str(nameID)
    
    trainingPickleFileName = os.path.join( pickleDataDir, ('TRAINING_LIST_' + nameID + '.pkl') )
    testPickleFileName = os.path.join( pickleDataDir, ('TEST_LIST_' + nameID + '.pkl') )
    
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
    createAllBasicData()
    reportStats()