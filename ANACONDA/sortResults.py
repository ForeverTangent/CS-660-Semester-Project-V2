"""
Quick Set of Functions to collect the results of ['TestLoss', 'TestAccuracy', 'NA', 'UP', 'DOWN', 'HOLE'] into
their own csv files.
"""
import os
import csv

resultsDirectory = os.path.join(os.getcwd(), os.pardir, 'RESULTS')
resultPlotsDirectory = os.path.join(os.getcwd(), os.pardir, 'RESULTS_PLOTS')

rootString = 'PyCHARM_TRAIN_PREDICTION_RESULTS'

TestLoss = []
TestAccuracy = []
NA = []
UP = []
DOWN = []
HOLE = []

headers = ['TestLoss', 'TestAccuracy', 'NA', 'UP', 'DOWN', 'HOLE']


def sortTheResulstOf(theList, toFileName):
    pathToOpen = os.path.join(resultsDirectory, toFileName)

    with open(pathToOpen, 'a', newline='\n') as csvfile:
        csvWriter = csv.writer(
            csvfile, delimiter=','
        )
        csvWriter.writerow(theList)


def sortResults():
    listFromDir = os.listdir(resultsDirectory)
    for element in listFromDir:
        if (rootString in element):

            rowHeader = element.replace('PyCHARM_TRAIN_PREDICTION_RESULTS_','')
            rowHeader = rowHeader.replace('_AT_0.txt', '')

            TestLoss.append(rowHeader)
            TestAccuracy.append(rowHeader)
            NA.append(rowHeader)
            UP.append(rowHeader)
            DOWN.append(rowHeader)
            HOLE.append(rowHeader)

            pathToOpen = os.path.join(resultsDirectory, element)
            with open(pathToOpen, newline='\n') as csvFile:
                csvReader = csv.reader(csvFile, delimiter=',')
                for row in csvReader:
                    if row[1] in headers:
                        pass
                    else:
                        TestLoss.append(float(row[0]))
                        TestAccuracy.append(float(row[1]))
                        NA.append(float(row[2]))
                        UP.append(float(row[3]))
                        DOWN.append(float(row[4]))
                        HOLE.append(float(row[5]))

            prefix = None
            if('ICOLOR' in pathToOpen):
                prefix = 'ICOLOR'
                sortTheResulstOf(TestLoss, (prefix + '_TestLoss_CollectedResults.csv'))
                sortTheResulstOf(TestAccuracy, (prefix + '_TestAccuracy_CollectedResults.csv'))
                sortTheResulstOf(NA, (prefix + '_NA_CollectedResults.csv'))
                sortTheResulstOf(UP, (prefix + '_UP_CollectedResults.csv'))
                sortTheResulstOf(DOWN, (prefix + '_DOWN_CollectedResults.csv'))
                sortTheResulstOf(HOLE, (prefix + '_HOLE_CollectedResults.csv'))
            elif('IDEPTH' in pathToOpen):
                prefix = 'IDEPTH'
                sortTheResulstOf(TestLoss, (prefix + '_TestLoss_CollectedResults.csv'))
                sortTheResulstOf(TestAccuracy, (prefix + '_TestAccuracy_CollectedResults.csv'))
                sortTheResulstOf(NA, (prefix + '_NA_CollectedResults.csv'))
                sortTheResulstOf(UP, (prefix + '_UP_CollectedResults.csv'))
                sortTheResulstOf(DOWN, (prefix + '_DOWN_CollectedResults.csv'))
                sortTheResulstOf(HOLE, (prefix + '_HOLE_CollectedResults.csv'))
            elif('PCLOUD' in pathToOpen):
                prefix = 'PCLOUD'
                sortTheResulstOf(TestLoss, (prefix + '_TestLoss_CollectedResults.csv'))
                sortTheResulstOf(TestAccuracy, (prefix + '_TestAccuracy_CollectedResults.csv'))
                sortTheResulstOf(NA, (prefix + '_NA_CollectedResults.csv'))
                sortTheResulstOf(UP, (prefix + '_UP_CollectedResults.csv'))
                sortTheResulstOf(DOWN, (prefix + '_DOWN_CollectedResults.csv'))
                sortTheResulstOf(HOLE, (prefix + '_HOLE_CollectedResults.csv'))
            else:
                pass

            TestLoss.clear()
            TestAccuracy.clear()
            NA.clear()
            UP.clear()
            DOWN.clear()
            HOLE.clear()


if __name__ == "__main__":
    """
    """
    sortResults()
    print("__main__ Done.")
