"""
Quick Set of Functions to pull out the name of the class of elements added for a training age.
"""

import os
import csv

resultsDirectory = os.path.join(os.getcwd(), os.pardir, 'RESULTS')
resultPlotsDirectory = os.path.join(os.getcwd(), os.pardir, 'RESULTS_PLOTS')

rootString = 'PYCHARM_ADDED_ELEMENTS_RECORD'

elementAdded = []



def sortTheResulstOf(theList, toFileName):
    pathToOpen = os.path.join(resultsDirectory, toFileName)

    with open(pathToOpen, 'a', newline='\n') as csvfile:
        csvWriter = csv.writer(
            csvfile, delimiter=','
        )
        csvWriter.writerow(theList)


def sortResultsElments():
    listFromDir = os.listdir(resultsDirectory)
    for element in listFromDir:
        if (rootString in element):

            rowHeader = element.replace('PyCHARM_','')
            rowHeader = rowHeader.replace('_AT_0.txt', '')

            elementAdded.append(rowHeader)

            pathToOpen = os.path.join(resultsDirectory, element)
            with open(pathToOpen, newline='\n') as csvFile:
                csvReader = csv.reader(csvFile, delimiter=',')
                for row in csvReader:
                    elementAdded.append(row[0])


            prefix = None
            if('ICOLOR' in pathToOpen):
                prefix = 'ICOLOR'
                sortTheResulstOf(elementAdded, (prefix + '_elementAddedInAge.csv'))
            elif('IDEPTH' in pathToOpen):
                prefix = 'IDEPTH'
                sortTheResulstOf(elementAdded, (prefix + '_elementAddedInAge.csv'))
            elif('PCLOUD' in pathToOpen):
                prefix = 'PCLOUD'
                sortTheResulstOf(elementAdded, (prefix + '_elementAddedInAge.csv'))
            else:
                pass

            elementAdded.clear()



if __name__ == "__main__":
    """
    """
    sortResultsElments()
    print("__main__ Done.")
