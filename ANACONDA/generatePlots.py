import os
import csv
import numpy as np
import matplotlib.pyplot as plt

resultsDirectory = os.path.join( os.getcwd(), os.pardir, 'RESULTS' )
resultPlotsDirectory = os.path.join( os.getcwd(), os.pardir, 'RESULTS_PLOTS' )

rootString = 'PyCHARM_TRAIN_PREDICTION_RESULTS'

TestLoss = []
TestAccuracy = []
NA = []
UP = []
DOWN = []
HOLE = []

headers = ['TestLoss','TestAccuracy','NA','UP','DOWN','HOLE']

def generatePlotsInResults():
	listFromDir = os.listdir( resultsDirectory )
	for element in listFromDir:
		if(rootString in element):
			pathToOpen = os.path.join( resultsDirectory, element )
			with open( pathToOpen, newline='\n') as csvFile:
				csvReader = csv.reader( csvFile, delimiter=',')
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

			plt.plot(TestAccuracy, TestAccuracy, 'r--')

			saveFileName = element + '.png'
			svaePath = os.path.join( resultPlotsDirectory, saveFileName ) 
			plt.savefig(savePath)

			TestLoss.clear()
			TestAccuracy.clear()
			NA.clear()
			UP.clear()
			DOWN.clear()
			HOLE.clear()

if __name__ == "__main__":
	"""
	"""
	generatePlotsInResults()

	print("__main__ Done.")
