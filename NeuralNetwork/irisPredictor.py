from NeuralNetwork.neuralnetwork import *
import matplotlib.pyplot as plt
import csv
from pathlib import Path

#data handling and normalization

data_folder = Path("../IrisData/")

csvfile = data_folder / "iris.csv"

csvinputs = []

def normalize(data):
    minimum = 200
    maximum = 0
    for list in data:
        if min(list) < minimum:
            minimum = min(list)
        if max(list) > maximum:
            maximum = max(list)
    newData = []
    for list in data:
        newList = []
        for element in list:
            newList.append((element-minimum)/(maximum-minimum))
        newData.append(newList)
    return newData

expectedOutput = []
with open(csvfile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        csvinputs.append([float(item) for item in row[:3]])

        if row[4] == "Iris-setosa":
            expectedOutput.append([1,0,0])
        elif row[4] == "Iris-versicolor":
            expectedOutput.append([0,1,0])
        elif row[4] == "Iris-virginica":
            expectedOutput.append([0, 0, 1])


actualinputs = normalize(csvinputs)



#Network creation and training
irisPredictor = NeuralNetwork(3,[3,5,3],3)
numberofepochs = 1000
irisPredictor.trainNetworkWithMultipleInputs(actualinputs,numberofepochs,expectedOutput)

# #Plotting

error = irisPredictor.getMeanSquaredError()
epochs = [i for i in range(numberofepochs)]
precision = irisPredictor.getPrecision()
plt.plot(epochs,error,label='error')
plt.plot(epochs, precision,label='precision')
plt.legend()

plt.xlabel('Epoch')

plt.title("Error and precision")
plt.show()