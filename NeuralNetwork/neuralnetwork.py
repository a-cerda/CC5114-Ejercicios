from NeuralNetwork.NeuronLayer import *


class NeuralNetwork:
    def __init__(self, numberOfLayers, neuronsPerLayer, numberOfInputs):
        """
        :param numberOfLayers: an integer specifying the number of layers
        :param neuronsPerLayer: a python list specifying the neurons per layer as an integer
        :param numberOfInputs: an integer specifying the number of inputs the network will take
        """
        self.precision = []
        self.meanSquaredError = []
        self.numberOfLayers = numberOfLayers
        self.inputLayer = [0 for i in range(numberOfInputs)]
        self.neuronsPerLayer = neuronsPerLayer
        self.layers = [NeuronLayer(self.neuronsPerLayer[0],0,numberOfInputs)]
        for i in range(1,numberOfLayers):
            self.layers.append(NeuronLayer(self.neuronsPerLayer[i],i,self.neuronsPerLayer[i-1]))

    def trainNetwork(self, trainingSet, expectedOutput):
        """

        :param trainingSet:
        :param expectedOutput:
        """
        self.inputLayer = trainingSet
        realOut = []
        for i in range(len(trainingSet)):
                realOut.append(self.feed(trainingSet[i]))

                self.backPropagate(expectedOutput[i])
                self.adjustWeights()
        self.calculatePrecision(realOut, expectedOutput)
        self.calculateError(realOut, expectedOutput)

    def trainNetworkWithMultipleInputs(self,trainingSet, numberOfEpochs, expectedOutput):
        """

        :param trainingSet:
        :param numberOfEpochs:
        :param expectedOutput:
        """
        for i in range(numberOfEpochs):
            self.trainNetwork(trainingSet,expectedOutput)

    def backPropagate(self, expectedOutput):
        """

        :param expectedOutput:
        """
        self.layers[-1].calculateOutputLayerDelta(expectedOutput) #we calculate the output layer delta
        for i in range(len(self.layers)-2,-1,-1):
            self.layers[i].calculateHiddenLayerDelta(self.layers[i+1])

    def adjustWeights(self):
   
        for i in range(len(self.layers)):
            self.layers[i].adjustWeights()

    def feed(self, data):
        self.inputLayer = data
        inputs = data
        for layer in self.layers:
            inputs = layer.feed(inputs)
        return inputs

    def calculateError(self,networkOutput,expectedOutput):
        error = 0
        for i in range(len(expectedOutput)):
            for j in range(len(expectedOutput[0])):
                error += (expectedOutput[i][j] - networkOutput[i][j])**2
        error /= len(expectedOutput)
        self.meanSquaredError.append(error)

    def getMeanSquaredError(self):
        return self.meanSquaredError

    def calculatePrecision(self, realOut, expectedOutput):
        precision = 0
        adaptedOut = []
        for i in range(len(expectedOutput)):
            adaptedList = []
            for j in range(len(expectedOutput[0])):
                if realOut[i][j] > 0.5:
                    adaptedList.append(1)
                else:
                    adaptedList.append(0)
            adaptedOut.append(adaptedList)
            if adaptedOut[i] == expectedOutput[i]:
                precision += 1
        precision = precision/len(expectedOutput)
        self.precision.append(precision)

    def getPrecision(self):
        return self.precision