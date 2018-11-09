from NeuralNetwork.NeuronLayer import *


class NeuralNetwork:
    def __init__(self, numberOfLayers, neuronsPerLayer, numberOfInputs):
        """
        :param numberOfLayers: an integer specifying the number of layers
        :param neuronsPerLayer: a python list specifying the neurons per layer as an integer
        """
        self.numberOfLayers = numberOfLayers
        self.inputLayer = [0 for i in range(numberOfInputs)]
        self.neuronsPerLayer = neuronsPerLayer
        self.layers = [NeuronLayer(self.neuronsPerLayer[0],0,numberOfInputs)]
        for i in range(1,numberOfLayers):
            self.layers.append(NeuronLayer(self.neuronsPerLayer[i],i,self.neuronsPerLayer[i-1]))

    def trainNetwork(self, trainingSet, numberOfEpochs, expectedOutput):
        self.inputLayer = trainingSet
        for i in range(len(trainingSet)):
                self.feed(trainingSet[i])
                self.backPropagate(expectedOutput[i])
                self.adjustWeights()

    def trainNetworkWithMultipleInputs(self,trainingSet, numberOfEpochs, expectedOutput):
        for i in range(numberOfEpochs):
            self.trainNetwork(trainingSet,numberOfEpochs,expectedOutput)

    def backPropagate(self, expectedOutput):
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
