from NeuralNetwork.Neuron import Neuron


class NeuronLayer:

    def __init__(self, numberOfNeurons, layerNumber, inputsPerNeuron):
        self.neurons = []
        self.layerNumber = layerNumber
        self.output = []
        self.delta = []
        for i in range(numberOfNeurons):
            self.neurons.append(Neuron(0.1,inputsPerNeuron))


    def getDelta(self):
        return self.delta

    def getNeurons(self):
        return self.neurons

    def feed(self, inputs):
        self.output = []
        for i in range(len(self.neurons)):
            self.output.append(self.neurons[i].feed(inputs))
        return self.output

    def calculateOutputLayerDelta(self,expectedOutput):
        self.delta = []
        transferDerivative = self.trasferDerivative()
        if expectedOutput is list:
            for i in range(len(expectedOutput)):
                self.delta.append((expectedOutput[i]-self.output[i]) * transferDerivative[i])
        else:
            self.delta.append((expectedOutput - self.output[0]) * transferDerivative[0])


    def trasferDerivative(self):
        trasferDer = []
        for output in self.output:
            trasferDer.append(output * (1.0 - output))
        return trasferDer

    def calculateHiddenLayerDelta(self,nextLayer):
        self.delta = []
        nextLayerNeurons = nextLayer.getNeurons()
        deltas = nextLayer.getDelta()
        for i in range(len(self.neurons)):
            error = 0
            for j in range(len(nextLayerNeurons)):
                neuronWeights = nextLayerNeurons[j].getWeights()
                error += deltas[j] * neuronWeights[i]
            self.delta.append(error * self.trasferDerivative()[i])


    def adjustWeights(self):
        for i in range(len(self.neurons)):
            self.neurons[i].adjustWeights(self.delta[i])
