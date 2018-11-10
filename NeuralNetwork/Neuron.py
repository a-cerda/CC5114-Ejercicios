
import random, math
#Neuron class: implements a basic sigmoid neuron with the following fields:
#weights: a python list of weights for the inputs
#bias: the bias for the neuron
#inputs: the inputs for the calculation



class Neuron:

    def __init__(self, learningRate, numberOfInputs):
        self.weights = [random.uniform(-2.0,2.0) for i in range (numberOfInputs) ]
        self.bias = random.uniform(-2.0, 2.0)
        self.learningRate = learningRate
        self.inputs = []
        self.output = -1
        
    def sigmoid(self,result):
        return 1/(1+math.exp(result))

    def getWeights(self):
        return self.weights
    """
        :param inputs: 
        :return: 
    """
    def feed(self, inputs):
        result = 0
        if(len(inputs) == len(self.weights)):
            self.inputs = inputs

        for i in range(len(self.weights)):
            result += self.weights[i] * self.inputs[i]
        result *= -1
        result -= self.bias
        result = self.sigmoid(result)
        self.output = result
        return self.output

    def adjustWeights(self, delta):
        self.bias += self.learningRate * delta
        for i in range(len(self.weights)):
            self.weights[i] += (self.learningRate * delta * self.inputs[i])

