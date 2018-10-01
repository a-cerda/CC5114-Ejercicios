

#Perceptron class: implements a basic perceptron with the following fields:
#weights: a python list of weights for the inputs
#bias: the bias for the perceptron
#inputs: the inputs for the calculation
class Perceptron:
    def __init__(self, bias, weights):
        self.weights = weights
        self.bias = bias
        self.inputs

    #think:
    def feed(self, inputs):
        if(len(inputs) == len(self.weights)):
            self.inputs = inputs
            result = 0
        for i in range(len(self.weights)):
            result += self.weights[i] * self.inputs[i]

        result += self.bias
