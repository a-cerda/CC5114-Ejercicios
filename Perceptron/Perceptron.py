

#Perceptron class: implements a basic perceptron with the following fields:
#weights: a python list of weights for the inputs
#bias: the bias for the perceptron
#inputs: the inputs for the calculation
class Perceptron:
    def __init__(self, bias, weights):
        self.weights = weights
        self.bias = bias


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

        result += self.bias
        if(result > 0):
            realresult = 1
        else:
            realresult = 0
        return realresult
