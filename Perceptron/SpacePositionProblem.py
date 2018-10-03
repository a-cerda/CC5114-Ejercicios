import random

from Perceptron.Perceptron import *
myPerceptron = Perceptron(0.1)


def whereIs(x, y):
    yLinea = 3*x + 2

    if(y <= yLinea):
        return 0
    else:
        return 1


