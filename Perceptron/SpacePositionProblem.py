import random
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import *

myPerceptron = Perceptron(0.1)


def whereIs(x, y):
    yLinea = 3*x + 2

    if(y <= yLinea):
        return 0
    else:
        return 1

for i in range(30):
    dotX = random.uniform(-5,5)
    dotY = random.uniform(-5,5)
    inputs = [dotX, dotY]
    myPerceptron.train(inputs, whereIs(dotX, dotY))

correct = []
realAns = []
answer = []
pointsX = []
pointsY = []
functionY = []
colors = []
for i in range(50):
    dotX = random.uniform(-5, 5)
    dotY = random.uniform(-5, 5)
    inputs = [dotX, dotY]
    pointsX.append(dotX)
    pointsY.append(dotY)
    answer.append( myPerceptron.feed(inputs) )
    functionY.append(3*dotX+2)
    realAns.append(whereIs(dotX,dotY))
    if (answer[i] == 1):
        colors.append('b')
    else:
        colors.append('r')

    if(realAns[i] == answer):
        correct.append(True)
    else:
        correct.append(False)

print(correct)

plt.plot(pointsX, functionY)
plt.scatter(pointsX,pointsY, c = colors)
plt.show()