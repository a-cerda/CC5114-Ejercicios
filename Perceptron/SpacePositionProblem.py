import random
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import *



def calculatePoints(numberOfPoints, trainingSteps):
    myPerceptron = Perceptron(0.1)
    steps = [i for i in range(trainingSteps)]
    trainingEvolution = []
    for i in range(trainingSteps):
        dotX = random.uniform(-5, 5)
        dotY = random.uniform(-5, 5)
        inputs = [dotX, dotY]
        myPerceptron.train(inputs, whereIs(dotX, dotY))

        correct = []
        realAns = []
        answer = []
        pointsX = []
        pointsY = []
        functionY = []
        colors = []
        for j in range(numberOfPoints):
            dotX = random.uniform(-5, 5)
            dotY = random.uniform(-5, 5)
            inputs = [dotX, dotY]
            pointsX.append(dotX)
            pointsY.append(dotY)
            answer.append(myPerceptron.feed(inputs))
            functionY.append(3 * dotX + 2)
            realAns.append(whereIs(dotX, dotY))
            if (answer[j] == 1):
                colors.append('b')
            else:
                colors.append('r')

            if (realAns[j] == answer):
                correct.append(True)
            else:
                correct.append(False)
        percentage = 0
        for (ans,real) in zip(answer,realAns):
            if ans == real:
                percentage += 1
        percentage /= numberOfPoints
        trainingEvolution.append(percentage)

    plt.plot(pointsX, functionY)
    plt.scatter(pointsX, pointsY, c=colors)
    plt.show()
    plt.plot(steps,trainingEvolution)
    plt.show()

def whereIs(x, y):
    yLinea = 3*x + 2

    if y <= yLinea:
        return 0
    else:
        return 1

calculatePoints(300,120)