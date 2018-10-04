import random
import matplotlib.pyplot as plt

from Perceptron import *
myPerceptron = Perceptron(0.1)


def whereIs(x, y):
    yLinea = 3*x + 2

    if(y <= yLinea):
        return 0
    else:
        return 1

for i in range(2):
    dotX = random.uniform(-5,5)
    dotY = random.uniform(-5,5)
    inputs = [dotX, dotY]
    myPerceptron.train(inputs, whereIs(dotX, dotY))

correct = []
realAns = []
answer = []
pointsX = []
pointsY = []
for i in range(50):
    dotX = random.uniform(-5, 5)
    dotY = random.uniform(-5, 5)
    inputs = [dotX, dotY]
    pointsX.append(dotX)
    pointsY.append(dotY)
    answer.append( myPerceptron.feed(inputs) )

    realAns.append(whereIs(dotX,dotY))


    if(realAns[i] == answer):
        correct.append(True)
    else:
        correct.append(False)

print(correct)

plt.scatter(pointsX,pointsY)
plt.show()