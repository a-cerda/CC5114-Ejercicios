import random
import seaborn

from Perceptron import *
myPerceptron = Perceptron(0.1)


def whereIs(x, y):
    yLinea = 3*x + 2

    if(y <= yLinea):
        return 0
    else:
        return 1

for i in range(100):
    dotX = random.uniform(-5,5)
    dotY = random.uniform(-5,5)
    inputs = [dotX, dotY]
    myPerceptron.train(inputs, whereIs(dotX, dotY))

correct = []
realAns = []
for i in range(50):
    dotX = random.uniform(-5, 5)
    dotY = random.uniform(-5, 5)
    inputs = [dotX, dotY]
    answer = myPerceptron.feed(inputs)

    realAns.append(whereIs(dotX,dotY))

    if(realAns == answer):
        correct.append(True)
    else:
        correct.append(False)
    print(correct[i])