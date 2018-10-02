from Perceptron import Perceptron


class Adder:
    def __init__(self):
        self.perceptron1 = Perceptron(3,[2,2])
        self.perceptron2 = Perceptron(3,[2,2])
        self.perceptron3 = Perceptron(3,[2,2])
        self.perceptron4 = Perceptron(3,[2,2])
        self.perceptroncarry = Perceptron(3,[2,2])

    def sum(self,input1,input2):
        result = []
        output1 = self.perceptron1.feed([input1,input2])
        output2 = self.perceptron2.feed([input1,output1])
        output3 = self.perceptron3.feed([input2,output1])
        sum = self.perceptron4.feed([output2,output3])
        result.append(sum)
        outputcarry = self.perceptroncarry.feed([output1,output1])
        result.append(outputcarry)
        return result

