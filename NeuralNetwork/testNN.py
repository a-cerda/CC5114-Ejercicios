from NeuralNetwork.neuralnetwork import *
import matplotlib.pyplot as plt

mynet = NeuralNetwork(3, [3,2,1], 2)
i=0
for layer in mynet.layers:
    print("this is layer: "+format(i))
    for neuron in layer.neurons:
        print(neuron.weights)
    i += 1
mynet.trainNetworkWithMultipleInputs([[1,1],[1,0],[0,1],[0,0]],2000,[0,1,1,0])
i = 0
for layer in mynet.layers:
    print("this is layer: " + format(i))
    for neuron in layer.neurons:
        print(neuron.weights)
    i += 1
#mynet.trainNetwork([1,1],500,0)
#mynet.trainNetwork([1,0],500,1)
#mynet.trainNetwork([0,1],500,1)
#mynet.trainNetwork([0,0],500,0)

# for layer in mynet.layers:
#     neurons = layer.getNeurons()
#     for neuron in neurons:
#         print("Neuron weights are: {}".format(neuron.getWeights()))
#         print("Neuon bias is:{}".format(neuron.bias))
error = mynet.getMeanSquaredError()
epochs = [i for i in range(2000)]
precision = mynet.getPrecision()
plt.plot(epochs,error,label='error')
plt.plot(epochs, precision,label='precision')
plt.legend()

plt.xlabel('Epoch')


plt.title("Error and precision")
plt.show()
print(mynet.feed([1,1]))
print(mynet.feed([1,0]))
print(mynet.feed([0,1]))
print(mynet.feed([0,0]))

