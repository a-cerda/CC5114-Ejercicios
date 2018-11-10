from NeuralNetwork.neuralnetwork import *

testnet = NeuralNetwork(2,[1,1],2)
for layer in testnet.layers:
    neuron = layer.getNeurons()
    if layer.layerNumber == 0:
        neuron[0].weights = [0.4,0.3]
        neuron[0].bias = 0.5
    elif layer.layerNumber == 1:
        neuron[0].weights = [0.3]
        neuron[0].bias = 0.4
testnet.trainNetwork([[1,1]],[1])

for layer in testnet.layers:
    for neuron in layer.getNeurons():
        print("Los pesos de la neurona son: "+format(neuron.getWeights()))
        print("El bias de la neurona es:"+format(neuron.bias))
