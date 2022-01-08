from typing import List
from utils import matrixAdd, matrixMultiply, randomRange, sigmoid, transpose


class NeuralNetwork:
    """Represents a Neural Network as described in https://youtu.be/aircAruvnKk"""
    def __init__(self, shape: List[int]):
        self.number_of_layers = len(shape)
        self.shape=shape
        "the shape of the neural network"
        self.biases = list(map(randomRange, shape[1:]))
        "these are the biases"
        self.weights: List[float] = []
        "these are the weights"
        for layer, nextLayer in zip(shape[:-1], shape[1:]):
            self.weights.append(list(map((lambda row: randomRange(layer)), randomRange(nextLayer))))
    def computeNextLayer(self, layer: int, activations: List[float]) -> List[float]:
        """ process of finding next layer as described at 13:28 of https://youtu.be/aircAruvnKk?t=808"""
        print("activations", activations)
        return list(map(lambda x : list(map(sigmoid, x)), matrixAdd(matrixMultiply(self.weights[layer], activations), [self.biases[layer]])))
    def  predict(self, vector: List[float]) -> List[float]:
        """given a vector of activations, feed through all the neurons and then find the results"""
        for i in range(len(self.shape)):
            next = self.computeNextLayer(i, vector)
            print("next is", next)
            vector = next
        return vector