from typing import List
from utils import matrixAdd, matrixMultiply, randomRange, sigmoid


class NeuralNetwork:
    """Represents a Neural Network as described in https://youtu.be/aircAruvnKk"""
    def __init__(self, shape: List[int]):
        self.shape=shape
        self.biases = list(map(randomRange, shape))
        # The wieghts
        self.weights: List[float] = [] 
        for layer, nextLayer in zip(shape[:-1], shape[1:]):
            self.weights.append(list(map((lambda row: randomRange(layer)), randomRange(nextLayer))))
    def computeNextLayer(self, layer: int, activations: List[float]):
        """ process of finding next layer as described at 13:28 of https://youtu.be/aircAruvnKk?t=808"""
        print("activations", activations)
        return list(map(lambda x : list(map(sigmoid, x)), matrixAdd(matrixMultiply([self.biases[layer]], activations), [self.biases[layer]])))
    def  feedForward(self, vector: List[float]):
        """given a vector of activations, feed through all the neurons and then find the results"""
        for i in range(len(self.shape)):
            vector = self.computeNextLayer(i, vector)
        return vector