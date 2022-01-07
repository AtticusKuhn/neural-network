from utils import matrixAdd, matrixMultiply, randomRange, sigmoid


class NeuralNetwork:
    """Represents a Neural Network as described in https://youtu.be/aircAruvnKk"""
    def __init__(self, shape):
        self.shape=shape
        self.biases = list(map(lambda x: randomRange(x), shape)) # TODO
        self.weights = [] # TODO
        for layer, nextLayer in zip(shape[:-1], shape[1:]):
            self.weights.append(list(map((lambda row: randomRange(layer)), randomRange(nextLayer))))
    def computeNextLayer(self, layer, activations):
        """ process of finding next layer as described at 13:28 of https://youtu.be/aircAruvnKk?t=808"""
        map(sigmoid, matrixAdd(matrixMultiply(self.biases[layer], activations), self.biases[layer]))
    def  feedForward(self, vector):
        """given a vector of activations, feed through all the neurons and then find the results"""
        for i in range(len(self.shape)):
            vector = self.computeNextLayer(i, vector)
        return vector