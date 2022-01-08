from typing import List, Tuple
from utils import matrixAdd, matrixDimensions, matrixMultiply, randomMatrix, sigmoid, zeroMatrix

class NeuralNetwork:
    """Represents a Neural Network as described in https://youtu.be/aircAruvnKk"""
    def __init__(self, shape: List[int]):
        self.number_of_layers = len(shape)
        "the number of layers in the neural network"
        self.shape=shape
        "the shape of the neural network"
        self.biases:List[List[List[float]]] = list(map(lambda x: randomMatrix(1, x), shape[1:]))
        "these are the biases"
        self.weights: List[List[List[float]]] = []
        "these are the weights"
        for layer, nextLayer in zip(shape[:-1], shape[1:]):
            self.weights.append(randomMatrix(layer, nextLayer))
    def computeNextLayer(self, layer: int, activations: List[float]) -> List[float]:
        """ process of finding next layer as described at 13:28 of https://youtu.be/aircAruvnKk?t=808"""
        print("activations", activations)
        return list(map(lambda x : list(map(sigmoid, x)), matrixAdd(matrixMultiply(self.weights[layer], activations), self.biases[layer])))
    def  predict(self, vector: List[float]) -> List[float]:
        """given a vector of activations, feed through all the neurons and then find the results"""
        for i in range(self.number_of_layers-1):
            next = self.computeNextLayer(i, vector)
            print("next is", next)
            vector = next
        return vector
    def cost(self, a: List[List[float]], b: List[List[float]]) -> float:
        """the cost function as defined at 3:37 in https://youtu.be/IHZwWFHWa-w?t=217 It takes in 2 vectors and returns the cost between them"""
        cost = 0
        dimensionsA = matrixDimensions(a)
        dimensionsB = matrixDimensions(b)
        if dimensionsA[1] != 1:
            raise Exception(f" vector a should have width 1, but it actually was {dimensionsA}")
        if dimensionsB[1] != 1:
            raise Exception(f" vector b should have width 1, but it actually was {dimensionsB}")
        if dimensionsA[0] != dimensionsB[0]:
            raise Exception(f"vectors a and b should have the same height (a was {dimensionsA} and b was {dimensionsB})")
        for index,element  in enumerate(a):
            cost += (element[0] - b[index][0]) ** 2
        return cost 
    def schoasticGradientDescent(self, trainingData: List[Tuple[ List[List[float]], List[List[float]]]], miniBatchSize = 10):
        """schoastic gradient descent as defined at 9:34 of https://youtu.be/Ilg3gGewQ5U?t=574 
         basically break up the training data into mini batches and train on each mini batch
         trainingData is a list of tuples where the first element of the tuple is the input and the second element of the tuple is the expected output
        """
        miniBatches: List[List[Tuple[ List[List[float]], List[List[float]]]]] = []
        "the list of mini batches"
        numberOfBatches: int = len(trainingData) / miniBatchSize
        """the number of batches"""
        for i in range(0, numberOfBatches, miniBatchSize): # loop over the batch size
            miniBatches.append(trainingData[i:i+miniBatchSize])
        for batch in miniBatches: # for each batch
            self.trainOnSingleMiniBatch(batch) # train on each batch
    def trainOnSingleMiniBatch(self, miniBatch: List[Tuple[ List[List[float]], List[List[float]]]]) -> None:
        """given a miniBatch, train the network on the given miniBatch """
        overallDeltaCWeights: List[List[List[float]]] = []
        for bias in self.biases:
            overallDeltaCWeights.append(zeroMatrix(len(bias), len(bias[0])))
        overallDeltaCBiases: List[List[List[float]]] = []
        for bias in self.biases:
            overallDeltaCBiases.append(zeroMatrix(len(bias), len(bias[0])))
        for miniBatchInput, miniBatchExpectedOutput in miniBatch:
            deltaCWeights, deltaCBiases = self.backPropagation(miniBatchInput, miniBatchExpectedOutput)
            for index, deltaCWeight in enumerate(deltaCWeights):
               overallDeltaCWeights[index] = matrixAdd(overallDeltaCWeights[index] , deltaCWeight)
            for index, deltaCBias in enumerate(deltaCBiases):
               overallDeltaCBiases[index] = matrixAdd(overallDeltaCBiases[index],  deltaCBias)
        self.biases = list(map(lambda x : matrixAdd(x[0], x[1]), zip(self.biases ,overallDeltaCBiases)))
        self.weights = list(map(lambda x : matrixAdd(x[0], x[1]), zip(self.weights ,overallDeltaCWeights)))
    def train(self, trainingData):
        """train the neural network on the given training data"""
        return self.schoasticGradientDescent(trainingData, 10)
    def backPropagation(self, expectedResult:List[List[float]], actualResult: List[List[float]]) -> Tuple[List[List[List[float]]], List[List[List[float]]]] :
        """backpropagation, as descibed in the video https://youtu.be/Ilg3gGewQ5U
        it returns a tuple of delta C at 1:41 of https://youtu.be/Ilg3gGewQ5U?t=101
        """
        pass
