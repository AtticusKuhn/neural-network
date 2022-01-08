from neuralNetwork import NeuralNetwork


def main():
    neuralNetork: NeuralNetwork = NeuralNetwork([3,2,3])
    testResult = neuralNetork.predict([[1],[2],[3]])
    print(f"testResult = {testResult}")
    print(f"cost {neuralNetork.cost(testResult, [[1], [2], [3]])}")
if __name__ =="__main__":
    main()