from neuralNetwork import NeuralNetwork


def main():
    neuralNetork: NeuralNetwork = NeuralNetwork([3,2,3])
    print("weights", neuralNetork.weights)
    print("biases", neuralNetork.biases)
    testResult = neuralNetork.feedForward([[1,2,3]])
    print(f"testResult = {testResult}")
if __name__ =="__main__":
    main()