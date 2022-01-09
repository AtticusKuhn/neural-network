from neuralNetwork import NeuralNetwork


def main():
    """the main function run when the program starts"""
    neuralNetork: NeuralNetwork = NeuralNetwork([3,2,3])
    print(f'weights before {neuralNetork.weights}')
    # testResult = neuralNetork.predict([[1],[2],[3]])
    # print(f"testResult = {testResult}")
    # print(f"cost {neuralNetork.cost(testResult, [[1], [2], [3]])}")
    neuralNetork.train([
        ([[1],[2],[3]],[[1],[2],[3]] ),
        ([[1],[2],[3]],[[1],[2],[3]] ),
        ([[1],[2],[3]],[[1],[2],[3]] ),
        ([[1],[2],[3]],[[1],[2],[3]] ),
    ])
    print(f'weights after {neuralNetork.weights}')
if __name__ =="__main__":
    main()