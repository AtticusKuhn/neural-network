import math
import random
from typing import Any, List, Tuple

def sigmoid(x):
    """The sigmoid function as described in 10:26 of https://youtu.be/aircAruvnKk?t=626"""
    print("x", x)
    return 1/(1+math.exp(-x))

def matrixMultiply(A: List[List[float]], B:List[List[float]]) -> List[List[float]]:
    """matrix multiplication as in linear algebra as in https://www.mathsisfun.com/algebra/matrix-multiplying.html"""
    print("A", A)
    print("B", B)
    if matrixDimensions(A)[1] != matrixDimensions(B)[0]:
        raise Exception(f"In matrix multiplication, the number of columns of the first must be the same as the nuber of rows in the second (the first matrix was {matrixDimensions(A)} and the second was {matrixDimensions(B)})")
    result = []
    for i in range(len(A)):
        result.append([0]*len(B[0]))
    # iterating by row of A
    for i in range(len(A)):
        # iterating by column by B
        for j in range(len(B[0])):
            # iterating by rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result    
    
def matrixAdd(X:List[List[float]], Y:List[List[float]]) -> List[List[float]]:
    """matrix addition as in linear algebra"""
    print("X", X)
    print("Y", Y)
    if not matrixDimensions(X) == matrixDimensions(Y):
        raise Exception(f"cannot add matricies with different dimensions (the first was {matrixDimensions(X)} and the second was {matrixDimensions(Y)})")
    result = []
    for i in range(len(X)):
        result.append([0]*len(X[0]))
    # iterate through rows
    for i in range(len(X)):  
    # iterate through columns
        for j in range(len(X[0])):
            result[i][j] = X[i][j] + Y[i][j]
    return result

def matrixDimensions(matrix: List[List[Any]]) -> Tuple[int, int]:
    """get the length and width of a matrix, i.e. how many rows and columns"""
    if not all(list(map(lambda x : len(x) == len(matrix[0]), matrix))):
        raise Exception(f" not all rows are the same length (lengths are {list(map(len, matrix))}") 
    return (len(matrix), len(matrix[0]))

def randomRange(length: int) -> List[float]:
    """returns a list of a length with random entries"""
    return list(map(lambda x: random.random(), list(range(length))))

def transpose(m):
    "transpose a matrix as defined in https://en.wikipedia.org/wiki/Transpose"
    return  [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

