import math
import random

def sigmoid(x):
    """The sigmoid function as described in 10:26 of https://youtu.be/aircAruvnKk?t=626"""
    return 1/(1+math.exp(-x))

def matrixMultiply(A, B):
    """matrix multiplication as in linear algebra as in https://www.mathsisfun.com/algebra/matrix-multiplying.html"""
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
    
def matrixAdd(matrix1, matrix2):
    print("matrix1 is", matrix1)
    """matrix addition as in linear algebra"""
    return list(map(lambda leftRow, rightRow: list(map(lambda l,r :l+r, zip(leftRow, rightRow))), zip(matrix1, matrix2)))
def randomRange(length):
    """returns a list of a length with random entries"""
    return list(map(lambda x: random.random(), list(range(length))))