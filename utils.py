import math
import random
from typing import List

def sigmoid(x):
    """The sigmoid function as described in 10:26 of https://youtu.be/aircAruvnKk?t=626"""
    return 1/(1+math.exp(-x))

def matrixMultiply(A: List[List[float]], B:List[List[float]]) -> List[List[float]]:
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
    
def matrixAdd(matrix1:List[List[float]], matrix2:List[List[float]]) -> List[List[float]]:
    print("matrix1 is", matrix1)
    """matrix addition as in linear algebra"""
    return list(map(addRow, zip(matrix1, matrix2)))
def addRow(row):
    (leftRow, rightRow) = row
    print("rightRow", rightRow)
    return list(map(addCol, zip(leftRow, rightRow)))
def addCol(col):
    (l,r) = col
    return l+r
def randomRange(length):
    """returns a list of a length with random entries"""
    return list(map(lambda x: random.random(), list(range(length))))