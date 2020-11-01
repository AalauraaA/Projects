"""
Fundamental library.
It is a multi-dimensional array library.
Numpy array are faster than python lists because numpy used fixed types.
Computers don't see numbers but binary so 5 is seen as 00000101 (int8) 8 bits or one byte. With NumPy 5 will be by
default casted as int32 type with four bytes 00000000 00000000 00000000 000000101 while list will used a lot more.
NumPy is faster to read less bytes in memory.
"""
# Import libraries
import numpy as np
import sys

" Basics "
a = np.array([1, 2, 3], dtype='int8')
b = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# print matrix
print(a)
print(b)

# get dimension
print('Dimension of a: ', a.ndim)
print('Dimension of b: ', b.ndim)

# get shape
print('Shape of a: ', a.shape)
print('Shape of b: ', b.shape)

# get type/size
print(f"Type of a: {a.dtype} \nSize of a: {a.itemsize} bytes \nTotal size of a: {a.nbytes} bytes")
print(f"Type of b: {b.dtype} \nSize of b: {b.itemsize} bytes \nTotal size of b: {b.nbytes} bytes")