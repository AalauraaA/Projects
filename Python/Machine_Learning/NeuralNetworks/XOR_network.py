# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:31:11 2020

@author: Laura
------------------------------------------------------------------------------
MACHINE LEARNING - XOR GATA
------------------------------------------------------------------------------
An XOR gate implements the digital logic exclusive OR operation. 
It takes two digital inputs, that can be equal to 0, representing a digital 
false value or 1, representing a digital true value and outputs 1 (true) if 
the inputs are different or 0 (false), if the inputs are equal. 
    * Input 0,0 --> Output 0
    * Input 0,1 --> Output 1
    * Input 1,0 --> Output 1
    * Input 1,1 --> Output 0
The XOR operation can be seen as a classification problem.
"""
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

np.random.seed(444)

" Input - 4 possible inputs "
X = np.array([[0, 0], 
              [0, 1],
              [1, 0],
              [1, 1]])

" Output - 2 possible outputs "
y = np.array([[0], [1], [1], [0]])

" Neural Network "
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid')) # output layer

#training of network - the weights
sgd = SGD(lr=0.1) # adjust the weights
model.compile(loss='mean_squared_error', optimizer=sgd) # minimize loss function

#applies the traning on training data
model.fit(X, y, batch_size=1, epochs=5000) # repeat training on data 5000 times

if __name__ == '__main__':
    print(model.predict(X))