# -*- coding: utf-8 -*-
# =============================================================================
# Group ID: 234
# Members: Laura Nyrup, Trine Jensen, Christian Toft.
# Date: 2018-10-24
# Lecture 8 Multilayer perceptron
# Dependencies: numpy, scipy.stats, scipy.io, matplotlib.pyplot, keras
# Python version: 3.6
# Functionality: Importet of the LDA reduced data. On this data the MLP is
# calculated using the keras packages.
# =============================================================================
import numpy as np
import scipy.io as io
from keras.models import Sequential
from keras.layers import Dense

data = io.loadmat('Data/mnist_lda.mat')

def accuracy(predicted, test_labels):
    """
    Compares the predicted classes with the class labels.

    Detailed description
    --------------------
        Using the Argmax function for each line in the cls matrix to get
        the class, the script then check and count the number of correct
        classes

    input:
    -----
        predicted:  #Test x #class matrix predicted from the model
                        The largest number is the expected class
        test_label: #Test x #class matrix given for the test set
                        1 for the class 0 for the rest.

    Calculating
    -----------
    r'$Sum_0^#test \mathds{1}(predicted = test_label)$'

    Output
    ------
        Accuracy for the model
    """
    N = predicted.shape[0]
    M = predicted.shape[1]
    classification = np.zeros(M)
    acc = 0
    for i in range(N):
        clf = np.argmax(predicted[i])
        if test_labels[i][clf] == 1:
            acc += 1
            classification[clf] += 1
    classification = classification/np.sum(test_labels, axis=0)
    print("Accuracy for:")
    print("All classes is %.2f" % (acc/N))
    for j in range(M):
        print("Class %d is %.2f" % (j, classification[j]))
    return acc/N, classification


# =============================================================================
# Data management
# =============================================================================
k = 10

# ALL DATA
AD = data['train_data']

# Labels for the data set
label = data['train_class']-1

# Matrix labels
Mlabel = data['train_class_01']

# Test data
TD = data['test_data']

# Test data labels
label_t = data['test_class']
Mlabel_t = data['test_class_01']

# =============================================================================
# MLP model creation
# =============================================================================
model = Sequential()
model.add(Dense(units=10, activation='sigmoid', init='normal', input_dim=9))
model.add(Dense(units=10, activation='sigmoid', init='normal'))
# model.add(Dense(units=10, activation='sigmoid', init='normal'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(AD, Mlabel, epochs=12, batch_size=32)
loss_and_metrics = model.evaluate(TD, Mlabel_t, batch_size=128)
Predicted = model.predict(TD)

print('\n')
print('The loss and metric calculated by keras is %.2f and %.2f, respectively'
      % (loss_and_metrics[0], loss_and_metrics[1]))
print('\n')

acc = accuracy(Predicted, Mlabel_t)
