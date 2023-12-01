#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 23:51:05 2023

@author: Jason Heinrich
"""


import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

data_train = pd.read_csv('diabetes_train.csv')
data_test = pd.read_csv('diabetes_test.csv')


print(data_train.head())

y = data_train['Outcome']
X = data_train.drop(['Outcome'], axis = 1)

y_test = data_test['Outcome']
X_test = data_test.drop(['Outcome'], axis = 1)

print(X.head())

print(X.shape)

SVM_Model_rbf = SVC(kernel='rbf')
SVM_Model_rbf.fit(X,y)
SVM_Model_rbf_pred = SVM_Model_rbf.predict(X_test)

SVM_Model_linear = SVC(kernel='linear')
SVM_Model_linear.fit(X,y)
SVM_Model_linear_pred = SVM_Model_linear.predict(X_test)

SVM_Model_poly = SVC(kernel='poly')
SVM_Model_poly.fit(X,y)
SVM_Model_poly_pred = SVM_Model_poly.predict(X_test)

SVM_Model_sigmoid = SVC(kernel='sigmoid')
SVM_Model_sigmoid.fit(X,y)
SVM_Model_sigmoid_pred = SVM_Model_sigmoid.predict(X_test)

Neural_Net_Model_Relu_001 = MLPClassifier(max_iter=10000, learning_rate_init = 0.001, activation = 'relu', hidden_layer_sizes = (100, 100))
Neural_Net_Model_Relu_001.fit(X,y)
Neural_Net_Model_Relu_001_pred = Neural_Net_Model_Relu_001.predict(X_test)

Neural_Net_Model_Relu_01 = MLPClassifier(max_iter=10000, learning_rate_init = 0.01, activation = 'relu', hidden_layer_sizes = (100, 100))
Neural_Net_Model_Relu_01.fit(X,y)
Neural_Net_Model_Relu_01_pred = Neural_Net_Model_Relu_01.predict(X_test)

Neural_Net_Model_Logistic_001 = MLPClassifier(max_iter=10000, learning_rate_init = 0.001, activation = 'logistic', hidden_layer_sizes = (100, 100))
Neural_Net_Model_Logistic_001.fit(X,y)
Neural_Net_Model_Logistic_001_pred = Neural_Net_Model_Logistic_001.predict(X_test)

Neural_Net_Model_Logistic_01 = MLPClassifier(max_iter=10000, learning_rate_init = 0.01, activation = 'logistic', hidden_layer_sizes = (100, 100))
Neural_Net_Model_Logistic_01.fit(X,y)
Neural_Net_Model_Logistic_01_pred = Neural_Net_Model_Logistic_01.predict(X_test)

#print(SVM_Model_rbf.support_vectors_)
#print(SVM_Model_rbf.support_)
#print(SVM_Model_rbf.n_support_)
#print(f'Accuracy - SVM w/ rbf - : {SVM_Model_rbf.score(X,y):.3f}')
print(f'SVM w/ rbf - Test Accuracy -  : {SVM_Model_rbf.score(X_test,y_test):.3f}')

#print(f'Accuracy - SVM w/ linear - : {SVM_Model_linear.score(X,y):.3f}')
print(f'SVM w/ linear - Test Accuracy -  : {SVM_Model_linear.score(X_test,y_test):.3f}')

#print(f'Accuracy - SVM w/ poly - : {SVM_Model_poly.score(X,y):.3f}')
print(f'SVM w/ poly  - Test Accuracy - : {SVM_Model_poly.score(X_test,y_test):.3f}')

#print(f'Accuracy - SVM w/ sigmoid - : {SVM_Model_sigmoid.score(X,y):.3f}')
print(f'SVM w/ sigmoid  - Test Accuracy - : {SVM_Model_sigmoid.score(X_test,y_test):.3f}')

print(f'NN w/ relu 0.001  - Test Accuracy - : {Neural_Net_Model_Relu_001.score(X_test,y_test):.3f}')

print(f'NN w/ relu 0.01 - Test Accuracy - : {Neural_Net_Model_Relu_01.score(X_test,y_test):.3f}')

print(f'NN w/ logistic 0.001  - Test Accuracy - : {Neural_Net_Model_Logistic_001.score(X_test,y_test):.3f}')

print(f'NN w/ logistic 0.01 - Test Accuracy - : {Neural_Net_Model_Logistic_01.score(X_test,y_test):.3f}')