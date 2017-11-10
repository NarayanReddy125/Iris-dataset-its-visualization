# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:43:30 2017

@author: narayanreddy
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importing dataset
dataset = pd.read_csv('Iris.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values

#Encoding the categorical data
#Encoding for dependent variables
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting data into train and test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Fitting lositic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(x_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




