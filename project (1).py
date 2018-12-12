#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:05:15 2018

@author: shaohanwang
"""
import pandas as pd
import numpy as np
train_data=pd.read_csv('2016.csv')
test_data=pd.read_csv('2015.csv')

x_train=train_data.iloc[:,6:20]
x_test=test_data.iloc[:,6:20]
train_data['Playoffs'] = train_data['Playoffs'].map({'Y': 1, 'N': 0})
test_data['Playoffs'] = test_data['Playoffs'].map({'Y': 1, 'N': 0})

y_train= train_data.iloc[:,-1]
y_test=test_data.iloc[:,-1]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
input_dim = np.size(x_train, 1)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim=5,init='uniform',activation='relu',input_dim=input_dim))
classifier.add(Dense(output_dim=5,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=10)
y_predict=classifier.predict(x_test)
y_predict=(y_predict>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)