#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:39:30 2018

@author: shaohanwang
"""

import pandas as pd
import numpy as np
train_data=pd.read_csv('all.csv')

train_data['Playoffs'] = train_data['Playoffs'].map({'Y': 1, 'N': 0})
train_data=train_data.dropna(axis='columns')
x=train_data.iloc[:,6:-2]


y= train_data.iloc[:,-1]
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
input_dim = np.size(x_train, 1)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim=42,init='uniform',activation='relu',input_dim=input_dim))

classifier.add(Dense(output_dim=42,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=100)
y_pre=classifier.predict(x_test)
y_pre=(y_pre>0.5)
from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,y_pre)
print(classification_report(y_test, y_pre))

#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Dense
#
#def build_classifier():
#    classifier = Sequential()
#    classifier.add(Dense(output_dim=42,init='uniform',activation='relu',input_dim=input_dim))
#
#    classifier.add(Dense(output_dim=42,init='uniform',activation='relu'))
#    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#    classifier.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
#    return classifier
#
#classifier=KerasClassifier(build_fn=build_classifier)
#parameters = {'batch_size':[25,32], 'epochs':[100,300]}
#
#grid_search=GridSearchCV(estimator=classifier, param_grid=parameters,scoring='accuracy',cv=10)
#grid_search=grid_search.fit(x_train,y_train)
#bset_parameters=grid_search.best_params_
#best_acc=grid_search.best_score_

