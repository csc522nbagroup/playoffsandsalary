#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:39:30 2018

@author: shaohanwang
"""
from numpy.random import seed
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
from numpy.random import seed
np.random.seed(7)
train_data=pd.read_csv('original.csv')
train_data2=pd.read_csv('after.csv')
train_data['Playoffs'] = train_data['Playoffs'].map({'Y': 1, 'N': 0})
train_data=train_data.dropna()
x=train_data.iloc[:,1:-1]
y= train_data.iloc[:,-1]
x_2=train_data2.iloc[:,1:-1]
y_2= train_data2.iloc[:,-1]
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train2,x_test2,y_train2,y_test2=train_test_split(x_2,y_2,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
input_dim = np.size(x_train, 1)
sc2=StandardScaler()
x_train2=sc2.fit_transform(x_train2)
x_test2=sc2.transform(x_test2)
input_dim2 = np.size(x_train2, 1)
#pca = PCA(n_components=10)
#x_train=pca.fit_transform(x_train)
#x_test=pca.transform(x_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
# def build_classifier():
#    classifier = Sequential()
#    classifier.add(Dense(output_dim=42,init='uniform',activation='relu',input_dim=input_dim))
#
#    classifier.add(Dense(output_dim=42,init='uniform',activation='relu'))
#    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
#    classifier.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
#    return classifier
# classifier= KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)

from sklearn.metrics import confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=96,init='uniform',activation='relu',input_dim=103))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=50)
y_predict=classifier.predict(x_test)
y_predict=(y_predict>0.5)


# classifier2 = Sequential()
# classifier2.add(Dense(output_dim=96,init='uniform',activation='relu',input_dim=input_dim2))
# classifier2.add(Dropout(p=0.1))
# classifier2.add(Dense(output_dim=96,init='uniform',activation='relu'))
# classifier2.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
# classifier2.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
# classifier2.fit(x_train2,y_train2,batch_size=10,epochs=50)
# y_predict2=classifier2.predict(y_test2)
# y_predict2=(y_predict2>0.5)

from sklearn.metrics import confusion_matrix, classification_report
cm=confusion_matrix(y_test,y_predict)
# cm2=confusion_matrix(y_test2,y_predict2)

print(classification_report(y_test,y_predict))
# print(classification_report(y_test2,y_predict2))

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_predict, y_test)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for  hyperparameter search(pca n=10)')
plt.legend(loc="lower right")
plt.show()
