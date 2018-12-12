# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:59:16 2018

@author: wangh
"""
import matplotlib.pyplot as plt
from original import y_probability,y_test,cm1
from hyperparameter import y_probability2,y_test2,cm2,classifier 
from hyperparameter_pca import y_probability3,y_test3,cm3
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc
import pandas as pd
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
y_test1=y_test
y_probability=y_probability
y_test2=y_test2
y_probability2=y_probability2
y_test3=y_test3
y_probability3=y_probability3
lw=2
plt.figure()

auc1 = roc_auc_score(y_test1, y_probability)
# calculate roc curveproject(ANN).zip
fpr, tpr, thresholds = roc_curve(y_test1, y_probability)
plt.plot(fpr, tpr, color='c', lw=1, label='original data(area = %0.7f)' %auc1)
auc2 = roc_auc_score(y_test2, y_probability2)
# calculate roc curve
fpr2, tpr2, thresholds = roc_curve(y_test2, y_probability2)
plt.plot(fpr2, tpr2, color='m', lw=1, label='hyperparameter search (area = %0.7f)' %auc2)


auc3 = roc_auc_score(y_test3, y_probability3)
# calculate roc curve
fpr3, tpr3, thresholds = roc_curve(y_test3, y_probability3)
plt.plot(fpr3, tpr3, color='chartreuse', lw=1, label='hyperparameter serach & pca  (area = %0.7f)' % auc3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.savefig('combined_roc.jpg')

plt.show()
