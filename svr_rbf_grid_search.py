
"""
@author: sakthi
"""

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np

#read data into df
df = pd.read_excel('All_data.xlsx')

X = np.array(df.drop('Salary', 1))
y = np.array(df['Salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#***************
#SVR RBF
#***************
svr_rbf = GridSearchCV(SVR(kernel='rbf', gamma='auto'), cv=5,
                  param_grid={"C": [0.1, 1, 40, 100],
                            "epsilon": [0.001, 0.1, 1, 10]})
svr_rbf.fit(X_train, y_train)

y_pred = svr_rbf.predict(X_test)
rms = sqrt(mean_squared_error(y_test, y_pred))
confidence = svr_rbf.score(X_test,y_test)
svr_rbf_best = svr_rbf.best_params_
print("SVR_RBF: ")
print("Best params: ", svr_rbf_best)
print("rms: ", rms)
print("R2-value: ", confidence)
print()
