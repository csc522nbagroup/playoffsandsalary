
"""
@author: sakthi
"""

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from math import sqrt
import statsmodels.api as sm

#read data into df
df = pd.read_excel('All_data.xlsx')
df.info()
df = df.dropna()
df.info()

droplist = []

for a, b in zip(list(df), df.dtypes):
    print("a=", a, "b=", b)
    if b == object:
        droplist.append(a)
print(droplist)

# print("Before rename and drop")
# print(list(df))

df = df.drop(droplist, 1)
X = np.array(df.drop('Salary', 1))

y = np.array(df['Salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())

'''
lin_reg = LinearRegression().fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
confidence = lin_reg.score(X_test, y_test)
rms = sqrt(mean_squared_error(y_test, y_pred))
print("Linear_Regression: ")
print("rms: ", rms)
print("R-squared value : ", confidence)
print()
'''
