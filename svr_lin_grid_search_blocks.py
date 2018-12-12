
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



Years = {1991:1995,
         1996:2000,
         2001:2005,
         2005:2010,
         2011:2016}

#read data into df
df = pd.read_excel('All_data.xlsx')
for start, end in Years.items():
    df_year = df.loc[df['Year'].between(start, end, inclusive=True)]

    X = np.array(df_year.drop('Salary', 1))
    y = np.array(df_year['Salary'])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # ***************
    # SVR Linear
    # ***************

    svr_linear = GridSearchCV(SVR(kernel='linear', gamma='auto'), cv=5,
                              param_grid={"C": [0.1, 1, 10, 100],
                                          "epsilon": [0.01, 0.1, 1, 10]})
    svr_linear.fit(X_train, y_train)

    svr_linear_best = svr_linear.best_params_
    y_pred = svr_linear.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    confidence = svr_linear.score(X_test, y_test)

    print("Best params: ", svr_linear_best)
    print("rms: ", rms)
    print("R2-value: ", confidence)
    print()
