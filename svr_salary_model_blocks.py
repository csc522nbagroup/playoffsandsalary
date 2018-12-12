import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt


data = {1:(1991, 1995,{'C': 0.1, 'epsilon': 0.01},{'C': 1, 'epsilon': 0.1}),
        2:(1996, 2000,{'C': 100, 'epsilon': 1}, {'C': 1, 'epsilon': 0.1}),
        3:(2001, 2005,{'C': 100, 'epsilon': 1}, {'C': 40, 'epsilon': 1}),
        4:(2006, 2010,{'C': 100, 'epsilon': 1}, {'C': 40, 'epsilon': 1}),
        5:(2011, 2016,{'C': 10, 'epsilon': 1}, {'C': 40, 'epsilon': 1})}


df = pd.read_excel('All_data.xlsx')
#Read data
for index, t in data.items():

    df_year = df.loc[df['Year'].between(t[0], t[1], inclusive=True)]

    X = np.array(df_year.drop('Salary', 1))
    y = np.array(df_year['Salary'])

    #Model definitions
    models = []
    lin_reg = LinearRegression()
    models = {'lin_reg':lin_reg}
    svr_lin = SVR(kernel='linear', C=t[2]['C'], epsilon=t[2]['epsilon'])
    models['svr_lin'] = svr_lin
    svr_rbf = SVR(kernel='rbf', C=t[3]['C'], epsilon=t[3]['epsilon'], gamma='auto')
    models['svr_rbf'] = svr_rbf

    #Split the data into five folds
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    cv_scores = []

    #Run Model
    for name, model in models.items():
        rmse = []
        adj_R2 = []
        r_squared = []
        for train_index, test_index in kf.split(X):
            #Split data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #Normalize data
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test = sc_X.transform(X_test)

            #Fit the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fold_rms = sqrt(mean_squared_error(y_test, y_pred))
            fold_confidence = model.score(X_test, y_test)
            adjusted_r_squared = 1 - (1 - fold_confidence) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
            r_squared.append(fold_confidence)
            rmse.append(fold_rms)
            adj_R2.append(adjusted_r_squared)
        #Append Results
        cv_scores.append((name, round(np.mean(rmse),5), round(np.mean(r_squared),5), round(np.mean(adj_R2),5)))

    print(t[0],"-", t[1])
    column_width=20
    print ("Model       rmse    R-squared    Adjusted-R-squared")
    for i1,i2,i3,i4 in cv_scores:
        print ("{:<11}{:<11}{:<11}{}".format(i1,i2,i3,i4))