import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

def SVR(index_ret, stock_ret, X_input): #support vector regression
    from sklearn.svm import SVR
    X = index_ret
    y = stock_ret
    regr = SVR()
    svr_ret = []
    for column in y.columns:
        regs = regr.fit(X, y[column])
        svr_ret.append(regs.predict(X_input)[0])
    name = stock_ret.columns.to_list()
    svr_ret = pd.DataFrame(svr_ret, index=name, columns={'Return'})
    stock_final_list = svr_ret.sort_values(by='Return', ascending=False)
    return stock_final_list

def KNN(index_ret, stock_ret, X_input): #K-nearest N Neighbors
    neigh = KNeighborsRegressor()
    neigh.fit(index_ret, stock_ret)
    neigh.predict(X_input)
    knn_ret = neigh.predict(X_input)[0]
    name = stock_ret.columns.to_list()
    knn_ret = pd.DataFrame(knn_ret, index=name, columns={'Return'})
    stock_final_list = knn_ret.sort_values(by='Return', ascending=False)
    return stock_final_list

def MLP(index_ret, stock_ret, X_input): #Multilayer perceptron
    X_input = X_input.reset_index(drop=True)
    regr = MLPRegressor().fit(index_ret, stock_ret)
    mlp_ret = regr.predict(X_input).T
    name = stock_ret.columns.to_list()
    mlp_ret = pd.DataFrame(mlp_ret, index=name, columns={'Return'})
    stock_final_list = mlp_ret.sort_values(by='Return', ascending=False)
    return stock_final_list

def RF(index_ret, stock_ret, X_input): #Random forest
    X = index_ret
    y = stock_ret
    regressor = RandomForestRegressor()
    regressor.fit(X, y)
    rf_ret = regressor.predict(X_input)[0]
    name = stock_ret.columns.to_list()
    rf_ret = pd.DataFrame(rf_ret, index=name, columns={'Return'})
    stock_final_list = rf_ret.sort_values(by='Return', ascending=False)
    return stock_final_list

def OLS(index_ret, stock_ret, X_input): #Ordinary least squares
    fact_X = sm.add_constant(index_ret)
    X_input.insert(loc=0, column='const', value=[1])
    regs = [sm.OLS(stock_ret[column], fact_X).fit() for column in stock_ret.columns]
    beta = [regs[i].params for i in range(len(regs))]
    stock_pred_ret = np.dot(np.array(X_input), np.array(beta).transpose())[0]
    name = stock_ret.columns.to_list()
    stock_pred_ret = pd.DataFrame(stock_pred_ret, index=name, columns={'Return'})
    stock_final_list = stock_pred_ret.sort_values(by='Return', ascending=False)
    return stock_final_list
