import numpy as np
from statsmodels.sandbox.tools.tools_pca import pca
import scipy.stats as ss
import machine_learning as ml

class PCA_method:

    #initial the class with the index dataset, stock dataset, machine learning model, number of week to backtest
    def __init__(self, big_index, big_stock, method, wk):
        self.big_index_ret = big_index.pct_change().reset_index(drop=True)
        self.big_stock_ret = big_stock.pct_change().reset_index(drop=True)
        self.method = method
        self.wk = wk
        self.All_factor = []

    def PCA_method(self, i):
        index_ret = self.big_index_ret[:][-self.wk - 101 + i:-self.wk-1 + i] #unfiltered training set (X)
        self.stock_ret = self.big_stock_ret[:][-self.wk-100 + i:-self.wk + i] #training set (Y) lagging one week
        unfiltered_X_input = self.big_index_ret.iloc[[-self.wk-1+i]] #all index's return in correponding week

        index_ret = index_ret.reset_index(drop=True)
        self.stock_ret = self.stock_ret.reset_index(drop=True)

        PCs = self.find_PC(self.stock_ret.reset_index(drop=True)) #get the principal components
        sel_factor = self.find_sel_factor(index_ret, PCs) #filter the factors for machine learning
        self.All_factor.append(sel_factor)
        sel_factor = sorted(sel_factor, key=sel_factor.get, reverse=True)
        X_input = unfiltered_X_input[sel_factor] #filtered index return in correponding week
        index_ret = index_ret[sel_factor] #filter to the real training set (X)
        #use the corresponding machine learning model
        if self.method == 'SVR':
            stock_final_list = ml.SVR(index_ret, self.stock_ret, X_input)
        elif self.method == 'KNN':
            stock_final_list = ml.KNN(index_ret, self.stock_ret, X_input)
        elif self.method == 'MLP':
            stock_final_list = ml.MLP(index_ret, self.stock_ret, X_input)
        elif self.method == 'RF':
            stock_final_list = ml.RF(index_ret, self.stock_ret, X_input)
        elif self.method == 'OLS':
            stock_final_list = ml.OLS(index_ret, self.stock_ret, X_input)
        return stock_final_list.head(25).index.tolist() #return the 25 top performance stocks

    def find_PC(self, stock_ret): #find principal components with criterion of explaining 80% of variation
        reqExp = 0.8
        eig_val, eig_vec = np.linalg.eig(stock_ret.cov())
        eigVAL = np.cumsum(eig_val / np.sum(eig_val))
        nPC = np.where(eigVAL > reqExp)[0][0] + 1
        PCs = pca(stock_ret)[1][:, :nPC].T
        return PCs

    def find_sel_factor(self, index_ret, PCs): #find the factors to be selected
        sel_factor = {}
        reqCorr = 0.4
        reqFcorr = 0.7
        while (not bool(sel_factor) == True):
            for PC in PCs:
                for n in index_ret.columns:
                    corr = abs(ss.pearsonr(PC.real, index_ret[n])[0])
                    if corr > reqCorr:
                        if len(sel_factor) == 0:
                            sel_factor.update({n: corr})
                        else:
                            Fcorr = max([round(abs(ss.pearsonr(index_ret[n], index_ret[sel_fact])[0]), 6)
                                        for sel_fact in sel_factor])
                            if Fcorr < reqFcorr:
                                sel_factor.update({n: corr})
            reqCorr-=0.05
        return sel_factor
