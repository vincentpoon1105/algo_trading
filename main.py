import pandas as pd
import numpy as np
from arch import arch_model
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import cost_return as cr
import PCA

factor = pd.read_csv('weekly_factor_price.csv', index_col = 0, header = 0)
stock = pd.read_csv('weekly_stock_price.csv', index_col = 0, header = 0)
factor[factor==0] = 0.00001 #avoid infinity when calculating percentage change
stock[stock==0] = 0.00001

All_factor = [] #check the factors used in the backtesting periods
All_port = [] ##check the stocks used in the backtesting periods

def Cov_Mat_Forecast(__25stocks,returns):
    #function for covariance matrix modelling
    N = len(__25stocks)
    returns =100*returns
    coeffs = []
    cond_vol = []
    std_resids = []
    models = []
    for asset in returns.columns:
        #fit univariate garch(1,1) to stocks' returns
        model = arch_model(returns[asset], mean='Constant', vol='GARCH', p=1, o=0, q=1,rescale = True).fit(update_freq=0, disp='off');
        coeffs.append(model.params)
        cond_vol.append(model.conditional_volatility)
        std_resids.append(model.resid / model.conditional_volatility)
        models.append(model)
    std_resids_df = pd.DataFrame(std_resids).transpose().set_axis(returns.columns,axis='columns', inplace=False)
    #calulate the conditional correlation
    R = std_resids_df.transpose().dot(std_resids_df).div(len(std_resids_df))
    diag = []
    D = np.zeros((N, N))
    for model in models:
        #append each variance to form the diagonal matrix
        diag.append(model.forecast(horizon=1,reindex=False).variance.values[-1][0])
    diag = np.sqrt(np.array(diag))
    np.fill_diagonal(D, diag)
    #Covariance of Portfolio: Upsigma = DxRxD
    CovMat = np.matmul(np.matmul(D, R.values), D)
    return CovMat

def SmartBeta(__25stocks,returns,method):
    #Function of Smart Beta Scheme
    N = len(__25stocks)
    if method=='GMV':
        #minimum variance portfolio
        Cov=Cov_Mat_Forecast(__25stocks,returns)
        def objective(w):
            return np.sqrt(np.dot(w.T,np.dot(Cov,w)))
        #constraint: sum of weight should be one
        def constraint(w):
            return np.sum(w)-1
        #initial guess
        w0 = np.ones(N)/N
        #b for bounds
        b = (0,0.1)
        bnds = []
        for i in range(N):
            bnds.append(b)
        constraints = ({'type':'eq','fun':constraint})
        cons = [constraints]
        sol = minimize(objective,w0,method='SLSQP',bounds = bnds,constraints=cons)
        #return solution
        return sol.x
    elif method=='EW':
        #equal weight scheme
        return np.ones(N)/N
    else:
        return None

def combine(stock, weight): #to incorporate the stocks and respective weighting into a dictionary
    port = {}
    for i in range(len(stock)):
        port.update({stock[i]:weight[i]})
    return port

machine_model = 'SVR' #'SVR'/'KNN'/'MLP'/'RF'/'OLS'
weighting_method = 'GMV' #'GMV'/'EW' (global minimum variance / equal weighting)
wk = 520 #number of week of the backtesting period
PCA_class = PCA.PCA_method(factor, stock, machine_model, wk) #define the class for PCA and machien learning

for i in range(wk):
    print((i + 1) / wk * 100) #you may keep this line to see the progress of running the code
    _25_stock = PCA_class.PCA_method(i) #find top 25 stocks by PCA and machine learning
    ret = PCA_class.big_stock_ret[_25_stock].iloc[-wk + i, :].tolist() #find the corresponding return of the week
    weighting = SmartBeta(_25_stock, PCA_class.stock_ret[_25_stock], weighting_method) #decide of the weighting
    weighting = pd.DataFrame(weighting, index=_25_stock, columns={'Weighting'}).iloc[:, 0].tolist()
    new_port = combine(_25_stock, weighting)
    All_port.append(new_port)
    if i == 0:
        cost_return = cr.cost_return(new_port, ret) #define the cost_return class in the first week
    else:
        cost_return.update_all(new_port, ret) #update each week's cost and return

nav = cost_return.cap #get the weekly NAV values of the portfolio along the backtesting period

for i in range(len(nav)): #turn nav into portfolio size divided by that at t=0
  nav[i] = nav[i]*0.0000001

benchmark = pd.read_csv('benchmark.csv', index_col = 0, header = 0) #S&P performance of the same backtesting period
benchmarkcul = benchmark[-521:].pct_change()
benchmarkcul = (benchmarkcul + 1).cumprod()
benchmarkcul.iloc[0,0] = 1

#plot the performance graph
plt.plot(benchmarkcul, label = 'S&P500')
plt.plot(nav, label = machine_model)
plt.legend()
plt.ylabel('Cumulative Return')
plt.xlabel('Week')
plt.title(machine_model+' Performace')
plt.savefig(machine_model+'_Performance.png')
plt.show()


def cum_return(x):
    return x.iloc[-1] / x.iloc[0] - 1

def geo_return(x):
    T = 520*7
    return (x.iloc[-1] / x.iloc[0]) ** (365/T) -1

def annualize_vol(x):
    return x.pct_change().std() * (52**0.5)

def sharpe_ratio(x):
    tmp = x.pct_change()
    return tmp.mean() / tmp.std() * (52**0.5)

def sortino_ratio(x):
    tmp = x.pct_change()
    return tmp.mean() /  tmp[tmp>0].std() * (52**0.5)

def drawdown(x):
    return x / x.cummax() - 1

def mdd(x):
    return np.min(x / x.cummax() - 1)

def calmar_ratio(x):
    return geo_return(x) / abs(mdd(x))

def VaR(x):
    tmp = x.pct_change()
    return -tmp.quantile(0.05)

def cVaR(x):
    tmp = x.pct_change()
    return -tmp[tmp < tmp.quantile(0.05)].mean()

#get the KPI data as defined above
x = pd.DataFrame(benchmarkcul)
x[machine_model] = nav
x.to_csv(machine_model+'return.csv')
metrics = [cum_return(x).tolist(), geo_return(x).tolist(), annualize_vol(x).tolist(), sharpe_ratio(x).tolist() , mdd(x).tolist(),
            sortino_ratio(x).tolist(), calmar_ratio(x).tolist(), VaR(x).tolist(), cVaR(x).tolist()]
DF = pd.DataFrame(metrics, index=['Cumulative Return', 'Annualized Geometric Return', 'Annualized Volatility', 'Sharpe Ratio',
                             'Maximum Drawdown', 'Sortino Ratio', 'Calmar Ratio', 'VaR', 'cVaR'], columns =['S&P 500','Portfolio'])
DF.to_csv(machine_model+'_KPI.csv')
print(DF)

#plot the drawdown grahp
plt.plot(drawdown(x).iloc[:,0]*100, label = "S&P500")
plt.plot(drawdown(x).iloc[:,1]*100, label = machine_model)
plt.legend()
plt.ylabel('%')
plt.xlabel('Week')
plt.title(machine_model+' Drawdown')
plt.savefig(machine_model+'_Drawdown.png')
