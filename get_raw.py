import pandas as pd
import yfinance as yf
from datetime import datetime
import requests
import quandl as quandl


nan_value = float("NaN")
perc = 10.0 # Like N %
fri=pd.date_range(start=str(2009), end=datetime.now(),freq='W-FRI').strftime('%Y-%m-%d').tolist()

#START FX RETURN-------------------------------------------------
url="https://finance.yahoo.com/currencies/"
html = requests.get(url).content
fx = pd.read_html(html,header=0)
fx_list = fx[-1]['Symbol'].tolist()
weeklyfxdf = yf.download(fx_list, start="2009-01-01")['Adj Close']

for i in range(len(weeklyfxdf)):
    if not (weeklyfxdf.index[i].weekday()==4):
       weeklyfxdf.iloc[i]=nan_value
weeklyfxdf.dropna(axis=0,how='all',inplace=True)

#END FX RETURN-------------------------------------------------





#START MARKET INDEX RETURN-------------------------------------------------
urli="https://finance.yahoo.com/world-indices/"
htmli = requests.get(urli).content
index = pd.read_html(htmli,header=0)
index_list = index[0]['Symbol'].tolist()
weeklyidf= yf.download(index_list, start="2009-01-01")['Adj Close']
for i in range(len(weeklyidf)):
    if not (weeklyidf.index[i].weekday()==4):
        weeklyidf.iloc[i]=nan_value
weeklyidf.dropna(axis=0,how='all',inplace=True)
#END  MARKET INDEX  RETURN-------------------------------------------------





#START Commodities-------------------------------------------------
urlc="https://sg.finance.yahoo.com/commodities/"
htmlc = requests.get(urlc).content
com = pd.read_html(htmlc,header=0)
com_list = com[0]['Symbol'].tolist()
weeklycomdf = yf.download(com_list, start="2009-01-01")['Adj Close']

for i in range(len(weeklycomdf)):
    if not (weeklycomdf.index[i].weekday()==4):
       weeklycomdf.iloc[i]=nan_value
weeklycomdf.dropna(axis=0,how='all',inplace=True)

#END Commodities RETURN-------------------------------------------------




#START Bonds-------------------------------------------------
urlb="https://sg.finance.yahoo.com/bonds/"
htmlb = requests.get(urlb).content
bond= pd.read_html(htmlb,header=0)
bond_list = bond[0]['Symbol'].tolist()
weeklybdf = yf.download(bond_list, start="2009-01-01")['Adj Close']

for i in range(len(weeklybdf)):
    if not (weeklybdf.index[i].weekday()==4):
       weeklybdf.iloc[i]=nan_value
weeklybdf.dropna(axis=0,how='all',inplace=True)

#END Bonds RETURN-------------------------------------------------


#START other index-------------------------------------------------
quandl.ApiConfig.api_key = 'yCGxrVf45xRo4T-cvqfa'
weeklyxdf = quandl.get(['USTREASURY/YIELD','ML/AAAEY','ML/BEY','ML/AEY','ML/BBBEY','ML/USEY','ML/EMHYY','FRED/T10YIE'],start_date='2009-01-01')
for i in range(len(weeklyxdf)):
    if not (weeklyxdf.index[i].weekday()==4):
       weeklyxdf.iloc[i]=nan_value
weeklyxdf.dropna(axis=0, how='all', inplace=True)
#end other index-------------------------------------------------


#Combind df
all_index = pd.concat([weeklyfxdf, weeklyidf,weeklycomdf,weeklybdf,weeklyxdf], axis=1)
min_count = int(((100-perc)/100)*all_index.shape[0] + 1)
all_index = all_index .dropna(axis=1,thresh=min_count)
all_index = all_index.interpolate(method ='linear', limit_direction ='forward')
all_index = all_index.interpolate(method ='linear', limit_direction ='backward')
all_index.to_csv('weekly_factor_price.csv')
#END FACTOR--------------------------------------------------------


#Start  STOCK--------------------------------------------------------
urls="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
htmls = requests.get(urls).content
stock = pd.read_html(htmls,header=0)
stock_list = stock[0]["Symbol"].to_list()
weeklysdf_raw = yf.download(stock_list, start="2009-01-01")['Adj Close']
weeklysdf=weeklysdf_raw
for i in range(len(weeklysdf)):
    if not (weeklysdf.index[i].weekday()==4):
        weeklysdf.iloc[i]=nan_value
weeklysdf.dropna(axis=0,how='all',inplace=True)

weeklysdf.index=weeklysdf.index.strftime('%Y-%m-%d').tolist()
z = list(set(fri)-set(weeklysdf.index))
zdf = pd.DataFrame(index=z)
weeklysdf=weeklysdf.append(zdf,sort=True)
weeklysdf=weeklysdf.sort_index()

min_count = int(((100-perc)/100)*weeklysdf.shape[0] + 1)
weeklysdf= weeklysdf.dropna(axis=1,thresh=min_count)
weeklysdf = weeklysdf.interpolate(method ='linear', limit_direction ='forward')
weeklysdf = weeklysdf.interpolate(method ='linear', limit_direction ='backward')
weeklysdf.to_csv('weekly_stock_price.csv')

#End  STOCK--------------------------------------------------------


#Start  benchmark--------------------------------------------------------
benchmark=yf.download('^GSPC',start="2009-01-01")['Adj Close']
benchmarkdf = benchmark.copy()
for i in range(len(benchmarkdf)):
    if not (benchmarkdf.index[i].weekday()==4):
        benchmarkdf.iloc[i]=nan_value
benchmarkdf.dropna(axis=0,how='all',inplace=True)

benchmarkdf.index=benchmarkdf.index.strftime('%Y-%m-%d').tolist()
zz = list(set(fri)-set(benchmarkdf.index))
zzdf = pd.DataFrame(index=zz)

benchmarkdf=pd.concat([benchmarkdf, zzdf], axis=0)
benchmarkdf=benchmarkdf.sort_index()

benchmarkdf = benchmarkdf.interpolate(method ='linear', limit_direction ='forward')
benchmarkdf= benchmarkdf.interpolate(method ='linear', limit_direction ='backward')
benchmarkdf.to_csv('benchmark.csv')

#End  benchmark--------------------------------------------------------