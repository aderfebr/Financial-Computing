import numpy as np
import pandas as pd
import requests
import matplotlib.pylab as plt
from scipy.stats import norm,chi2,genpareto
from arch import arch_model

# VaR模型检验
def my_stat(r,var,pstar=0.05):
    N=np.sum(r>var)
    T=len(r)
    LRuc=-2*((T-N)*np.log(1-pstar)+N*np.log(pstar)) + 2*((T-N)*np.log(1-N/T)+N*np.log(N/T))
    pLRuc=1-chi2.cdf(LRuc,1)
    print("LRuc: {:.4f}".format(LRuc))
    print("pLRuc: {:.4f}".format(pLRuc))

    ind=r>var
    ind1=ind[:-1]
    ind2=ind[1:]
    n00=np.sum((ind1==0)&(ind2==0))
    n01=np.sum((ind1==0)&(ind2==1))
    n10=np.sum((ind1==1)&(ind2==0))
    n11=np.sum((ind1==1)&(ind2==1))
    pi01=n01/(n01+n00)
    pi11=n11/(n10+n11)
    pi2=(n01+n11)/(n00+n01+n10+n11)
    LRind=(n00+n10)*np.log(1-pi2)+(n01+n11)*np.log(pi2)-n00*np.log(1-pi01)-n01*np.log(pi01)-n10*np.log(1-pi11)-n11*np.log(pi11)
    LRind=-2*LRind
    pLRind=1-chi2.cdf(LRind,1)
    print("LRind: {:.4f}".format(LRind))
    print("pLRind: {:.4f}".format(pLRind))

    LRcc=LRuc+LRind
    pLRcc=1-chi2.cdf(LRcc,2)
    print("LRcc: {:.4f}".format(LRcc))
    print("pLRcc: {:.4f}".format(pLRcc))

# 爬虫
res = requests.get('http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol=sh600293&scale=240&ma=no&datalen=10000')
data_json = res.json()
data = pd.DataFrame(data_json)

data['close']=data['close'].astype('float')
data['return']=np.log(data['close'])-np.log(data['close']).shift(1)
data['date']=pd.to_datetime(data['day'],format="%Y-%m-%d")
data=data[(data['return']>=-0.1)&(data['return']<=0.1)]
data=data[data['date']>=pd.to_datetime("2010-01-01",format="%Y-%m-%d")]

# RiskMetrics方法
r=np.array(data['return'])*100
l=round(len(r)/3)
var_rm=np.zeros(len(r))
qalpha=norm.ppf(0.05)
for i in range(l,len(r)):
    mhat,shat=norm.fit(r[i-50:i])
    var_rm[i]=-(mhat+qalpha*shat)
plt.plot(r)
plt.plot(var_rm*-1)
plt.show()
print("RiskMetrics")
my_stat(r[l:]*-1,var_rm[l:])
print()

# Garch-Normal方法
var_gn=np.zeros(len(r))
for i in range(l,len(r)):
    am_ar_garch=arch_model(r[i-50:i],mean='AR',lags=1,vol='GARCH',p=2,q=2,dist='normal',rescale=False)
    res_ar_garch=am_ar_garch.fit(disp='off')
    a=res_ar_garch.forecast(horizon=1,align='origin')
    mean=a.mean['h.1'].iloc[-1]
    sigma=a.variance['h.1'].iloc[-1]
    var_gn[i]=-(mean+qalpha*np.sqrt(sigma))
plt.plot(r)
plt.plot(var_gn*-1)
plt.show()
print("GarchNormal")
my_stat(r[l:]*-1,var_gn[l:])
print()

# 历史模拟方法
var_hs=np.zeros(len(r))
qalpha=int(0.05*50)
for i in range(l,len(r)):
    his_sample=r[i-50:i]
    his_sample=np.sort(his_sample)
    var_hs[i]=-his_sample[qalpha-1]
plt.plot(r)
plt.plot(var_hs*-1)
plt.show()
print("HisSim")
my_stat(r[l:]*-1,var_hs[l:])
print()

# POT方法
var_evt=np.zeros(len(r))
alpha=0.05
for i in range(l,len(r)):
    his_sample=r[i-50:i]
    his_sample=np.sort(his_sample)
    ind=round(len(his_sample)*0.1)
    x=np.abs(his_sample[:ind])
    u=x[-1]
    y=x-u
    n=len(his_sample)
    Nu=len(y)
    parhat=genpareto.fit(y,floc=0)
    khat=parhat[0]
    sigmahat=parhat[2]
    var_evt[i]=u+sigmahat/khat*((alpha*n/Nu)**-khat-1)
plt.plot(r)
plt.plot(var_evt*-1)
plt.show()
print("EVT GPD")
my_stat(r[l:]*-1,var_evt[l:])
print()