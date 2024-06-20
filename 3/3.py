import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

# 导入EP和size
data=pd.DataFrame()
for i in range(1,11):
    data_read=pd.read_excel("RESSET_MRESSTK_"+str(i)+".xls",usecols=[0,1,2,4,7])
    data=pd.concat([data,data_read])
data.columns=['code','date','close','tshare','EP']
data['EP']=1/data['EP']
data['size']=data['close']*data['tshare']
data['size']=np.log(data['size'])
data['yearmonth']=pd.to_datetime(data['date']).dt.strftime('%Y%m')
code=np.unique(data['code'].values)
for i in code:
    tmp=data[data['code']==i]
    if len(tmp)<12:
        continue
    p=tmp['close'].values
    r=(np.log(p[12:])-np.log(p[:-12]))/12
    r=np.concatenate((r,[np.nan]*12))
    data.loc[data['code']==i,'return']=r

# 导入beta
beta=pd.DataFrame()
for i in range(1,7):
    data_read=pd.read_csv("RESSET_SMONRETBETA_BFDT_"+str(i)+".csv",encoding='gbk',usecols=range(3))
    beta=pd.concat([beta,data_read])
beta.columns=['code','date','beta']
beta['yearmonth']=pd.to_datetime(beta['date']).dt.strftime('%Y%m')

# 拼接矩阵
data=pd.merge(left=data[['code','yearmonth','EP','size','return']],right=beta[['code','yearmonth','beta']],on=['yearmonth','code'],how='inner')
data.dropna(inplace=True)
yearmonth=np.unique(data['yearmonth'].values)

# 对每一个假设进行检验
assumption=[['size'],['EP'],['beta'],['size','EP'],['size','beta'],['EP','beta'],['size','EP','beta']]
for ass in assumption:
    coef=np.zeros((1,len(ass)+1))
    T=len(yearmonth)
    for i in range(T):
        tmp=data.loc[data['yearmonth']==yearmonth[i]]
        Y=tmp.loc[:,['return']].values
        X=tmp.loc[:,ass].values
        X=sm.add_constant(X)
        model=sm.OLS(Y,X)
        results=model.fit()
        coef=np.concatenate((coef,[results.params]))

    coef_mean=coef.mean(axis=0)
    tmp=np.zeros((1,4))
    tmp=coef.var(axis=0,ddof=1)
    tmp=coef_mean/np.sqrt(tmp/T)
    print("{:>10}".format('intercept'),end=' ')  # 假设
    for i in range(len(ass)):
        print("{:>10}".format(ass[i]),end=' ')  # 假设
    print()
    for i in range(len(ass)+1):
        print("{:10.4f}".format(coef_mean[i]),end=' ')  # 均值
    print()
    for i in range(len(ass)+1):
        print("{:10.4f}".format(tmp[i]),end=' ')  # 统计量
    print()
    for i in range(len(ass)+1):
        print("{:10.4f}".format(2-2*norm.cdf(abs(tmp[i]))),end=' ')  # p值 双侧检验
    print()