import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

# 导入三因子
data_factors=pd.read_csv('Data_FFFactors.csv',encoding='gbk',usecols=[2,6,7,8])
data_factors.columns=['date','mkt','smb','hml']
data_factors['date']=pd.to_datetime(data_factors['date'])
data_factors['yearweek']=data_factors['date'].dt.strftime('%Y%U')

# 导入收益率
data_index=pd.read_csv('Data_Index.csv',encoding='gbk',usecols=[1,2,4])
data_index.columns=['idxname','date','return']
data_index['date']=pd.to_datetime(data_index['date'])
data_index['yearweek']=data_index['date'].dt.strftime('%Y%U')

# 按照代码提取指数
idxname=np.unique(data_index['idxname'].values)

data_list=[]
for i in idxname:
    data_list.append(data_index[data_index['idxname']==i])

# 导入无风险收益率
data_rf=pd.read_csv('Data_RiskFreeReturn.csv',encoding='gbk',usecols=[0,2])
data_rf.columns=['date','rfreturn']
data_rf['date']=pd.to_datetime(data_rf['date'])
data_rf['yearweek']=data_rf['date'].dt.strftime('%Y%U')

# 拼接指数矩阵
data_matrix=pd.merge(left=data_factors,right=data_rf[['yearweek','rfreturn']],on=['yearweek'],how='inner')
for i in range(len(data_list)):
    data_matrix=pd.merge(left=data_matrix,right=data_list[i][['yearweek','return']],on=['yearweek'],how='inner')
    data_matrix=data_matrix.rename(columns={'return': idxname[i]})
data_matrix.dropna(inplace=True)
data_matrix.sort_values(by='date',inplace=True)

# 多因子单资产
for i in idxname:
    print(i)
    x=data_matrix.loc[:,['mkt','smb','hml']].values
    ret_rf=data_matrix.loc[:,['rfreturn']].values
    ret_pt=data_matrix.loc[:,[i]].values
    X=sm.add_constant(x)
    Y=ret_pt-ret_rf
    model=sm.OLS(Y,X)
    results=model.fit()
    print(results.summary())

# 多因子多资产
T=len(Y)
N=10
K=3
ym=data_matrix.iloc[:,6:].values-data_matrix.loc[:,['rfreturn']].values
xm=sm.add_constant(x)
xmTxm=np.dot(xm.T,xm)
xmTym=np.dot(xm.T,ym)
AB_hat=np.dot(np.linalg.inv(xmTxm),xmTym)
ALPHA=AB_hat[0]

RESD=ym-np.dot(xm,AB_hat)
COV=np.dot(RESD.T,RESD)/T
invCOV=np.linalg.inv(COV)

fs=xm[:,1:]
mu_hat=np.mean(fs,axis=0).T
fs=fs-np.mean(fs,axis=0)
omega_hat=np.dot(fs.T,fs)/T
invOMG=np.linalg.inv(omega_hat)
xxx=np.dot(np.dot(mu_hat.T,invOMG),mu_hat)
yyy=np.dot(np.dot(ALPHA,invCOV),ALPHA.T)

GRS=(T-N-K)/N/(1+xxx)*yyy
print(GRS)
p_value=1-f.cdf(GRS,N,T-N-K)
print(p_value)