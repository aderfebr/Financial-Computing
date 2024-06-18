import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.linear_model as sklm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 读取月度数据
tmp1=pd.read_csv("RESSET_MRESSTK_1.csv",encoding='gbk',usecols=range(1,10))
tmp1.columns=['date','trd','turn','ret','rf','PE','EPS','ROE','PS']
tmp1['date']=pd.to_datetime(tmp1['date'])
tmp1['yyyymm']=tmp1['date'].dt.strftime("%Y%m")
tmp2=pd.read_csv("RESSET_SMONRETBETA_BFDT_1.csv",encoding='gbk',usecols=range(1,3))
tmp2.columns=['date','beta']
tmp2['date']=pd.to_datetime(tmp2['date'])
tmp2['yyyymm']=tmp2['date'].dt.strftime("%Y%m")
data_mon=pd.merge(left=tmp1,right=tmp2,on='yyyymm',how='inner')
data_mon['exret']=data_mon['ret']-data_mon['rf']
data_mon['flow']=np.abs(data_mon['ret'])/np.log(data_mon['trd'])

# 读取日度数据
tmp1=pd.read_csv("RESSET_DRESSTK_2001_2010_1.csv",encoding='gbk',usecols=range(1,4))
tmp1.columns=['date','close','ret']
tmp1['date']=pd.to_datetime(tmp1['date'])
tmp2=pd.read_csv("RESSET_DRESSTK_2011_2015_1.csv",encoding='gbk',usecols=range(1,4))
tmp2.columns=['date','close','ret']
tmp2['date']=pd.to_datetime(tmp2['date'])
tmp3=pd.read_csv("RESSET_DRESSTK_2016_2020_1.csv",encoding='gbk',usecols=range(1,4))
tmp3.columns=['date','close','ret']
tmp3['date']=pd.to_datetime(tmp3['date'])
tmp4=pd.read_csv("RESSET_DRESSTK_2021__1.csv",encoding='gbk',usecols=range(1,4))
tmp4.columns=['date','close','ret']
tmp4['date']=pd.to_datetime(tmp4['date'])
data_day=pd.concat((tmp1,tmp2,tmp3,tmp4))
data_day['yyyymm']=data_day['date'].dt.strftime("%Y%m")
data_day.dropna(inplace=True)
dates=pd.unique(data_day['yyyymm'])
data_mon=data_mon[data_mon['yyyymm'].isin(dates)]

# 计算因子
vol=np.zeros(len(dates))
high=np.zeros(len(dates))
skew=np.zeros(len(dates))
for i in range(len(dates)):
    tmp=data_day[data_day['yyyymm']==dates[i]]
    vol[i]=np.sum(tmp['ret'].values**2)
    tmp_3=data_day[(data_day['yyyymm']==dates[i-1])|(data_day['yyyymm']==dates[i-2])|(data_day['yyyymm']==dates[i-3])]
    high[i]=np.max(tmp['ret'].values)/np.max(tmp_3['ret'].values)
    skew[i]=np.sqrt(len(tmp['ret'].values))*np.sum(tmp['ret'].values**3)/np.sum(tmp['ret'].values**2)**1.5

data_mon['vol']=vol
data_mon['high']=high
data_mon['skew']=skew
data_mon=data_mon.iloc[3:]
data_mon.dropna(inplace=True)

# 统计显著性检验
def myfun_stat_gains(rout, rmean, rreal):
    R2os = 1 - np.sum((rreal-rout)**2)/np.sum((rreal-rmean)**2)
    d = (rreal - rmean)**2 - ((rreal-rout)**2 - (rmean-rout)**2)
    x = sm.add_constant(np.arange(len(d))+1)
    model = sm.OLS(d, x)
    fitres = model.fit()
    MSFEadj = fitres.tvalues[0]
    pvalue_MSFEadj = fitres.pvalues[0]
    if (R2os > 0) & (pvalue_MSFEadj <= 0.01):
        jud = '在1%的显著性水平下有样本外预测能力'
    elif (R2os > 0) & (pvalue_MSFEadj > 0.01) & (pvalue_MSFEadj <= 0.05):
        jud = '在5%的显著性水平下有样本外预测能力'
    elif (R2os > 0) & (pvalue_MSFEadj > 0.05) & (pvalue_MSFEadj <= 0.1):
        jud = '在10%的显著性水平下有样本外预测能力'
    else:
        jud = '无样本外预测能力'
    print('Stat gains: R2os = {:.4f}, MSFEadj = {:.4f}, MSFEpvalue = {:.4f}'.format(R2os, MSFEadj, pvalue_MSFEadj))
    print('Inference: {:s}'.format(jud))
    return R2os, MSFEadj, pvalue_MSFEadj

# 经济显著性检验
def myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5):
    omg_out = rout/volt2/gmm
    rp_out = rfree + omg_out*rreal
    Uout = np.mean(rp_out) - 0.5*gmm*np.var(rp_out)
    omg_mean = rmean/volt2/gmm
    rp_mean = rfree + omg_mean*rreal
    Umean = np.mean(rp_mean) - 0.5*gmm*np.var(rp_mean)
    DeltaU = Uout - Umean
    if DeltaU < 10**-6:
        jud = '没有经济意义'
    else:
        jud = '有经济意义'
    print('Econ Gains: Delta U = {:.4f}, Upred = {:.4f}, Umean = {:.4f}'.format(DeltaU, Uout, Umean))
    print('Inference: {:s}'.format(jud))
    return Uout, Umean, DeltaU

# 样本内检验
def test_in(factor):
    Y=data_mon['exret'].iloc[1:].values
    X=data_mon[factor].iloc[:-1].values
    X=sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    if results.pvalues[1] <= 0.01:
        jud = '在1%的显著性水平下有样本内预测能力'
    elif (results.pvalues[1] > 0.01) & (results.pvalues[1] <= 0.05):
        jud = '在5%的显著性水平下有样本内预测能力'
    elif (results.pvalues[1] > 0.05) & (results.pvalues[1] <= 0.10):
        jud = '在10%的显著性水平下有样本内预测能力'
    else:
        jud = '无样本内预测能力'
    print("因子:{}".format(factor))
    print('alpha = {:.4f}, beta = {:.4f}'.format(results.params[0], results.params[1]))
    print('p = {:.4f}, p = {:.4f}'.format(results.pvalues[0], results.pvalues[1]))
    print(jud)

# 样本外检验
def test_out(factor,model_name):
    datafit = data_mon.copy(deep=True)
    n_in = np.sum(datafit['yyyymm'] <= '201500')
    n_out = np.sum(datafit['yyyymm'] > '201500')
    rout = np.zeros(n_out)
    rmean = np.zeros(n_out)
    rreal = np.zeros(n_out)
    rfree = np.zeros(n_out)
    volt2 = np.zeros(n_out)

    for i in range(n_out):
        Y=data_mon['exret'].iloc[1:n_in+i].values
        X=data_mon[factor].iloc[:n_in+i-1].values

        if model_name=='Linear':
            reg=sklm.LinearRegression()

        elif model_name=='Lasso':
            reg=sklm.LassoCV()

        elif model_name=='Ridge':
            reg=sklm.RidgeCV()

        elif model_name=='ElasticNet':
            reg=sklm.ElasticNetCV()

        elif model_name=='SVR':
            reg=SVR()
        
        elif model_name=='KNN':
            reg=KNeighborsRegressor()

        elif model_name=='RF':
            reg=RandomForestRegressor()

        scaler=StandardScaler()
        X = scaler.fit_transform(X)
        reg.fit(X,Y)
        f = datafit[factor].iloc[n_in+i-1].values.reshape(1,-1)
        f = scaler.transform(f)
        rout[i]=reg.predict(f)[0]
        
        rmean[i]=np.mean(datafit['exret'].iloc[1:n_in+i])
        rreal[i]=datafit['exret'].iloc[n_in+i]
        rfree[i]=datafit['rf'].iloc[n_in+i]
        volt2[i]=np.sum(datafit['ret'].iloc[(n_in+i-12):(n_in+i)]**2)

    print('因子:{} 模型:{}'.format(factor,model_name))
    myfun_stat_gains(rout, rmean, rreal)
    myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)

factors=['PE','EPS','ROE','PS','turn','beta','vol','flow','high','skew']
for i in factors:
    test_in(i)
    test_out([i],'Linear')
test_out(factors,'Lasso')
test_out(factors,'Ridge')
test_out(factors,'ElasticNet')
test_out(factors,'SVR')
test_out(factors,'KNN')
test_out(factors,'RF')