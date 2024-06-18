import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.linear_model as sklm
from sklearn.preprocessing import StandardScaler

# 数据导入
data=pd.read_csv("data_sina_cjwl.csv")
data['day']=pd.to_datetime(data['day']).dt.strftime('%y%m%d')
tmp1=pd.read_csv("emo.csv")
tmp1['day']=pd.to_datetime(tmp1['day']).dt.strftime('%y%m%d')
tmp2=pd.read_csv("RESSET_BDDRFRET_1.csv")
data=pd.merge(left=data,right=tmp1,on='day',how='inner')
tmp2['day']=pd.to_datetime(tmp2['day']).dt.strftime('%y%m%d')
data=pd.merge(left=data,right=tmp2,on='day',how='inner')
data['volume']=np.log(data['volume'])
data['ret']=np.log(data['close'])-np.log(data['close'].shift(1))
data['exret']=data['ret']-data['rf']
data.dropna(inplace=True)

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
    Y=data['exret'].iloc[1:].values
    X=data[factor].iloc[:-1].values
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
def test_out(factor):
    datafit = data.copy(deep=True)
    n_in = np.sum(datafit['day'] <= '220000')
    n_out = np.sum(datafit['day'] > '220000')
    rout = np.zeros(n_out)
    rmean = np.zeros(n_out)
    rreal = np.zeros(n_out)
    rfree = np.zeros(n_out)
    volt2 = np.zeros(n_out)

    for i in range(n_out):
        Y=data['exret'].iloc[1:n_in+i].values
        X=data[factor].iloc[:n_in+i-1].values

        reg=sklm.LinearRegression()
        reg.fit(X,Y)
        f = datafit[factor].iloc[n_in+i-1].values.reshape(1,-1)
        rout[i]=reg.predict(f)[0]
        
        rmean[i]=np.mean(datafit['exret'].iloc[:n_in+i])
        rreal[i]=datafit['exret'].iloc[n_in+i]
        rfree[i]=datafit['rf'].iloc[n_in+i]
        volt2[i]=np.sum(datafit['ret'].iloc[(n_in+i-30):(n_in+i)]**2)

    print('因子:{}'.format(factor))
    myfun_stat_gains(rout, rmean, rreal)
    myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)

factors=['open','high','low','close','volume','ret','score']
for i in factors:
    test_in(i)
    test_out([i])

test_out(['ret','score'])