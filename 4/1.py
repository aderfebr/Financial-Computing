import numpy as np
import pandas as pd
from math import pi
import statsmodels.api as sm
import sklearn.linear_model as sklm
from sklearn.preprocessing import StandardScaler

# 数据导入和因子构建
data=pd.read_excel("1EData_PredictorData2019.xlsx",sheet_name='Monthly')
data['DP']=data['D12'].apply(np.log)-data['Index'].apply(np.log)
data['EP']=data['E12'].apply(np.log)-data['Index'].apply(np.log)
data['VOL']=data['CRSP_SPvw'].abs().rolling(window=12).mean()*np.sqrt(pi*6)
data['BILL']=data['tbl']-data['tbl'].rolling(window=12).mean()
data['BOND']=data['lty']-data['lty'].rolling(window=12).mean()
data['TERM']=data['tbl']-data['lty']
data['CREDIT']=data['AAA']-data['lty']
data['MA112']=data['Index']>=data['Index'].rolling(window=12).mean()
data['MA312']=data['Index'].rolling(window=3).mean()>=data['Index'].rolling(window=12).mean()
data['MOM6']=data['Index']>=data['Index'].shift(periods=6)
data['ExRet']=data['CRSP_SPvw']-data['Rfree']
data[['MA112','MA312','MOM6']]=data[['MA112','MA312','MOM6']].astype(int)

# 滞后一阶
data =  pd.concat((data[['yyyymm','CRSP_SPvw','Rfree','ExRet','DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG','MA112','MA312','MOM6']]
                   ,data[['DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG','MA112','MA312','MOM6']].shift(1)),axis=1)
data.columns = ['yyyymm','Ret','Rfree','ExRet','DP','EP','VOL','BILL','BOND','TERM','CREDIT','PPIG','IPG','MA112','MA312','MOM6',
                'DPL1','EPL1','VOLL1','BILLL1','BONDL1','TERML1','CREDITL1','PPIGL1','IPGL1','MA112L1','MA312L1','MOM6L1']
data=data[data['yyyymm']>=192701]
data.reset_index(drop=True,inplace=True)

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
    factor=factor+'L1'
    Y=data['ExRet']
    X=data[factor]
    X=sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    rg_con = results.params.iloc[0]
    rg_con_pvalue = results.pvalues.iloc[0]
    rg_DP = results.params.iloc[1]
    rg_DP_pvalue = results.pvalues.iloc[1]
    if rg_DP_pvalue <= 0.01:
        jud = '在1%的显著性水平下有样本内预测能力'
    elif (rg_DP_pvalue > 0.01) & (rg_DP_pvalue <= 0.05):
        jud = '在5%的显著性水平下有样本内预测能力'
    elif (rg_DP_pvalue > 0.05) & (rg_DP_pvalue <= 0.10):
        jud = '在10%的显著性水平下有样本内预测能力'
    else:
        jud = '无样本内预测能力'
    print('In-sample tests for one factor model with OLS:')
    print('Predictor: {:s}'.format(factor))
    print('Regressing Results: alpha = {:.4f}, beta = {:.4f}'.format(rg_con, rg_DP))
    print('Regressing Pvalues: p = {:.4f}, p = {:.4f}'.format(rg_con_pvalue, rg_DP_pvalue))
    print('Inference: {:s}'.format(jud))

# 样本外检验
def test_out(factor,model_name):
    factor_out = factor
    factor_lag = [i+'L1' for i in factor_out]
    datafit = data.copy(deep=True)

    n_in = np.sum(datafit['yyyymm'] <= 195612)
    n_out = np.sum(datafit['yyyymm'] > 195612)
    rout = np.zeros(n_out)
    rmean = np.zeros(n_out)
    rreal = np.zeros(n_out)
    rfree = np.zeros(n_out)
    volt2 = np.zeros(n_out)
    
    if model_name=='Linear':
        reg = sklm.LinearRegression()
    elif model_name=='Lasso':
        reg = sklm.LassoCV()
    elif model_name=='Ridge':
        reg = sklm.RidgeCV()
    elif model_name=='ElasticNet':
        reg = sklm.ElasticNetCV()
    for i in range(n_out):
        X = datafit[factor_lag].iloc[:n_in+i,:].values
        y = datafit['ExRet'].iloc[:n_in+i].values

        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        reg.fit(X,y)
        f = datafit[factor].iloc[n_in+i-1].values.reshape(1,-1)
        f=scaler.transform(f)
        rout[i]=reg.predict(f)[0]

        rmean[i] = np.mean(datafit['ExRet'].iloc[:n_in+i].values)
        rreal[i] = datafit['ExRet'].iloc[n_in+i]
        rfree[i] = datafit['Rfree'].iloc[n_in+i]
        volt2[i] = np.sum(datafit['Ret'].iloc[(n_in+i-12):(n_in+i)].values**2)

    print('Out-of-sample tests for model with '+model_name+' method:')
    print('Predictor: {}'.format(factor_out))
    R2os, MSFEadj, pvalue_MSFEadj = myfun_stat_gains(rout, rmean, rreal)
    Uout, Umean, DeltaU = myfun_econ_gains(rout, rmean, rreal, rfree, volt2, gmm=5)

factors = ['DP', 'EP', 'VOL', 'BILL', 'BOND', 'TERM', 'CREDIT', 'PPIG', 'IPG', 'MA112', 'MA312', 'MOM6']
for i in factors:
    test_in(i)
    test_out([i],'Linear')
test_out(factors,'Linear')
test_out(factors,'Lasso')
test_out(factors,'Ridge')
test_out(factors,'ElasticNet')