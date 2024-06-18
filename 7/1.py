import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import scipy.stats as st
import statsmodels.api as sm
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 冲击检验
def zt_test(zt,m):
    n=1000
    X=np.zeros((n-m,m))
    Y=np.zeros((n-m))
    for i in range(n-m):
        X[i]=zt[i:i+m]
        Y[i]=zt[i+m]
    X=sm.add_constant(X)
    reg=sm.OLS(Y,X)
    res=reg.fit()
    print("滞后阶数为: {:d}".format(m))
    print("统计量: {:.4f}".format((n-m)*res.rsquared))
    print("p值: {:.4f}".format(1-st.chi2.cdf((n-m)*res.rsquared,m)))

tmp1=pd.read_csv('RESSET_DRESSTK_1990_2000_1.csv',encoding='gbk',usecols=[0,2,3])
tmp2=pd.read_csv('RESSET_DRESSTK_2001_2010_1.csv',encoding='gbk',usecols=[0,2,3])
tmp3=pd.read_csv('RESSET_DRESSTK_2011_2015_1.csv',encoding='gbk',usecols=[0,2,3])
tmp4=pd.read_csv('RESSET_DRESSTK_2016_2020_1.csv',encoding='gbk',usecols=[0,2,3])
data=pd.concat((tmp1,tmp2,tmp3,tmp4))
data.columns=['code','date','close']
tmp=pd.read_csv('RESSET_IDXQTTN_1.csv',encoding='gbk',usecols=[0,2,3])
tmp.columns=['code','date','close']
data=pd.concat((data,tmp))

data=data[data['code']==1]
data['ret']=np.log(data['close'])-np.log(data['close'].shift(1))
r=data['ret'].values[1:]
r=r[(r>=-0.1)&(r<=0.1)]

steps=[1,2,3,4,5,10,15,20]

# 收益率描述性统计
print("收益率:")
print("交易天数: {:d}".format(len(data)))
print("均值: {:.4f}".format(r.mean()))
print("标准差: {:.4f}".format(r.std(ddof=1)))
print("峰度: {:.4f}".format(st.kurtosis(r)))
print("偏度: {:.4f}".format(st.skew(r)))
print("最大值: {:.4f}".format(r.max()))
print("最小值: {:.4f}".format(r.min()))
print("自相关系数:")
for step in steps:
    print("{:.4f}".format(sm.tsa.acf(r)[step]),end=' ')
print('\n')

# 收益率绝对值描述性统计
r_abs=np.abs(r)
print("收益率绝对值:")
print("交易天数: {:d}".format(len(data)))
print("均值: {:.4f}".format(r_abs.mean()))
print("标准差: {:.4f}".format(r_abs.std(ddof=1)))
print("峰度: {:.4f}".format(st.kurtosis(r_abs)))
print("偏度: {:.4f}".format(st.skew(r_abs)))
print("最大值: {:.4f}".format(r_abs.max()))
print("最小值: {:.4f}".format(r_abs.min()))
print("自相关系数:")
for step in steps:
    print("{:.4f}".format(sm.tsa.acf(r_abs)[step]),end=' ')
print()

# ARCH模型估计
am_arch=arch_model(r*100,mean='constant',vol='ARCH',p=1,dist='normal')
res_arch=am_arch.fit()
# ARCH效应检验
print(res_arch.arch_lm_test(lags=5))
print(res_arch.arch_lm_test(lags=10))
print(res_arch.arch_lm_test(lags=15))
print(res_arch.summary())
# ARCH冲击检验
zt_test(res_arch.resid/res_arch.conditional_volatility,5)
zt_test(res_arch.resid/res_arch.conditional_volatility,10)
zt_test(res_arch.resid/res_arch.conditional_volatility,15)
# ARCH模型模拟
sim_arch=arch_model(None,mean='constant',vol='ARCH',p=1,dist='normal')
sim_arch_data=sim_arch.simulate(res_arch.params,1000)
plt.subplot(311)
plt.plot(r[:1000],label="真实值")
plt.plot(sim_arch_data['data']/100,label="模拟值")
plt.legend()
plt.title("ARCH")

# GARCH模型估计
am_garch=arch_model(r*100,mean='constant',vol='GARCH',p=1,q=1,dist='normal')
res_garch=am_garch.fit()
print(res_garch.summary())
# GARCH冲击检验
zt_test(res_garch.resid/res_garch.conditional_volatility,5)
zt_test(res_garch.resid/res_garch.conditional_volatility,10)
zt_test(res_garch.resid/res_garch.conditional_volatility,15)
# GARCH模型模拟
sim_garch=arch_model(None,mean='constant',vol='GARCH',p=1,q=1,dist='normal')
sim_garch_data=sim_garch.simulate(res_garch.params,1000)
plt.subplot(312)
plt.plot(r[:1000],label="真实值")
plt.plot(sim_garch_data['data']/100,label="模拟值")
plt.legend()
plt.title("GARCH")

# EGARCH模型估计
am_egarch=arch_model(r*100,mean='constant',vol='EGARCH',p=1,q=1,o=1,dist='normal')
res_egarch=am_egarch.fit()
print(res_egarch.summary())
# EGARCH冲击检验
zt_test(res_egarch.resid/res_egarch.conditional_volatility,5)
zt_test(res_egarch.resid/res_egarch.conditional_volatility,10)
zt_test(res_egarch.resid/res_egarch.conditional_volatility,15)
# EGARCH模型模拟
sim_egarch=arch_model(None,mean='constant',vol='EGARCH',p=1,q=1,o=1,dist='normal')
sim_egarch_data=sim_egarch.simulate(res_egarch.params,1000)
plt.subplot(313)
plt.plot(r[:1000],label="真实值")
plt.plot(sim_egarch_data['data']/100,label="模拟值")
plt.legend()
plt.title("EGARCH")
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0, hspace=0.1)
plt.show()