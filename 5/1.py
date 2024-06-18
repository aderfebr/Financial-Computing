import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据导入
data_tmp=pd.read_csv("RESSET_IDXMONRET_1.csv",encoding="gbk",usecols=range(3))
data_tmp.columns=['code','date','close']
data_tmp['date']=pd.to_datetime(data_tmp['date'])
data_tmp['yyyymm']=data_tmp['date'].dt.strftime("%Y%m")
grouped=data_tmp.groupby('code')
data=pd.DataFrame()
data['yyyymm']=np.unique(data_tmp['yyyymm'].values)
for code,close in grouped:
    data[code]=close['close'].values
data_tmp=pd.read_csv("RESSET_BDMONRFRET_1.csv",encoding="gbk",usecols=range(2))
data_tmp.columns=['date','rf']
data_tmp['date']=pd.to_datetime(data_tmp['date'])
data_tmp['yyyymm']=data_tmp['date'].dt.strftime("%Y%m")
data=pd.merge(left=data,right=data_tmp[['yyyymm','rf']],on='yyyymm',how='inner')
data_tmp=pd.read_csv("RESSET_THRFACDAT_MONTHLY_1.csv",encoding="gbk",usecols=range(4))
data_tmp.columns=['date','mkt','smb','hml']
data_tmp['date']=pd.to_datetime(data_tmp['date'])
data_tmp['yyyymm']=data_tmp['date'].dt.strftime("%Y%m")
data=pd.merge(left=data,right=data_tmp[['yyyymm','mkt','smb','hml']],on='yyyymm',how='inner')
dates=np.unique(data['yyyymm'])
codes=data.columns[1:11]

omega1=[]
omega2=[]
ewma=[]

for idx in range(len(dates)):
    if(dates[idx]>'201800'):

        rf=data.iloc[:idx,11].values
        mkt=data.iloc[:idx,12].values
        smb=data.iloc[:idx,13].values
        hml=data.iloc[:idx,14].values

        p=[]
        r=[]
        for i in range(10):
            p.append(data.iloc[:idx,i+1].values)
            r.append(np.log(p[i][1:])-np.log(p[i][:-1]))
            r[i]=r[i]-rf[1:]

        r=np.array(r).T

        # 样本方差-协方差矩阵
        Cov_Sample = np.mat(np.cov(r, rowvar=False))

        # 常量估计法
        Cov_Const = np.eye(Cov_Sample.shape[0])*np.mean(np.diag(Cov_Sample))
        Cov_Const += (np.ones(Cov_Sample.shape)-np.eye(Cov_Sample.shape[0]))*np.mean(Cov_Sample-np.diag(np.diag(Cov_Sample)))
        Cov_Const = np.matrix(Cov_Const)

        # 因子模型估计法
        X = np.mat(np.concatenate([np.ones((len(mkt)-1, 1)),mkt[1:, None],smb[1:, None],hml[1:, None]], axis=1))
        Y = np.mat(r)
        AB_hat = (X.T*X).I*(X.T*Y)
        ALPHA = AB_hat[0]
        BETA = AB_hat[1]
        RESD = Y - X*AB_hat
        covfactor = np.cov(mkt[1:])
        covresidual = np.diag(np.diag(np.cov(RESD, rowvar=False)))
        Cov_Factor = BETA.T*covfactor*BETA + covresidual

        # 压缩估计法
        c = min(((1-2/Cov_Sample.shape[0])*np.trace(Cov_Sample**2)+np.trace(Cov_Sample)**2)/(r.shape[0]-2/Cov_Sample.shape[0])/(np.trace(Cov_Sample**2)-np.trace(Cov_Sample)**2/Cov_Sample.shape[0]),1)
        Cov_Shrink = (1-c)*Cov_Sample + c*Cov_Factor

        # 指数加权移动平均法
        r_tmp=(r[i]-np.mean(r,axis=0)).reshape(-1,1)
        lam=0.95
        if not len(ewma):
            ewma.append(Cov_Sample)
        else:
            ewma.append((1-lam)*(r_tmp*r_tmp.T)+lam*ewma[-1])

        uhat=np.mean(r,axis=0)
        A=np.matrix(np.concatenate([uhat[:,None],np.ones((len(uhat),1))],axis=1)).T
        up=np.mean(uhat)
        b=np.mat(np.array([up,1])[:,None])

        # 最小化期望方差
        omega1.append(Cov_Factor.I*A.T*(A*Cov_Factor.I*A.T).I*b)

        # 最大化期望效用
        Q=3*Cov_Shrink
        omega2.append(Q.I*A.T*(A*Q.I*A.T).I*b-Q.I*(np.eye(len(uhat))-A.T*(A*Q.I*A.T).I*A*Q.I)*(-uhat[:,None]))
        
        if dates[idx]=='201805' or dates[idx]=='201907' or dates[idx]=='202010':
            plt.figure(figsize=(16, 12),dpi=60)
            plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.95,hspace=0.2,wspace=0.05)
            plt.subplot(3,2,1)
            sns.heatmap(Cov_Sample,annot=True,fmt='.4f',cmap='coolwarm')
            plt.title("{} 样本方差-协方差矩阵".format(dates[idx+1]))
            plt.subplot(3,2,2)
            sns.heatmap(Cov_Const,annot=True,fmt='.4f',cmap='coolwarm')
            plt.title("{} 常量估计法".format(dates[idx+1]))
            plt.subplot(3,2,3)
            sns.heatmap(Cov_Factor,annot=True,fmt='.4f',cmap='coolwarm')
            plt.title("{} 因子模型估计法".format(dates[idx+1]))
            plt.subplot(3,2,4)
            sns.heatmap(Cov_Shrink,annot=True,fmt='.4f',cmap='coolwarm')
            plt.title("{} 压缩估计法".format(dates[idx+1]))
            plt.subplot(3,2,5)
            sns.heatmap(ewma[-1],annot=True,fmt='.4f',cmap='coolwarm')
            plt.title("{} 指数加权移动平均估计法".format(dates[idx+1]))
            plt.show()

omega1=np.squeeze(omega1)
plt.plot(omega1,label=codes)
plt.legend()
plt.show()

N=len(omega1)
r=r[-N:]
ret=[]
for i in range(N-1):
    ret.append(np.sum(omega1[i]*r[i+1]))
plt.plot(ret)
plt.show()
print("最小化期望风险/因子模型估计法:")
print("均值:{:.4f}".format(np.mean(ret)))
print("标准差:{:.4f}".format(np.std(ret,ddof=1)))
print("夏普比率:{:.4f}".format(np.mean(ret)/np.std(ret,ddof=1)))

omega2=np.squeeze(omega2)
plt.plot(omega2,label=codes)
plt.legend()
plt.show()

N=len(omega2)
r=r[-N:]
ret=[]
for i in range(N-1):
    ret.append(np.sum(omega2[i]*r[i+1]))
plt.plot(ret)
plt.show()
print("最大化效用/压缩估计法:")
print("均值:{:.4f}".format(np.mean(ret)))
print("标准差:{:.4f}".format(np.std(ret,ddof=1)))
print("夏普比率:{:.4f}".format(np.mean(ret)/np.std(ret,ddof=1)))