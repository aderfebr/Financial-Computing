import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def RMSE(rout,rreal):
    return np.sqrt(np.mean((rreal-rout)**2))

def MAE(rout,rreal):
    return np.mean(np.abs(rreal-rout))

def Theil(rout,rreal):
    return np.sqrt(np.mean((rreal-rout)**2))/(np.sqrt(np.mean(rreal**2))+np.sqrt(np.mean(rout**2)))

def R2os(rout, rreal, rmean):
    return 1 - np.sum((rreal-rout)**2)/np.sum((rreal-rmean)**2)

def U(rout, rreal, rf, sigma, gmm=5):
    omg = rout/sigma/gmm
    rp = rf + omg*rreal
    return np.mean(rp) - 0.5*gmm*np.var(rp)

#读取模型数据
a=pd.read_csv("./Res/LGBMRegressor().csv",encoding='gbk',usecols=range(49))
b=pd.read_csv("./Res/Real.csv",encoding='gbk',usecols=range(49))
c=pd.read_csv("./Res/Mean.csv",encoding='gbk',usecols=range(49))
d=pd.read_csv("./Res/Sigma.csv",encoding='gbk',usecols=range(49))
rf=pd.read_csv("./Res/Rf.csv",encoding='gbk',usecols=range(48))
rf=rf.iloc[0]

data=pd.DataFrame()
for i in range(6):
    tmp=pd.read_csv("./Data/RESSET_MRESSTK_"+str(i+1)+".csv",encoding='gbk',usecols=range(13))
    tmp.columns=['code','date','close','vol','turn','shr','ret','rf','PE','PB','EPS','ROE','NAPS']
    tmp['date']=pd.to_datetime(tmp['date'])
    data=pd.concat((data,tmp))

year_end=19

data['yymm']=data['date'].dt.strftime("%y%m")
data=data[data['yymm']==str(year_end)+'01']

num=len(a)
ret=[[],[],[],[],[]]
ans=np.zeros(5)
plt.figure(figsize=(12,6),dpi=100)
plt.subplot(1,2,1)

for i in range(5):
    for year in range(year_end,23):
        for month in range(1,13):
            date='20{:0>2}-{:0>2}'.format(year,month)
            codes=a.sort_values(date,ascending=False)[round(i*num/5):round((i+1)*num/5)]['code']
            ret[i].append(np.mean(b[b['code'].isin(codes)][date]))
    tmp=pd.Series(ret[i])
    plt.plot((tmp+1).cumprod(),label='Q'+str(i+1))
    ans[i]=tmp.mean()/tmp.std(ddof=1)*np.sqrt(12)


for i in range(5):
    print("Q{}:{:.4f}".format(i+1,ans[i]))
ret=np.mean(b.iloc[:,1:].values,axis=0)
print("全样本:{:.4f}".format(ret.mean()/ret.std(ddof=1)*np.sqrt(12)))
print("Q1-Q5:{:.4f}".format(ans[0]-ans[4]))

plt.plot((ret+1).cumprod(),label='全样本')
plt.legend()
plt.title("各分位累计收益")

plt.subplot(1,2,2)
plt.bar(['Q1','Q2','Q3','Q4','Q5'],ans-ret.mean()/ret.std(ddof=1)*np.sqrt(12))
plt.title("各分位夏普比率相对全样本之差")
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0)
plt.show()

ans=np.zeros((5,len(a)))
for i in range(len(a)):
    label=b.iloc[i,1:]
    x=a.iloc[i,1:]
    mean=c.iloc[i,1:]
    sigma=d.iloc[i,1:]
    ans[0][i]=RMSE(x,label)
    ans[1][i]=MAE(x,label)
    ans[2][i]=Theil(x,label)
    ans[3][i]=R2os(x,label,mean)
    ans[4][i]=U(x,label,rf,sigma)

print('')
res=np.mean(ans,axis=1)
print('RMSE:{:.4f}'.format(res[0]))
print('MAE:{:.4f}'.format(res[1]))
print('Theil:{:.4f}'.format(res[2]))
print('R2os:{:.4f}'.format(res[3]))
print('U:{:.4f}'.format(res[4]))