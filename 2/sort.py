import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 导入数据并清洗
stock_data=pd.DataFrame()
for i in range(1,8):
    data_read=pd.read_csv("RESSET_MRESSTK_"+str(i)+".csv",encoding='gbk',usecols=[0,1,2,3])
    stock_data=pd.concat([stock_data,data_read],axis=0)
stock_data.columns=['code','date','close','return']
stock_data['date']=pd.to_datetime(stock_data['date']).dt.strftime("%Y%m")
dates=np.unique(stock_data['date'])

N_list=[1,3,6,12]  #跟踪期
M_list=[1,3,6,12]  #回测期

for N in N_list:
    for M in M_list:
        Q=np.zeros((1,5))
        for i in range(N,len(dates)-M):
            tmp1=stock_data[stock_data['date']==dates[i-N]]
            tmp2=stock_data[stock_data['date']==dates[i]]
            tmp3=stock_data[stock_data['date']==dates[i+M]]
            # 按照跟踪期进行排序分为5组
            tmp=pd.merge(left=tmp1,right=tmp2,on='code',how='inner')
            tmp['return']=(tmp['close_y']-tmp['close_x'])/tmp['close_x']
            tmp.sort_values(by='return',inplace=True,ignore_index=True)
            # 计算回测期各分位收益率
            temp=pd.merge(left=tmp2,right=tmp3,on='code',how='inner')
            temp['return']=(temp['close_y']-temp['close_x'])/temp['close_x']
            indices=np.linspace(0,len(tmp),6,dtype=int)
            Q_tmp=np.zeros((1,5))
            for i in range(5):
                group=tmp.loc[indices[i]:indices[i+1]]['code']
                ans=pd.merge(left=group,right=temp,on='code',how='inner')
                Q_tmp[0][i]=ans['return'].mean()
            Q=np.concatenate((Q,Q_tmp),axis=0)
        Q=Q[1:,:]
        Q_ans=Q.mean(axis=0)*100
        print("N:{} M:{}".format(N,M))
        for i in range(5):
            print("Q{}:{:.4f}%".format(i+1,Q_ans[i]),end=' ')
        print("\nQ5-Q1:{:.4f}%".format(Q_ans[4]-Q_ans[0]))