from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model,ensemble
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

year_end=19
#模型切换
model=linear_model.LinearRegression()
#保存路径
model_name=str(model)

data=pd.DataFrame()
for i in range(6):
    tmp=pd.read_csv("./Data/RESSET_MRESSTK_"+str(i+1)+".csv",encoding='gbk',usecols=range(13))
    tmp.columns=['code','date','close','vol','turn','shr','ret','rf','PE','PB','EPS','ROE','NAPS']
    tmp['date']=pd.to_datetime(tmp['date'])
    data=pd.concat((data,tmp))

data['yymm']=data['date'].dt.strftime("%y%m")
data_dict={}
grouped=data.groupby('code')
for name,group in grouped:
    data_dict[name]=group
tmp1=pd.read_csv("./Data/RESSET_MACHINACPI_1.csv",encoding='gbk',usecols=range(2))
tmp1.columns=['date','CPI']
tmp1['date']=pd.to_datetime(tmp1['date'])
tmp2=pd.read_csv("./Data/RESSET_MAINDUPPI_1.csv",encoding='gbk',usecols=range(2))
tmp2.columns=['date','PPI']
tmp2['date']=pd.to_datetime(tmp2['date'])
tmp=pd.merge(left=tmp1,right=tmp2,on='date',how='inner')
tmp['yymm']=tmp['date'].dt.strftime("%y%m")

fi=open('./Res/'+model_name+'.csv','w')
fi.write('code,')
for year in range(year_end,23):
    for month in range(1,13):
        fi.write('20{:0>2}-{:0>2},'.format(year,month))
fi.write("\n")

for code in data_dict:
    data=data_dict[code]
    data=pd.merge(left=data,right=tmp,on='yymm',how='inner')
    data['vol']=np.log(data['vol'])
    data['turn']=np.log(data['turn'])
    data['exret']=data['ret']-data['rf']
    data['ma']=data['close'].rolling(window=3).mean()
    data['tma']=data['ma'].rolling(window=3).mean()
    data['mbi']=(data['close']-data['ma'])/data['ma']
    data['S_K']=(data['close']-data['close'].rolling(window=12).min())/(data['close'].rolling(window=12).max()-data['close'].rolling(window=12).min())
    data['S_D']=data['S_K'].rolling(window=3).mean()
    data['PSY']=(data['ret']>0).rolling(window=12).mean()
    data.dropna(inplace=True)
    data.reset_index(inplace=True,drop=True)

    n_in = np.sum(data['yymm'] <= str(year_end)+'00')
    n_out = np.sum(data['yymm'] > str(year_end)+'00')
    if n_out!=(23-year_end)*12 or n_in<=120:
        continue
    rout = np.zeros(n_out)

    factors=['vol','turn','PE','PB','EPS','ROE','NAPS','ret','ma','tma','mbi','S_K','S_D','PSY','CPI','PPI']

    for i in range(n_out):
        X=data.loc[:n_in+i-2,factors].values
        Y=data.loc[1:n_in+i-1,'exret'].values

        scaler=StandardScaler()
        X=scaler.fit_transform(X)

        model.fit(X,Y)
        f = data.loc[n_in+i-1,factors].values.reshape(1,-1)
        f = scaler.transform(f)
        rout[i]=model.predict(f)[0]

    fi.write("{},".format(code))
    for j in rout:
        fi.write("{},".format(j))
    fi.write("\n")

    print('{} 预测成功!'.format(code))

fi.close()