import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# 数据导入清洗
stock1=pd.read_excel("RESSET_DRESSTK_2001_2010_1.xls")
stock2=pd.read_excel("RESSET_DRESSTK_2011_2015_1.xls")
stock3=pd.read_excel("RESSET_DRESSTK_2016_2020_1.xls")
stock4=pd.read_excel("RESSET_DRESSTK_2021__1.xls")
index=pd.read_excel("RESSET_IDXQTTN_1.xls")

stock_data=pd.concat((stock1,stock2,stock3,stock4),axis=0)
stock_data.columns=['code','date','close','ret_rf']
stock_data.dropna(inplace=True)
stock_data.sort_values(by='code',inplace=True)
stk_codes=np.unique(stock_data['code'].values)
stock=[]

# 按照代码提取股票数据
for code in stk_codes:
    tmp=stock_data[stock_data['code']==code]
    tmp['date']=pd.to_datetime(tmp['date'])
    tmp.sort_values(by='date',inplace=True)
    tmp['return']=np.log(tmp['close'])-np.log(tmp['close'].shift(periods=1))
    tmp.dropna(inplace=True)
    ind=(tmp['return']>=-0.1)&(tmp['return']<=0.1)
    tmp=tmp[ind]
    stock.append(tmp)

# 构建市场因子以及无风险收益率
index.columns=['code','date','close','flag']
index['index']=np.log(index['close'])-np.log(index['close'].shift(periods=1))
index.dropna(inplace=True)
ind=(index['index']>=-0.1)&(index['index']<=0.1)
index=index[ind]
index=pd.merge(left=index[['date','index']],right=stock[0][['date','ret_rf']],on='date',how='inner')

# 对每支股票进行单资产检验
merge_data=index
for i in range(len(stock)):
    merge_data=pd.merge(left=merge_data,right=stock[i][['date','return']],on='date',how='inner')
    merge_data['return']-=merge_data['ret_rf']
    merge_data=merge_data.rename(columns={'return': stk_codes[i]})
    x=merge_data['index'].values
    x=sm.add_constant(x)
    y=merge_data[stk_codes[i]].values
    model=sm.OLS(y,x)
    results=model.fit()
    print(results.summary())