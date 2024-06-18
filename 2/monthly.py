import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
sys.dont_write_bytecode=True
from MultiAssets import MultiAssets

# 数据导入清洗
stock_data=pd.read_excel("RESSET_MRESSTK_1.xls")
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
    stock.append(tmp)

# 构建市场因子以及无风险收益率
index=pd.read_excel("RESSET_IDXQTTN_1.xls")
index.columns=['code','date','close','flag']
index=index.loc[index['flag']==1]
index['index']=np.log(index['close'])-np.log(index['close'].shift(periods=1))
index=pd.merge(left=index[['date','index']],right=stock[0][['date','ret_rf']],on='date',how='inner')

# 对所有股票进行多资产检验
merge_data=index
for i in range(len(stock)):
    merge_data=pd.merge(left=merge_data,right=stock[i][['date','return']],on='date',how='inner')
    merge_data['return']-=merge_data['ret_rf']
    merge_data=merge_data.rename(columns={'return': stk_codes[i]})

ret_ind=merge_data['index'].values
ret_stock=merge_data[stk_codes].values
MultiAssets(ret_ind,ret_stock)