import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 评论按照日期分组
txt=pd.read_csv("data_guba_cjwl.csv")
txt['日期']=pd.to_datetime(txt['日期']).dt.strftime('%y%m%d')
grouped=txt.groupby('日期')

f=open("emo.csv","w",encoding='utf-8')
f.write("day,score\n")

# 基于关键词法构建情绪因子
for name,group in grouped:
    name=str(name)
    w1=group['阅读量'].values
    w2=group['回复数'].values*100
    w=w1+w2
    factor=[]
    for emo in group['评论']:
        emo=str(emo)
        score=0
        score+=emo.count('涨')+emo.count('赚')+emo.count('加仓')+emo.count('补仓')
        score-=2*(emo.count('跌')+emo.count('亏')+emo.count('垃圾')+emo.count('清仓'))
        factor.append(score)
    factor=np.array(factor)
    f.write('{},{}\n'.format('20'+name[:2]+'-'+name[2:4]+'-'+name[4:],np.dot(w1.T,factor)))

f.close()