import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm,linregress

# .mat格式文件导入
matadata=loadmat("SSEC_min.mat")
matadata=matadata['p'][:,0]

# 定义时间间隔以及绘图颜色
freq_time=[1,5,10,30,60,120,240]
colors=['r','g','b','y','c','m','k']

def draw1(index):  # 画整体经验概率分布
    # 求收益率并清洗
    p_min=matadata[::freq_time[index]]
    r_min=np.log(p_min[1:])-np.log(p_min[:-1])
    r_min = r_min[(r_min>=-0.1)&(r_min<=0.1)]

    # 求经验概率分布
    bin=np.linspace(-0.1,0.1,101)
    x=np.zeros(len(bin)-1)
    y=np.zeros(len(bin)-1)
    for i in range(len(bin)-1):
        x[i]=(bin[i]+bin[i+1])/2
        y[i]=np.sum((r_min>=bin[i])&(r_min<bin[i+1]))/len(r_min)/(bin[i+1]-bin[i])
    ind=y>0
    x=x[ind]
    y=y[ind]

    plt.semilogy(x,y,'o-'+colors[index],label=str(freq_time[index])+'min',lw=1,ms=3)

for i in range(7):
    draw1(i)

plt.xlim([-0.1, 0.1])
plt.xticks(np.arange(-0.1,0.1,0.02))
plt.ylim([10.**-4, 10**3])
plt.yticks(10.**np.arange(-4,4,2))
plt.legend(loc='upper right', fontsize=10)
plt.xlabel("$r$",fontsize=20)
plt.ylabel("$p(r)$",fontsize=20)
plt.show()

def draw2(index):  # 画经验概率分布拟合
    # 求收益率并清洗
    p_min=matadata[::freq_time[index]]
    r_min=np.log(p_min[1:])-np.log(p_min[:-1])
    r_min = r_min[(r_min>=-0.1)&(r_min<=0.1)]

    # 求经验概率分布
    bin=np.linspace(-0.1,0.1,101)
    x=np.zeros(len(bin)-1)
    y=np.zeros(len(bin)-1)
    for i in range(len(bin)-1):
        x[i]=(bin[i]+bin[i+1])/2
        y[i]=np.sum((r_min>=bin[i])&(r_min<bin[i+1]))/len(r_min)/(bin[i+1]-bin[i])
    ind=y>0
    x=x[ind]
    y=y[ind]

    plt.semilogy(x,y,'o-r',label=str(freq_time[index])+'min')

    mu_min, sigma_min = norm.fit(r_min)
    x=np.linspace(-0.1, 0.1, 300)
    y=norm.pdf(x, loc=mu_min, scale=sigma_min)
    plt.semilogy(x,y,'k',label=str(freq_time[index])+'min_norm')
    plt.xlim([-0.1, 0.1])
    plt.xticks(np.arange(-0.1,0.1,0.02))
    plt.ylim([10.**-4, 10**3])
    plt.yticks(10.**np.arange(-4,4,2))
    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel("$r$",fontsize=20)
    plt.ylabel("$p(r)$",fontsize=20)
    plt.show()

for i in range(7):
    draw2(i)

def draw3(index):  # 画幂尾分布拟合
    # 求收益率并清洗
    p_min=matadata[::freq_time[index]]
    r_min=np.log(p_min[1:])-np.log(p_min[:-1])
    r_min = r_min[(r_min>=-0.1)&(r_min<=0.1)]

    # 求经验概率分布
    bin=np.logspace(-3,0,31)
    x=np.zeros(len(bin)-1)
    y=np.zeros(len(bin)-1)
    for i in range(len(bin)-1):
        x[i]=np.sqrt(bin[i]*bin[i+1])
        y[i]=np.sum((r_min>=bin[i])&(r_min<bin[i+1]))/len(r_min)/(bin[i+1]-bin[i])

    ind=y>0
    x=x[ind]
    y=y[ind]

    plt.loglog(x,y,'or',label=str(freq_time[index])+'min')

    # 拟合正尾
    a=np.log(x)
    b=np.log(y)
    slope, intercept, r_value, p_value, std_err = linregress(a,b)
    text1="$y=$"+"{:.4f}".format(slope)+"$x+$"+"{:.4f}".format(intercept)
    text2="$R^{2}$="+"{:.4f}".format(r_value**2)
    plt.loglog(x,(x**slope)*(np.exp(intercept)),'k',label=str(freq_time[index])+'min_pos')
    plt.text(.001,np.min(y),text1,fontsize=15)
    plt.text(.001,np.min(y)*10,text2,fontsize=15)

    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel("$r$",fontsize=20)
    plt.ylabel("$p(r)$",fontsize=20)
    plt.show()

    # 求经验概率分布
    bin=-np.logspace(-3,0,31)
    x=np.zeros(len(bin)-1)
    y=np.zeros(len(bin)-1)
    for i in range(len(bin)-1):
        x[i]=np.sqrt(bin[i]*bin[i+1])
        y[i]=np.sum((r_min<bin[i])&(r_min>=bin[i+1]))/len(r_min)/(bin[i]-bin[i+1])

    ind=y>0
    x=x[ind]
    y=y[ind]

    plt.loglog(x,y,'ob',label=str(freq_time[index])+'min')

    # 拟合负尾
    a=np.log(x)
    b=np.log(y)
    slope, intercept, r_value, p_value, std_err = linregress(a,b)
    text1="$y=$"+"{:.4f}".format(slope)+"$x+$"+"{:.4f}".format(intercept)
    text2="$R^{2}$="+"{:.4f}".format(r_value**2)
    plt.loglog(x,(x**slope)*(np.exp(intercept)),'k',label=str(freq_time[index])+'min_neg')
    plt.text(.001,np.min(y),text1,fontsize=15)
    plt.text(.001,np.min(y)*10,text2,fontsize=15)

    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel("$r$",fontsize=20)
    plt.ylabel("$p(r)$",fontsize=20)
    plt.show()

for i in range(7):
    draw3(i)