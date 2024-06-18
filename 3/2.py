import numpy as np
import pandas as pd

# 排序法检验类
class sort_portfolio:
    
    def __init__(self, data, months, gnum):
        '''
        data:数据(stk,stksize,stkep,yearmonth,monexcret)
        months:月份
        gnum:组数
        '''
        self.data = data
        self.months = months
        self.gnum = gnum
    
    def data_months(self):
        dm = self.data.loc[self.data['yearmonth'] == self.months[0], ['stk', 'stksize', 'stkep']]
        dm.dropna(inplace = True)
        for i in range(1, len(self.months)):
            ind = self.data['yearmonth'] == self.months[i]
            dm = pd.merge(left = dm,
                         right = self.data.loc[ind, ['stk', 'monexcret']],
                         on='stk',
                         how='left',
                         sort=True,
                         suffixes=(str(i)+'_x',str(i)+'_y'))
        dm.columns = ['stk', 'size6', 'ep6', 'ret7', 'ret8', 'ret9', 'ret10', 'ret11', 
                      'ret12', 'retn1', 'retn2', 'retn3', 'retn4', 'retn5', 'retn6']
        return dm

    
    def sort_single_ind(self):
        L = np.sum(self.data['yearmonth'] == self.months[0])
        n = np.fix(L/self.gnum).astype(int)
        x = np.ones(L)
        i = 0
        while i < self.gnum:
            if i == self.gnum-1:
                x[i*n:] = x[i*n:]*i
            else:
                x[i*n:(i+1)*n] = x[i*n:(i+1)*n]*i
            i = i+1
        ssi = x.astype(int)
        return ssi
        
    def sort_double_ind(self):
        L = np.sum(self.data['yearmonth'] == self.months[0])
        l = np.fix(L/self.gnum).astype(int)
        n = np.fix(L/(self.gnum**2)).astype(int)
        x = np.ones(L)
        i = 0
        while i < self.gnum:
            j = 0
            while j < self.gnum:
                if j == self.gnum-1:
                    if i == self.gnum-1:
                        x[(i*l+j*n):] = x[(i*l+j*n):]*j
                    else:
                        x[(i*l+j*n):((i+1)*l)] = x[(i*l+j*n):((i+1)*l)]*j
                else:
                    x[(i*l+j*n):(i*l+(j+1)*n)] = x[(i*l+j*n):(i*l+(j+1)*n)]*j
                j=j+1
            i=i+1
        sdi = x.astype(int)
        return sdi
    
    def sequence_sort(self):
        dm = self.data_months()
        ssi = self.sort_single_ind()
        sdi = self.sort_double_ind()
        dm.sort_values(by=['size6'], ascending=True, inplace=True)
        dm['sinsort'] = ssi
        dm.sort_values(by=['sinsort', 'ep6'], ascending=[True, True], inplace=True)
        dm['dousort'] = sdi
        return dm
    
    
    def sequence_sort_mreturn(self):
        sp = self.sequence_sort()
        spmreturn = sp.loc[:, ['ret7', 'sinsort', 'dousort']].dropna().groupby(
                    by=['sinsort', 'dousort'])['ret7'].mean()
        lret = ['ret8', 'ret9', 'ret10', 'ret11', 'ret12', 'retn1', 'retn2', 'retn3', 'retn4', 'retn5', 'retn6']
        for i in lret:
            a = sp.loc[:, [i, 'sinsort', 'dousort']].dropna().groupby(
                by=['sinsort', 'dousort'])[i].mean()
            spmreturn = pd.concat([spmreturn, a], axis=1)
        spmreturn['mret'] = spmreturn.apply(lambda x: x.mean(), axis=1)
        return spmreturn

# 导入size和ep
data=pd.DataFrame()
for i in range(1,9):
    data_read=pd.read_csv('RESSET_MRESSTK_'+str(i)+'.csv',encoding='GBK',usecols=range(8))
    data=pd.concat([data,data_read])
data.columns=['stk','date','close','fshare','tshare','monret','monrf','pe']
data['date']=pd.to_datetime(data['date'])
data['yearmonth']=data['date'].dt.strftime("%Y%m")
data['stksize']=data['close']*data['fshare']
data['stkep']=1/data['pe']
data['monexcret']=data['monret']-data['monrf']
data.dropna(inplace=True,subset=['stksize','stkep'])
uym=np.unique(data['yearmonth'])

# sp=sort_portfolio(data,uym[5:258],5)

meanret = []
lcname = []
for i in range(5, 258, 12):
    if len(uym[i:i+13]) == 13:
        lcname.append(str(uym[i]))
        sp = sort_portfolio(data, uym[i:i+13], 5)
        spmreturn = sp.sequence_sort_mreturn()
        if len(meanret) == 0:
            meanret = spmreturn['mret']
        else:
            meanret = pd.concat([meanret, spmreturn['mret']], axis=1)
meanret.columns = lcname

meanret['meanreturn'] = meanret.apply(lambda x: x.mean(), axis=1)
a = meanret['meanreturn'].values.reshape((5,5))

print('{:>10s} {:>10s}, {:>10s}, {:>10s}, {:>10s}, {:>10s}'.format('', 'EP1', 'EP2', 'EP3', 'EP4', 'EP5'))
for i in range(5):
    print('{:>10s} {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}, {:10.5f}'.format('SIZE'+str(i+1), 
                                                                            a[i, 0], 
                                                                            a[i, 1], 
                                                                            a[i, 2], 
                                                                            a[i, 3], 
                                                                            a[i, 4]))