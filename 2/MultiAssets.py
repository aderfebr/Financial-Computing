import pandas as pd
import numpy as np
from scipy.stats import chi2,f

# 单因子多资产检验
def MultiAssets(ret_ind,ret_stock):
    # ret_ind: T*1
    # ret_stock: T*N
    T=ret_ind.shape[0]
    N=ret_stock.shape[1]
    mu_ind=np.mean(ret_ind)
    sigma_ind=np.sum((ret_ind-mu_ind)**2)/T

    x=np.ones((T,2))
    x[:,1]=ret_ind
    y=ret_stock
    xTx=np.dot(x.T,x)
    xTy=np.dot(x.T,y)
    AB_hat=np.dot(np.linalg.inv(xTx),xTy)
    alpha=AB_hat[0]
    beta=AB_hat[1]
    res=y-np.dot(x,AB_hat)
    cov=np.dot(res.T,res)/T
    invcov=np.linalg.inv(cov)

    xr=np.ones((T,1))
    xr[:,0]=ret_ind
    yr=ret_stock
    xrTxr=np.dot(xr.T,xr)
    xrTyr=np.dot(xr.T,yr)
    ABr_hat=np.dot(np.linalg.inv(xrTxr),xrTyr)
    resr=yr-np.dot(xr,ABr_hat)
    covr=np.dot(resr.T,resr)/T
    invcovr=np.linalg.inv(covr)

    trans_alpha=np.ones((len(alpha),1))
    trans_alpha[:,0]=alpha
    SWchi2=T*(1/(1+mu_ind**2/sigma_ind))*np.dot(np.dot(alpha,invcov),trans_alpha)
    SWF=(T-N-1)/N*(1/(1+mu_ind**2/sigma_ind))*np.dot(np.dot(alpha,invcov),trans_alpha)
    p_SWchi2=1-chi2.cdf(SWchi2[0],N)
    p_SWF=1-f.cdf(SWF[0],N,T-N-1)
    print("SWchi2: {:.4f} ({:.4f})".format(SWchi2[0],p_SWchi2))
    print("SWF: {:.4f} ({:.4f})".format(SWF[0],p_SWF))

    SLRchi2=T*(np.log(np.linalg.det(covr))-np.log(np.linalg.det(cov)))
    p_SLRchi2=1-chi2.cdf(SLRchi2,N)
    print("SLRchi2: {:.4f} ({:.4f})".format(SLRchi2,p_SLRchi2))

    a=np.zeros((N,1))
    a[:,0]=np.sum(resr,axis=0)
    salpha=np.dot(invcovr,a)
    b=np.dot(ret_ind,resr)
    sbeta=np.zeros((N,1))
    sbeta[:,0]=np.dot(invcovr,b)
    score=np.concatenate((salpha,sbeta),axis=0)

    a=np.concatenate((invcovr*T,invcovr*np.sum(ret_ind)),axis=1)
    b=np.concatenate((invcovr*np.sum(ret_ind),invcovr*np.sum(ret_ind**2)),axis=1)
    info=np.concatenate((a,b),axis=0)
    SLMchi2=np.dot(np.dot(score.T,np.linalg.inv(info)),score)
    p_SLMchi2=p_SLRchi2=1-chi2.cdf(SLMchi2[0][0],N)
    print("SLMchi2: {:.4f} ({:.4f})".format(SLMchi2[0][0],p_SLMchi2))