import pandas as pd
import numpy as np
from fit import *
import pickle as pkl
import os

d_vals= [5]#np.arange(5,15)
N_vals=[10]#np.arange(10,200,10)
n_iters=10#1000

eps=10**-4
max_iter=1#10**5

save_dir='results/test0' #or None

res_gaussian=[]

for _ in range(n_iters):
    for d in d_vals:
        for N in N_vals:
            X=np.random.randn(N,d)
            Xtest=np.random.randn(1,d)
            P,c,errs=MinVolEllipse(X,eps,max_iter=max_iter,delta0=0)
            thresh=np.max(dist_ellipsoid(X,P,c))
            p=np.mean(dist_ellipsoid(Xtest,P,c)<thresh)
            res_gaussian.append([d,N,p])

res_gaussian=pd.DataFrame(res_gaussian,columns=['d','N','p'])

if save_dir is not None:
    os.mkdir(save_dir)
    hparams=dict({'eps':eps,'max_iter':max_iter,'d_vals':d_vals,'N_vals':N_vals,'n_iters':n_iters,'name':'gaussian'})
    with open(save_dir+'/values.pkl','wb') as f:
        pkl.dump((res_gaussian,hparams),f)
