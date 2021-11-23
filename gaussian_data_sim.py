import pandas as pd
import numpy as np
from fit import *
import pickle as pkl
import os
import time

d_vals= np.arange(3,13)
N_vals=np.arange(10,200,10)
n_iters=1000

eps=10**-4
max_iter=10**5

save_dir='results/gaussian' #or None

res_gaussian=[]

START=time.time()
for ii in range(n_iters):
    for d in d_vals:
        for N in N_vals:
            X=np.random.randn(N,d)
            Xtest=np.random.randn(1,d)
            P,c,errs=MinVolEllipse(X,eps,max_iter=max_iter,delta0=0)
            thresh=np.max(dist_ellipsoid(X,P,c))
            p=np.mean(dist_ellipsoid(Xtest,P,c)<thresh)
            res_gaussian.append([d,N,p])
    print(f'iter {ii}')
    n_remaining=n_iters-1-ii
    time_per_iter=(time.time()-START)/(ii+1)
    time_remaining=n_remaining*time_per_iter
    print(f'est time remaining={time_remaining/3600} hr.' )


res_gaussian=pd.DataFrame(res_gaussian,columns=['d','N','p'])

if save_dir is not None:
    os.mkdir(save_dir)
    hparams=dict({'eps':eps,'max_iter':max_iter,'d_vals':d_vals,'N_vals':N_vals,'n_iters':n_iters,'name':'gaussian'})
    with open(save_dir+'/values.pkl','wb') as f:
        pkl.dump((res_gaussian,hparams),f)
