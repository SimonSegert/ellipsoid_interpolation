import pandas as pd
import numpy as np
from fit import *
import pickle as pkl
import os
import time

d_vals= np.arange(2,13)
N_vals=np.arange(10,82,2)
n_iters=50
test_pts_per_iter=500

eps=10**-3
max_iter=10**5

save_dir='results/gaussian3' #or None

res_gaussian=[]

START=time.time()
for ii in range(n_iters):
    for d in d_vals:
        for N in N_vals:
            if N<d+1:
                #the data lie in a proper affine subspace, so the minimal ellipse is degenerate
                continue
            X=np.random.randn(N,d)
            Xtest=np.random.randn(test_pts_per_iter,d)
            P,c,errs=MinVolEllipse(X,eps,max_iter=max_iter,delta0=0)
            thresh=1#np.max(dist_ellipsoid(X,P,c))
            p=np.mean(dist_ellipsoid(Xtest,P,c)<thresh)
            res_gaussian.append([d,N,p])
            print(res_gaussian[-1])
    print(f'iter {ii}')
    n_remaining=n_iters-1-ii
    time_per_iter=(time.time()-START)/(ii+1)
    time_remaining=n_remaining*time_per_iter
    print(f'est time remaining={time_remaining/3600} hr.' )


res_gaussian=pd.DataFrame(res_gaussian,columns=['d','N','p'])

if save_dir is not None:
    os.mkdir(save_dir)
    hparams=dict({'eps':eps,'max_iter':max_iter,'d_vals':d_vals,'N_vals':N_vals,'n_iters':n_iters,
                  'test_pts_per_iter':test_pts_per_iter,'name':'gaussian'})
    with open(save_dir+'/values.pkl','wb') as f:
        pkl.dump((res_gaussian,hparams),f)
