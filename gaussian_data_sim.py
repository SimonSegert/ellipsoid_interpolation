import pandas as pd
import numpy as np
from fit import *

d_vals= np.arange(5,15)
N_vals=np.arange(10,200,10)
n_iters=1000

res_gaussian=[]

for _ in range(n_iters):
    for d in d_vals:
        for N in N_vals:
            X=np.random.randn(N,d)
            Xtest=np.random.randn(1,d)
            P,c,errs=MinVolEllipse(X,10**-4,max_iter=10**5,delta0=0)
            thresh=np.max(dist_ellipsoid(X,P,c))
            p=np.mean(dist_ellipsoid(Xtest,P,c)<thresh)
            res_gaussian.append([d,N,p])

res_gaussian=pd.DataFrame(res_gaussian,columns=['d','N','p'])