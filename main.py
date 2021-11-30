import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir='results/mnist_cifar2'
with open(data_dir+'/values.pkl','rb') as f:
    r,h=pkl.load(f)

res=[]
for ds_name in r.keys():
    dists,tr_dists,dims=r[ds_name]
    interp_probs=np.mean(dists<1,axis=1)
    for ii,(d,p) in enumerate(zip(dims,interp_probs)):
        res.append([d,p,ds_name,np.mean(tr_dists[ii]<1)])
res=pd.DataFrame(res,columns=['dimension considered','interp prob','name','train interp prob'])
res=res.groupby(['dimension considered','name']).mean().reset_index()

for i,ds_name in enumerate(r.keys()):
    plt.subplot(1,2,i+1)
    ids=res['name']==ds_name
    plt.scatter(res[ids]['dimension considered'],res[ids]['interp prob'])
    plt.ylim(.96,1)
    plt.title(ds_name)
    plt.xlabel('dimension considered')
    if i==0:
        plt.ylabel('proportion of test set in ellipse')
