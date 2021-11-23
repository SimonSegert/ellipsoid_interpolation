import numpy as np
from fit import *
import torchvision
import os
import pickle as pkl

max_iter=10**5
eps=10**-4

max_dim=400
min_dim=10

save_dir='results/mnist_cifar' #or None

res=dict({})
for ds_name in ['mnist','cifar10']:
    if ds_name=='mnist':
        ds_loc='/home/simon/Desktop/Data/MNIST'
        ds=torchvision.datasets.MNIST(root=ds_loc,transform=torchvision.transforms.ToTensor(),train=True,
                                      download=True)
        ds_test=torchvision.datasets.MNIST(root=ds_loc,transform=torchvision.transforms.ToTensor(),train=False,
                                           download=True)

    elif ds_name=='cifar10':
        transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ds_loc='/home/simon/Desktop/Data/CIFAR'

        ds = torchvision.datasets.CIFAR10(root=ds_loc, train=True,
                                                download=True, transform=transform)
        ds_test = torchvision.datasets.CIFAR10(root=ds_loc, train=False,
                                                download=True, transform=transform)

    dists=[]
    tr_dists=[]
    dims=[]
    d=28 if ds_name=='mnist' else 32
    n_ch=1 if ds_name=='mnist' else 3
    #n_ch(d-2*crop)^2=ndims
    #2crop=d-(ndims/nch)**.5

    min_crop=int((d-(max_dim/n_ch)**.5)/2)

    max_crop=int((d-(min_dim/n_ch)**.5)/2)
    for cropx in range(max_crop,min_crop,-1):
        for cropy in range(max_crop,min_crop,-1):
            X_tr,X_test=get_cropped_ds(ds,ds_test,cropx,cropy,ds_name)
            X=X_tr
            P,c,errs=MinVolEllipse(X,eps,max_iter=max_iter,delta0=0)
            converged=len(errs)<max_iter

            tr_dist=dist_ellipsoid(X,P,c)

            test_dist=dist_ellipsoid(X_test,P,c)
            dists.append(test_dist)
            tr_dists.append(tr_dist)
            dims.append(X_tr.shape[1])
            #due to convergence issues, the fitted ellipse ends up being too small (some train points have
            #dist slightly larger than 1)
            #to correct for this, we allow test point
            print(ds_name,X_tr.shape[-1],np.mean(test_dist<np.max(tr_dist)))
    dists=np.array(dists)
    tr_dists=np.array(tr_dists)
    dims=np.array(dims)
    res[ds_name]=(dists,tr_dists,dims)

if not save_dir is None:
    os.mkdir(save_dir)
    hparams=dict({'max_iter':max_iter,'eps':eps,'max_dim':max_dim,'min_dim':min_dim,'name':'mnist/cifar'})
    with open(save_dir+'/values.pkl','wb') as f:
        pkl.dump((res,hparams),f)


