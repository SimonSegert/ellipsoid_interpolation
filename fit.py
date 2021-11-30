import time
import numpy as np

def woodbury_update(Ainv,u,v):
    #compute (A+uv^t)^{-1} given A^{-1},u,v

    uu=np.dot(Ainv,u)
    vv=np.dot(Ainv,v)
    denom=1+np.sum(v*uu)
    return Ainv-np.outer(uu,vv)/(denom)




def MinVolEllipse(Pt, tolerance,max_iter=100,use_naive=False,delta0=0,verbose=False):
    #adapted from https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid,

    #original matlab code assumes that first array dimension is embedding dimension,
    P=np.copy(Pt).T
    d, N = P.shape
    Q = np.zeros((d+1,N))
    Q[0:d,:] = P[0:d,0:N]
    Q[d,:] = np.ones((1,N))

    count = 1
    err = 1
    u = (1/N) * np.ones((N,1))
    errs=[]
    START=time.time()
    pct_in_ellipse=0 #training samples inside of ellipse, used for stopping condition
    while True:
        if count==1 or use_naive:
            X=np.einsum('ji,i,ki->jk',Q,u.squeeze(),Q,optimize='optimal')+delta0*np.eye(len(Q))
            #equivalent to X = np.linalg.multi_dot((Q,np.diag(u.squeeze()), Q.T)),
            Xinv=np.linalg.inv(X)
            M=np.einsum('ij,ik,kj->j',Q,Xinv,Q,optimize='optimal')
            #equivalent to M = np.diag(np.linalg.multi_dot((Q.T,np.linalg.inv(X), Q))),

            #this matrix is maintained so that the partial quadratic forms can be quickly computed
            PuPt_inv=np.linalg.inv(P@np.diag(u.squeeze())@P.T)
            #PuPt'=aPuPt+bP_jP_j^t
            #PuPt_inv'=a^{-1}inv(PuPt+(b/a)P_jP_j^t)

        else:
            v=np.dot(Xinv,Q[:,j])
            a=1-step_size
            b=step_size
            g=b*np.sum(Q[:,j]*v)/a
            c=b*(a**-2)/(1+g)
            w=np.dot(Q.T,v)
            M=M/a-c*w*w
            #X=a*X+b*np.outer(Q[:,j],Q[:,j]),
            #we don't use the woodbury update function here, because the intermediate values are also used to update m
            Xinv=Xinv/a-c*np.outer(v,v)

            PuPt_inv=woodbury_update(PuPt_inv,(b/a)*P[:,j],P[:,j])/a


        maximum, j = np.max(M),np.argmax(M)
        step_size = (maximum - d-1 )/((d+1)*(maximum-1));
        new_u = (1 - step_size)*u
        new_u[j] = new_u[j] + step_size;
        err = np.linalg.norm(new_u - u)
        old_u=u
        u = new_u;
        if count==max_iter:
            print('max iterations reached')
            break
        if verbose and count%500==0:
            print(f'iter {count}, err={np.mean(errs[-50:])}')

        #M/(d+1) are the values of the quadratic form of the ellipse at each data points
        #cf. section 4 in Moshtagh paper
        if maximum/(d+1)<1+tolerance:
            break
        #if count%100==0:
            #U1 = np.diag(u.squeeze())
            #A1 = (1 / d) * np.linalg.inv(P @ U1 @ P.T - (P @ u) @ (P @ u).T)
            #c1=P@u
            #A1=(1/d)*woodbury_update(PuPt_inv,c1,-c1)
            #pct_in_ellipse=np.mean(dist_ellipsoid(Pt,A1,c1)<1+tolerance)
            #A1=(PUP^T)^{-1}+b*OP((PUP^t)^{-1}P@u)
        errs.append(err)
        count = count + 1;
    if verbose:
        print('time per iteration:')
        print((time.time()-START)/(count-1))
    U = np.diag(u.squeeze())
    c = P @ u
    A = (1 / d) * woodbury_update(PuPt_inv, c, -c)
    #A = (1/d) * np.linalg.inv(P @ U @ P.T - (P @ u)@(P@u).T )
    return A,c,np.array(errs)

def dist_ellipsoid(X,P,c):
    #X:array of points
    #ellipsoid is defined as all points such that (x-c)^TP(x-c)<=1
    Xc=X-c.squeeze()
    return np.einsum('ij,jk,ik->i',Xc,P,Xc,optimize='optimal')

def get_cropped_ds(ds,ds_test,padx,pady,ds_name,discard_const=False):
    #crop out images as in lecun paper
    if ds_name=='mnist':
        X_tr=ds.data.squeeze()
        X_tr=X_tr[:,padx:28-padx,pady:28-pady]
        X_tr=X_tr.reshape((len(X_tr),-1)).numpy()

        X_test=ds_test.data.squeeze()
        X_test=X_test[:,padx:28-padx,pady:28-pady]
        X_test=X_test.reshape((len(X_test),-1)).numpy()
    elif ds_name=='cifar10':
        X_tr=ds.data.squeeze()
        X_tr=X_tr[:,padx:32-padx,pady:32-pady]
        X_tr=X_tr.reshape((len(X_tr),-1))

        X_test=ds_test.data.squeeze()
        X_test=X_test[:,padx:32-padx,pady:32-pady]
        X_test=X_test.reshape((len(X_test),-1))

    #constant dimensions (e.g. pixels that are always black) cause instability in algorithm
    #(bc then the points lie on proper subspace)
    if discard_const:
        nonconst_dims=np.where(np.std(X_tr,axis=0)>10**-6)[0]
        X_tr=X_tr[:,nonconst_dims]
        X_test=X_test[:,nonconst_dims]
    return X_tr,X_test

