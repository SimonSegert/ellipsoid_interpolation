import time
import numpy as np


def MinVolEllipse(P, tolerance,max_iter=100,use_naive=False,delta0=0,verbose=False):
    #adapted from https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid,

    #original matlab code assumes that first array dimension is embedding dimension,
    P=P.T
    d, N = P.shape
    Q = np.zeros((d+1,N))
    Q[0:d,:] = P[0:d,0:N]
    Q[d,:] = np.ones((1,N))

    count = 1
    err = 1
    u = (1/N) * np.ones((N,1))
    errs=[]
    START=time.time()
    while err > tolerance:
        #note: this can still be optimized considerably,
        #because u1=au0+be_i (where e_i is standard basis vector),
        #so X1=aX0+b Q[:,i]Q[:,i]^t (rank 1 update),
        #we can calculate X1^{-1} by rank 1 update to (aX0)^{-1},
        #https://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices,
        #the rank 1 part is (aX0)^{-1} bQ[:,i]Q[:,i]^t (aX0)^{-1}=ba^{-2}*X0^{-1}Q[:,i] \\otimes (itself),
        #(note that X is a symmetric matrix),

        #we also need g=Tr((aX0)^{-1}bQ_iQ_i^t)=bQ_i^tX_0^{-1}Q[i]/a,

        #so we can update the value of X and X^{-1} without the new einsum,
        #X_1^{-1}=a^{-1}X0^{-1}-ba^{-2}OP(X0^{-1}Q[i])/(1+g),

        #set c=ba^{-2}/(1+g), then,
        #we have M1=a^{-1}M0-c diag(QX0^{-1}Q[i] x (transpose)),
        #in general, diag(vw^t)=v (pointwise prod) w,
        #thus we only need to compute QX0^{-1}Q[i],
        if count==1 or use_naive:
            X=np.einsum('ji,i,ki->jk',Q,u.squeeze(),Q,optimize='optimal')+delta0*np.eye(len(Q))
            #equivalent to X = np.linalg.multi_dot((Q,np.diag(u.squeeze()), Q.T)),
            Xinv=np.linalg.inv(X)
            M=np.einsum('ij,ik,kj->j',Q,Xinv,Q,optimize='optimal')
            #equivalent to M = np.diag(np.linalg.multi_dot((Q.T,np.linalg.inv(X), Q))),
        else:
            #u=(1-step_size) old_u+step_size*e_j,
            #rank 1 updates, much faster than naive method,
            #(note that np.dot is much faster than np.solve,
            #so updating Xinv is much faster than updating X and then solving the system of eqns),
            #v=np.linalg.solve(X,Q[:,j]),
            v=np.dot(Xinv,Q[:,j])
            a=1-step_size
            b=step_size
            g=b*np.sum(Q[:,j]*v)/a
            c=b*(a**-2)/(1+g)
            w=np.dot(Q.T,v)
            M=M/a-c*w*w
            #X=a*X+b*np.outer(Q[:,j],Q[:,j]),
            Xinv=Xinv/a-c*np.outer(v,v)


        maximum, j = np.max(M),np.argmax(M)
        step_size = (maximum - d-1 )/((d+1)*(maximum-1));
        new_u = (1 - step_size)*u
        new_u[j] = new_u[j] + step_size;
        count = count + 1;
        err = np.linalg.norm(new_u - u)
        old_u=u
        u = new_u;
        if count==max_iter:
            print('max iterations reached')
            break
        if verbose and count%500==0:
            print(f'iter {count}, err={np.mean(errs[-50:])}')
        errs.append(err)
    if verbose:
        print('time per iteration:')
        print((time.time()-START)/(count-1))
    U = np.diag(u.squeeze())
    A = (1/d) * np.linalg.inv(P @ U @ P.T - (P @ u)@(P@u).T )
    c = P @ u
    return A,c,np.array(errs)

def dist_ellipsoid(X,P,c):
    #X:array of points
    #ellipsoid is defined as all points such that (x-c)^TP(x-c)<=1
    Xc=X-c.squeeze()
    return np.einsum('ij,jk,ik->i',Xc,P,Xc,optimize='optimal')

def get_cropped_ds(ds,ds_test,padx,pady,ds_name):
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

    return X_tr,X_test

