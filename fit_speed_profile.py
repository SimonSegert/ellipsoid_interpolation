import numpy as np
import cvxpy as cp
import time
import convhull
#validation of the optimization algorithm-make sure that it attains comparable results to cvx

test_fp='/Users/simon/Documents/GitHub/ellipsoid_interpolation/testH.csv'
tr_fp='/Users/simon/Documents/GitHub/ellipsoid_interpolation/trH.csv'

def validate_solution(X,y,yproj,eps=10**-5):
    #checks that the vector dy from point to projection is orthogonal to a face, (assumes that y is not in hull)
    #this can be detected by checking that (dy,x_i)=(dy,x_j) for at least one pair x_i\neq x_i
    #returns number of points that have same dot product w dy (so return value >1 indicates the condition is satisfied)
    dy=y-yproj
    dp=np.sort(np.abs(np.diff(np.sort(np.dot(X,dy)))))
    return np.sum(dp<eps)+1


print('loading data...')
X=np.loadtxt(tr_fp,delimiter=',')
Y=np.loadtxt(test_fp,delimiter=',')
print('done')


cvx_times=[]
cvx_obj=[]
mirror_times=[]
mirror_obj=[]
mirror_etas=[]
hyperplane_times=[]
hyperplane_obj=[]

n_pts=50000
use_cvx=False
for i in range(10):
    y = Y[i]  # .5*(X[0]+X[2])

    if use_cvx:
        Z=cp.Variable(n_pts)
        obj=cp.sum_squares(X[:n_pts].T@Z-y)
        constraints=[Z>=0,cp.sum(Z)==1]

        prob = cp.Problem(cp.Minimize(obj),constraints)
        START=time.time()
        prob.solve(solver=cp.SCS,eps=10**-5)
        elapsed=time.time()-START
        proj=X[:n_pts].T@Z.value

        face_size=validate_solution(X[:n_pts],y,proj)
        print(f'cvxpy:')
        print(f'n_pts={n_pts},time={elapsed},obj={obj.value},face size={face_size}')
        cvx_times.append(elapsed)
        cvx_obj.append(obj.value)

    START=time.time()
    obj=float('inf')
    proj=None
    for eta in [.1,.3]:#[.01,.03,.1,.3,1,3]:
        new_proj=convhull.closest_pt_hull(X[:n_pts], y, eta=eta, n_iters=250, eps=None)
        new_obj=np.sum((new_proj-y)**2)
        if new_obj<obj:
            proj=new_proj
            obj=new_obj
            opt_eta=eta
    elapsed=time.time()-START
    face_size=validate_solution(X[:n_pts],y,proj)
    print(f'mirror:')
    print(f'n_pts={n_pts},time={elapsed},obj={obj},face size={face_size},opt eta={opt_eta}')
    mirror_times.append(elapsed)
    mirror_obj.append(obj)
    mirror_etas.append(opt_eta)

    #cvxpy to solve the dual "svm" formulation of the problem (note this assumes that y is not in the hull)
    Z=cp.Variable(64)
    b=cp.Variable(1)
    obj=cp.sum_squares(Z)
    constraints=[X[:n_pts]@Z+b>=1,cp.sum(Z*y)+b<=-1]
    prob = cp.Problem(cp.Minimize(obj),constraints)
    START=time.time()
    prob.solve(solver=cp.SCS)
    elapsed=time.time()-START
    dist=(2/np.linalg.norm(Z.value))**2
    print(f'hyperplanes:')
    print(f'time={elapsed},obj={dist}')
    hyperplane_times.append(elapsed)
    hyperplane_obj.append(dist)



cvx_times = np.array(cvx_times)
cvx_obj = np.array(cvx_obj)
mirror_times = np.array(mirror_times)
mirror_obj = np.array(mirror_obj)
mirror_etas = np.array(mirror_etas)
hyperplane_obj=np.array(hyperplane_obj)
hyperplane_times=np.array(hyperplane_times)

print(f'cvx time ={cvx_times.mean()} +- {cvx_times.std()}')
print(f'mirror time ={mirror_times.mean()} +- {mirror_times.std()}')

print(f'percentage difference on objective ={100*(np.mean(mirror_obj/cvx_obj-1))}')

