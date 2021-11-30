import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir='results/gaussian2'
with open(data_dir+'/values.pkl','rb') as f:
    r,h=pkl.load(f)

r=r[r['N']>1.1*r['d']]
r=r.groupby(['d','N']).mean().reset_index()

#for each dimension, the minimal dataset size needed for an interpolation prob of .5
critical_Ns=[]

for d in np.unique(r['d']):
    ids=r['d']==d
    plt.plot(r[ids]['N'],r[ids]['p'],label=f'd={d}')

    #compute critical N
    #first make sure that the sizes are in sorted order
    idsN=np.argsort(r[ids]['N'].values)
    Ns=r[ids]['N'].values[idsN]
    ps=r[ids]['p'].values[idsN]

    critical_N=Ns[np.where(ps>.5)[0][0]]
    critical_Ns.append(critical_N)
plt.axhline(.5,c='black',linestyle='--')
plt.xlabel('N')
plt.ylabel('interp prob')
plt.legend()
plt.savefig('results/gaussian/curves.pdf')

plt.close()
plt.scatter(np.unique(r['d'])[2:],critical_Ns[2:])
plt.xlabel('d')
plt.ylabel('critical N')
plt.savefig('results/gaussian/criticalN.pdf')
