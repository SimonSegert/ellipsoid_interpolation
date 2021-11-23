import pickle as pkl
import numpy as np
import pandas as pd

'''
data_dir='results/test0'
with open(data_dir+'/values.pkl','rb') as f:
    r,h=pkl.load(f)
print(r.head())
'''
data_dir='results/mnist_cifar'
with open(data_dir+'/values.pkl','rb') as f:
    r,h=pkl.load(f)
print(r['mnist'][0])


