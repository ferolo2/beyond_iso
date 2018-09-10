import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import H_mat
from F3 import F2_alt as F2
from F3 import K2i_mat as K2
from F3 import Gmatrix as Gm
import math
import numpy as np
import time 
from numba import jit,njit,prange,autojit
from F3 import sums_alt as sums


@autojit
def detrange2(L,a,energies):
    res=[0.,0.,0.]
    lenergies= len(list(energies))

    for i in prange(lenergies):        
       res[i]= np.linalg.det(H_mat.Hmat(energies[i],L,a,0.,0.,1.0,0.5))        
    return res







L=5.2
E=3.0282
nnk = np.array([0,0,0])
alpha=0.5

#Gt = Gmatrix.Gmat(E,L)
print(np.diag(Gm.Gmat(E,L)))
print(np.linalg.det(Gm.Gmat(E,L)))


start = time.time()
print(sums.sum_nnk(E,L,nnk,2,0,2,0,alpha))
end = time.time()
print('time is:', end - start, ' s')






start = time.time()
#print(sums.sum_000(E,L,nnk,2,0,2,0,alpha))
print(sums.sum_000(E,L,2,0,2,0,alpha))
end = time.time()
print('time is:', end - start, ' s')





