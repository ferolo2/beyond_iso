import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import H_mat
from F3 import F2_alt as F2
from F3 import K2i_mat as K2
from F3 import Gmatrix as Gm
import defns
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







L=18
E=3.0007
nnk = [1,2,3]
alpha=0.5

#Fmat_k(E,L,nnk,alpha)

#Gt = Gmatrix.Gmat(E,L)


start = time.time()
#shell_nnk_list
print(defns.shell_nnk_list(nnk))
#print(F2.Fmat_shell(E,L,nnk,alpha))
#print(np.shape(F2.Fmat_shell(E,L,nnk,alpha)))

#print(sums.sum_nnk(E,L,nnk,2,0,2,0,alpha))
end = time.time()
print('time is:', end - start, ' s')






