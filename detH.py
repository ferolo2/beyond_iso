import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import H_mat
import math
import numpy as np
import time
from numba import jit,njit,prange,autojit
from multiprocessing import Pool

@autojit
def detrange2(L,a,energies):
    res=[0.,0.,0.]
    lenergies= len(list(energies))

    pool = Pool(processes=3)

    result = pool.map(H_mat.Hmat, energies, [L,L,L],[a,a,a],[0,0,0],[0,0,0],[0.25,0.25,0.25],[0.5,0.5,0.5])

    for i in prange(lenergies):
#       res[i]= np.linalg.det(H_mat.Hmat(energies[i],L,a,0.,0.,1.0,0.5))
#        res[i]= np.linalg.det(H_mat.Hmat(energies[i],L,a,0.,0.,0.25,0.5))
#        res[i]= np.linalg.det(H_mat.Hmat00(energies[i],L,a,0.,0.,1.0))
        res[i] = result[i]

#       Hmat00(E,L,a0,r0,P0,alpha)
    return res


def myreal(arr):
    out=list(arr)
    for i in range(len(arr)):
        out[i] = arr[i].real
    return out

def H2(e,L,a):
    return H_mat.Hmat(e,L,a,0.,0.,0.25,0.5)
def H0(e,L,a):
    return H_mat.Hmat00(e,L,a,0.,0.25,0.5)

@autojit
def EWrange2(L,a,energies):
    res=[0.,0.,0.]
    lenergies= len(list(energies))
    pool = Pool(processes=3)
    result = pool.starmap(H2, [(energies[0],L,a),(energies[1],L,a),(energies[2],L,a)])
#    result = pool.starmap(H0, [(energies[0],L,a),(energies[1],L,a),(energies[2],L,a)])


    for i in prange(lenergies):
#        res[i]= sorted(np.linalg.eig(H_mat.Hmat(energies[i],L,a,0.,0.,0.25,0.5))[0])[-1].real
#        res[i]= sorted(np.linalg.eig(H_mat.Hmat00(energies[i],L,a,0.,0.,0.5))[0])[-1].real
        res[i] = sorted(np.linalg.eig(result[i])[0])[-1].real

    return res






L=27
a=0.1
Etest = 3.0003
start = time.time()

#energies= [3.000193448341778, 3.000193562184019, 3.0001935760262595]  # [3.000213837, 3.000216951408889, 3.000220065817778]
energies= [Etest-0.00001, Etest, Etest+0.00001] # [3.000213837, 3.000216951408889, 3.000220065817778]
energies = [3.0001935599880523, 3.0001935621675373, 3.0001935643470223]
#energies = [3.000216852878521, 3.0002168542293832, 3.0002168555802453]

for i in range(3):
    print(energies)
    res = EWrange2(L,a,energies)
    print(res)
    resaux=np.array(res)/max(res)

    print(resaux)

    if(res[0]>res[1]):
        resaux=np.dot(-1.,np.array(res))/max(res)
    Ener=np.interp(0,resaux,energies)

    print('E=%.12f' % Ener )


    if(res[0]/res[1]<0):
        energies[2]=energies[1]
        energies[1]=Ener

    elif(res[1]/res[2]<0):
        energies[0]=energies[1]
        energies[1]=Ener
    else:
        exit()

    dEner = min([abs(energies[0]-Ener),abs(energies[2]-Ener)])

    energies[0] = Ener-dEner
    energies[2] = Ener+dEner

    print(energies)


end = time.time()
print('time is:', end - start, ' s')
