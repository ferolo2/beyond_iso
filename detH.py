import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import H_mat
import math
import numpy as np
import time 
from numba import jit,njit,prange,autojit

@autojit
def detrange2(L,a,energies):
    res=[0.,0.,0.]
    lenergies= len(list(energies))

    for i in prange(lenergies):        
#       res[i]= np.linalg.det(H_mat.Hmat(energies[i],L,a,0.,0.,1.0,0.5))        
#        res[i]= np.linalg.det(H_mat.Hmat(energies[i],L,a,0.,0.,0.25,0.5))
        res[i]= np.linalg.det(H_mat.Hmat00(energies[i],L,a,0.,0.,0.5))

#       Hmat00(E,L,a0,r0,P0,alpha)
    return res


def myreal(arr):
    out=list(arr)
    for i in range(len(arr)):
        out[i] = arr[i].real
    return out


@autojit
def EWrange2(L,a,energies):
    res=[0.,0.,0.]
    lenergies= len(list(energies))


#    aux= np.linalg.eig(H_mat.Hmat(energies[0],L,a,0.,0.,0.25,0.5))
    
#    print(sorted(aux[0]))
#    print(aux[1])
        
#    exit()
    
    for i in prange(lenergies):        
#       res[i]= np.linalg.det(H_mat.Hmat(energies[i],L,a,0.,0.,1.0,0.5))        
#        res[i]= sorted(np.linalg.eig(H_mat.Hmat(energies[i],L,a,0.,0.,0.25,0.5))[0])[-1].real
        res[i]= sorted(np.linalg.eig(H_mat.Hmat00(energies[i],L,a,0.,0.,0.5))[0])[-1].real
        
#        print(res[i])
        
#        res[i]= np.linalg.det(H_mat.Hmat00(energies[i],L,a,0.,0.,0.5))

#       Hmat00(E,L,a0,r0,P0,alpha)
    return res






L=25
a=0.1
start = time.time()

energies= [3.0002440383032125, 3.0002440383734075, 3.0002440384436024] # [3.000204029622465, 3.0002352028624968, 3.000263761025286]# [3.0003351812227046, 3.0003351812227095, 3.0003351812227144] 


for i in range(4):
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





