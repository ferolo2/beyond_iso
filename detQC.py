import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import H_mat,F3_mat
import math
import numpy as np
import time
import projections as pr
from numba import jit,njit,prange,autojit
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


def F3i(e,L,a0,a2):
    f3mat= F3_mat.F3i_mat(e,L,a0,0.,0.,a2,0.3)
    return pr.irrep_proj(f3mat,e,L,"A1")

def H2(e,L,a,a2):
    return H_mat.Hmat(e,L,a,0.,0.,a2,0.3)
def H0(e,L,a,a2):
    return H_mat.Hmat00(e,L,a,0.,0.,0.3)

def EWrange2(L,a,a2,energies):
    res=np.zeros(len(energies))
    lenergies= len(list(energies))
    pool = Pool(processes=4)

    aux = []
    for i in range(lenergies):
        aux.append((energies[i],L,a,a2))

    res=[]    
#    print(aux)
#    exit()
    
#    result = pool.starmap(F3i, [(energies[0],L,a,a2),(energies[1],L,a,a2),(energies[2],L,a,a2)])
    result = pool.starmap(F3i, aux)
#    result = pool.starmap(H0, [(energies[0],L,a,a2),(energies[1],L,a,a2),(energies[2],L,a,a2)])
    for i in prange(lenergies):
#        print(np.linalg.eig(result[i])[0])
#        res[i] = sorted(np.linalg.eig(result[i])[0])[-1].real  
        resaux = sorted(np.linalg.eig(result[i])[0])
#        print(resaux)
#        res[i] = np.prod(resaux[-4:1]).real
#        res[i] = np.prod(resaux[0:2]).real
        res.append([energies[i],resaux[0].real,resaux[1].real])
        #print("EV ",resaux[0:2])
    return res


def Ethres(L,a):
    I = -8.914
    J = I**2+16.532
    E3 = 12*math.pi*a/L**3*(1 - (a/math.pi/L)*I +(a/math.pi/L)**2*J )
    return E3




L=5.6
a=0.1
a2=0.1
Etest = 3+Ethres(L,a)
dEtest = Ethres(L,a)*0.2
Etest = 2*np.sqrt(1+(2*math.pi/L)**2)+1-0.00002

Etest = 4.00591
dEtest = 0.000041/1600
nmax=240
#Etest = 3.000059325557361  + 1e-12
#dEtest = 2e-12



start = time.time()

energies= [Etest-dEtest, Etest, Etest+dEtest] # [3.000213837, 3.000216951408889, 3.000220065817778]
#energies = [3.000088646152035, 3.000088646154394, 3.000088646156753]
#energies = [3.0001409580067375, 3.0001409580069174, 3.0001409580070972]
#energies = [3.000140957589557, 3.0001409580070972, 3.0001409584246375]


energies = []
for i in range(0,nmax):
    energies.append(Etest+i*dEtest)


data = np.array(EWrange2(L,a,a2,energies))

EN = data[:,0]
EV1 = data[:,1]
EV2 = data[:,2]

xnew = np.linspace(Etest, Etest + (nmax-1)*dEtest, num=100, endpoint=True)
f = interp1d(EN, EV1)
f2 = interp1d(EN, EV1, kind='quadratic')
f3 = interp1d(EN, EV1, kind='cubic')
fb = interp1d(EN, EV2)
f2b = interp1d(EN, EV2, kind='quadratic')
f3b = interp1d(EN, EV2, kind='cubic')
f4b = interp1d(EN, EV2, kind=5)
f5b = interp1d(EN, EV2, kind=7)

import matplotlib.pyplot as plt
#plt.plot(EN, EV1, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--',EN,EV2,'o',xnew, fb(xnew), '-', xnew, f2b(xnew), '--')
plt.plot(EN,EV2,'o',xnew, fb(xnew), '-', xnew, f2b(xnew), '--')
plt.legend(['data', 'linear', 'cubic','data2', 'linear2', 'cubic2'], loc='best')

print('sol from linear E = %.12f' % fsolve(fb,Etest)[0],'other sols',fsolve(fb,Etest))
print('sol from quad E = %.12f' % fsolve(f2b,Etest)[0],'other sols',fsolve(f2b,Etest))
print('sol from cubic E = %.12f' % fsolve(f3b,Etest)[0],'other sols',fsolve(f3b,Etest))
print('sol from 5th E = %.12f' % fsolve(f4b,Etest)[0],'other sols',fsolve(f4b,Etest))
print('sol from 7th E = %.12f' % fsolve(f5b,Etest)[0],'other sols',fsolve(f5b,Etest))

plt.show()


exit()




print(Ethres(L,a))



for i in range(2):
    print(energies)
    res = EWrange2(L,a,a2,energies)
    print(res)
    resaux=res
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





