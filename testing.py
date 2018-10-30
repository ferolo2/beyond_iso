import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
from F3 import H_mat
from F3 import sums_alt as sums
from F3 import F2_alt as F2
from F3 import K2i_mat as K2
from F3 import Gmatrix as Gm
import defns
import math
import numpy as np
import time
from numba import jit,njit,prange,autojit,vectorize
from F3 import sums_alt as sums
from scipy.special import erfc,erfc
from scipy.optimize import fsolve
from joblib import Parallel, delayed
npsqrt = np.sqrt
exp = np.exp
def getnmax(cutoff,alpha,x2,gamma):
    eqn = lambda l : -cutoff + 2*math.pi*npsqrt(math.pi/alpha) * exp(alpha*x2)*erfc(npsqrt(alpha)*l)
    n0=1
    solution=fsolve(eqn, n0)
    print(max(solution*gamma,1)+3)

    return int(np.round(max(solution*gamma,1)+3))

L=20
E=3.01
nnk = np.array([1,2,3])
alpha=0.5
gamma = sums.gam(E,npsqrt(1+4+9)*2*math.pi/L)
x2= sums.xx2(E, L, npsqrt(1+4+9)*2*math.pi/L)
cutoff=1e-9
#Fmat_k(E,L,nnk,alpha)
#Gt = Gmatrix.Gmat(E,L)
start = time.time()
Parallel(n_jobs=4)(delayed(sums.sum_nnk)(E, L,nnk,2,0,2,0,alpha) for i in range(1000))
end = time.time()
print('time is:', end - start, ' s')
start = time.time()
for i in range(1000):
    sums.sum_nnk(E, L,nnk,2,0,2,0,alpha)
end = time.time()
print('time is:', end - start, ' s')
