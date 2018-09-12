import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from matplotlib import pyplot as plt
from ast import literal_eval

from scipy.linalg import eig

import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')
#sys.path.insert(0,cwd)

from K3df import K3A, K3B, K3quad
from F3 import F2_alt, Gmatrix, sums_alt as sums, K2i_mat
from F3 import H_mat, F3_mat
import defns, projections as proj, analysis_defns as AD, group_theory_defns as GT


# Eigenvalue projection analysis
E=3.2; L=5.0; alpha=0.5
a0=1e-8; r0=0; P0=0; a2=0.5


#K2i = K2i_mat.K2inv_mat(E,L,a0,r0,P0,a2)
#K2i_00 = K2i_mat.K2inv_mat00(E,L,a0,r0,P0)
#K2i_22 = K2i_mat.K2inv_mat22(E,L,a2)

#F2 = F2_alt.Fmat(E,L,alpha)
#F2_00 = F2_alt.Fmat00(E,L,alpha)
#F2_22 = F2_alt.Fmat22(E,L,alpha)

G = Gmatrix.Gmat(E,L)
G00 = Gmatrix.Gmat00(E,L)
G22 = Gmatrix.Gmat22(E,L)

#print(np.amax(abs(proj.l0_proj(F2)-F2_00)))
#print(np.amax(abs(proj.l0_proj(G)-G00)),np.amax(abs(proj.l2_proj(G)-G22)),np.amax(abs(G)))


M = G; M00=G00; M22=G22


N = len(defns.list_nnk(E,L))
if N != int(len(M)/6):
  print('wtf man')
for i in range(N):
  for j in range(N):
    for di in range(0,5):
      for dj in range(0,5):
        if abs(M22[5*i+di,5*j+dj] - M[6*i+di+1,6*j+dj+1] > 1e-10):
          #print(((i,di),(j,dj)),M22[5*i+di,5*j+dj],M[6*i+di+1,6*j+dj+1])
          pass




#S = np.zeros((95,0))
#for I in GT.irrep_list():
#  S = np.concatenate((S,proj.P_irrep_subspace_22(E,L,I)),axis=1)
#print(np.amax(abs(S.T@S-np.identity(95))))


I = 'T1-'
P = proj.P_irrep_22(E,L,I)
Psub = proj.P_irrep_subspace_22(E,L,I)
M = F3_mat.F3mat22(E,L,a0,r0,P0,a2,alpha)
Mi = defns.chop(LA.inv(M))


a=np.amax(abs(M@P - P@M)); b=np.mean(abs(M))
c=np.amax(abs(Mi@P - P@Mi)); d=np.mean(abs(Mi))
print(a,b,a/b)
print(c,d,c/d)


#eigs = sorted(defns.chop(LA.eigvals(Mi)))
#eigs1 = sorted(defns.chop(LA.eigvals(defns.chop(P@Mi@P)).real))
#eigs2 = sorted(LA.eigvals(defns.chop(Psub.T@Mi@Psub)).real)

#eigs3 = sorted(LA.eigvals(defns.chop(LA.inv(Psub.T@M@Psub))).real)

#print([e for e in eigs1 if abs(e)>1e-10],'\n')
#print(eigs2,'\n')
#print(eigs3,'\n')

#print([e for e in eigs if round(e/100) in [round(i/100) for i in eigs2]])





