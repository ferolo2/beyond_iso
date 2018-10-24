import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
#from matplotlib import pyplot as plt
#from ast import literal_eval

from scipy.linalg import block_diag

import os, sys, time, pickle
cwd = os.getcwd()
sys.path.insert(0,cwd+'/F3')
sys.path.insert(0,cwd+'/K3df')

from K3df import K3A, K3B, K3quad
from F3 import F2_alt, Gmatrix, sums_alt as sums, K2i_mat
from F3 import H_mat, F3_mat
import defns, projections as proj, analysis_defns as AD, group_theory_defns as GT

alpha = 0.5
a0 = 0.1; r0=0; P0=0; a2=0.3
K0=0; K1=0; K2=0; A=0; B=1e2
L = 6
#E = 3.89594387 # E1=3.8959438608040386  
E = 4.57393085 # E2=4.573930845701579
I = 'A1+'

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

#####################################################
# Numerical checks

if a2==0:
  F = F2_alt.Fmat00(E,L,alpha)
  G = Gmatrix.Gmat00(E,L)
  K2i = K2i_mat.K2inv_mat00(E,L,a0,r0,P0)
  K3 = proj.l0_proj(K3quad.K3mat(E,L,K0,K1,K2,A,B,'r'))
  P = proj.P_irrep_subspace_00(E,L,I)
elif a0==0:
  F = F2_alt.Fmat22(E,L,alpha)
  G = Gmatrix.Gmat22(E,L)
  K2i = K2i_mat.K2inv_mat22(E,L,a2)
  K3 = proj.l2_proj(K3quad.K3mat(E,L,K0,K1,K2,A,B,'r'))
  P = proj.P_irrep_subspace_22(E,L,I)
else:
  F = F2_alt.Fmat(E,L,alpha)
  G = Gmatrix.Gmat(E,L)
  K2i = K2i_mat.K2inv_mat(E,L,a0,r0,P0,a2)
  K3 = K3quad.K3mat(E,L,K0,K1,K2,A,B,'r')
  P = proj.P_irrep_subspace(E,L,I)

S = defns.chop(F+G)
H = defns.chop(F+G+K2i); Hi = defns.chop(LA.inv(H))
FHi = F@Hi;  HiF = Hi@F
FHiF = F@Hi@F
T1 = np.identity(len(F)) - 3*F @ Hi
T2 = np.identity(len(F)) - 3*Hi @ F
F3 = T1 @ F / (3*L**3); F3i = defns.chop(LA.inv(F3))
#F3 = (F/3 - F@Hi@F) / L**3; F3i = defns.chop(LA.inv(F3))

F_I = P.T@F@P
G_I = P.T@G@P
K2i_I = P.T@K2i@P
S_I = F_I+G_I
H_I = F_I+G_I+K2i_I; Hi_I = defns.chop(LA.inv(H_I))
FHi_I = F_I@Hi_I
HiF_I = Hi_I@F_I
FHiF_I = F_I@Hi_I@F_I
T1_I = P.T@T1@P
T2_I = P.T@T2@P
F3_I = P.T@F3@P; F3i_I = defns.chop(LA.inv(F3_I))
K3_I = P.T@K3@P


#nnp=[0,0,0]
#nnk=[0,1,0]
##print(sums.F2KSS(E,L,nnk,0,0,0,0,alpha))
##print(sums.F2KSS(E,L,nnk,2,0,2,0,alpha))
##print(F[0:6,0:6])
#
#F00 = sums.F2KSS(E,L,nnk,0,0,0,0,alpha)
##print(F00,F[6,6]/F00)
#print(F00,F[12,12]/F00)
#F2020 = sums.F2KSS(E,L,nnk,2,0,2,0,alpha)
##print(F2020,F2020/F[9,9],F2020/F00)
#print(F2020,F2020/F[15,15],F2020/F[12,12])
##print(F[6:12,6:12]/F[6,6])
#
#
#for i in range(1,7):
#  print(np.around(F[6*i:6*(i+1),6*i:6*(i+1)]/F[6,6],4))
#  print(np.around(G[0:6,6*i:6*(i+1)]/G[0,6],4))
#
##print(Gmatrix.G(E,L,[0,0,0],[0,0,1],0,0,0,0))
##print(Gmatrix.G(E, L, nnp, nnk,2,0,2,0))#/Gmatrix.G(E,L,[0,0,0],[0,0,1],0,0,0,0))



print(sorted(LA.eigvals(F_I).real,key=abs,reverse=True))
print(sorted(LA.eigvals(G_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(K2i_I).real,key=abs,reverse=True))
print(sorted(LA.eigvals(S_I).real,key=abs,reverse=True))
print(sorted(LA.eigvals(H_I).real,key=abs,reverse=True))
print(sorted(LA.eigvals(Hi_I).real,key=abs))
#print(sorted(LA.eigvals(FHi_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(FHiF_I).real,key=abs,reverse=True))
print(sorted(LA.eigvals(F3_I).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(F3i_I).real,key=abs))
#print(sorted(LA.eigvals(K3_I).real,key=abs,reverse=True))
print(sorted(LA.eigvals(F3i_I+K3_I).real,key=abs))
#print(sorted(LA.eigvals(np.identity(len(F3_I))+F3_I@K3_I).real,key=abs))

#print(sorted(LA.eigvals(F).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(G).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(S).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(F3).real,key=abs,reverse=True))
#print(sorted(LA.eigvals(K3).real,key=abs,reverse=True)[0:11])
#print(sorted(LA.eigvals(F3i+K3).real,key=abs))


[F3_elist,F3_vlist] = LA.eig(F3i_I)
#[F3_elist,F3_vlist] = LA.eig(F3i)
F3_ivec = [i for i,e in enumerate(F3_elist) if abs(e)<1e-2]


[K3_elist,K3_vlist] = LA.eig(K3_I)
#[K3_elist,K3_vlist] = LA.eig(K3)
K3_ivec = [i for i,e in enumerate(K3_elist) if abs(e)>1e-2]
print()
for i in F3_ivec:
  e_F3 = F3_elist[i].real
  v_F3 = F3_vlist[:,i]
  for j in K3_ivec:
    e_K3 = K3_elist[j].real
    v_K3 = K3_vlist[:,j]
    #print(v_F3)
    print(e_F3,e_K3,abs(np.dot(v_F3,v_K3)))


#print(F_I/p)
#print(F3_I)
#print(K3_I)
#print((F3i_I+K3_I)/p)

#print(p)

