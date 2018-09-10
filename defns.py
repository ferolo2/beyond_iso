import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
from itertools import permutations as perms

####################################################################################
# This file defines several basic functions that get called multiple times
####################################################################################

# om_k
def omega(k):
	return sqrt( k**2 + 1 )

# E2k*
def E2k(E,k):
	return sqrt( 1 + E**2 - 2 * E * omega(k) ) # should always be >=0

# qk*
def qst(E,k):
  Estar = E2k(E,k)
  if Estar >= 2:
    return sqrt( Estar**2/4 - 1 ) # can be real...
  else:
    return 1j*sqrt( 1-Estar**2/4 ) # ...or imaginary

# gam_k
def gamma(E, k):
    return (E - sqrt(1. + k**2))/E2k(E,k)


# List first n free energies for given L
def E_free_list(L,count,*args):
  if len(args)==0:
    nmax = 5
  else:
    nmax = args[0]

  nvec_list = []
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        nvec_list.append([n1,n2,n3])

  out = {}
  for i1 in range(len(nvec_list)):
    for i2 in range(i1,len(nvec_list)):
      m1_2 = sum([x**2 for x in nvec_list[i1]])
      m2_2 = sum([x**2 for x in nvec_list[i2]])
      m12_2 = sum([x**2 for x in [-nvec_list[i1][i]-nvec_list[i2][i] for i in range(3)]])

      E1 = sqrt((2*pi/L)**2*m1_2+1)
      E2 = sqrt((2*pi/L)**2*m2_2+1)        
      E12 = sqrt((2*pi/L)**2*m12_2+1)

      out[tuple(sorted([m1_2,m2_2,m12_2]))] = E1+E2+E12

  tmp = sorted(out.items(), key=lambda x: x[1])[0:count]
  out={}
  for i in tmp:
    out[i[0]]=i[1]
  return out


# List first n 2-pt. free energies for given L
def E_2pt_free_list(L,count,*args):
  if len(args)==0:
    nmax = 5
  else:
    nmax = args[0]

  nvec_list = []
  for n1 in range(-nmax,nmax+1):
    for n2 in range(-nmax,nmax+1):
      for n3 in range(-nmax,nmax+1):
        nvec_list.append([n1,n2,n3])

  out = {}
  for i1 in range(len(nvec_list)):
      m1_2 = sum([x**2 for x in nvec_list[i1]])

      E1 = sqrt((2*pi/L)**2*m1_2+1)

      out[m1_2] = 2*E1

  tmp = sorted(out.items(), key=lambda x: x[1])[0:count]
  out={}
  for i in tmp:
    out[i[0]]=i[1]
  return out


##########################################################
# Convert block-matrix index to (l,m)
def lm_idx(i):
  if i not in range(6):
    print('Error in lm_idx: invalid index input')
  else:
    if i==0:
      [l,m] = [0,0]
    else:
      [l,m] = [2,i-3]
  return [l,m]

# Replace small real numbers in array with zero
def chop(arr):
  arr = np.array(arr)
  arr[abs(arr)<1e-13]=0
  return arr

####################################################################################
# Create nnk_list, etc.
####################################################################################
# Find maximum allowed norm(k)
# TB: Copied from Fernando's defs.pyx
def kmax(E):
  alpH = -1.
  aux1 = (1. + alpH)/4.
  aux2 = (3. - alpH)/4.
  xmin = 0.01
  return sqrt((((E**2+1)/4-aux2*xmin-aux1)*(2/E))**2-1)


# Find energy where a certain shell turns on for given L
# Note: This is only exact if xmin=0 in kmax(E)
def Emin(shell,L,alpH=-1,xmin=0.01):
  c = (3-alpH)*xmin + (1+alpH)
  k = LA.norm([x*2*pi/L for x in shell])
  return omega(k) + sqrt(k**2+c)
  

# Create list of shells/orbits 
# Permutation conventions: 000, 00a, aa0, aaa, ab0, aab, abc
def shell_list(E,L):
  nmaxreal = kmax(E)*L/(2*pi)
  nmax = int(np.floor(nmaxreal))

  shells = []
  for n1 in range(nmax+1):
    for n2 in range(n1,nmax+1):
      for n3 in range(n2,nmax+1):
        if n1**2+n2**2+n3**2 <= nmaxreal**2:
          # need to permute for aa0, ab0, aab
          if (n1==0<n2) or (n1>0 and n2==n3):
            shells.append((n2,n3,n1))
          else:
            shells.append((n1,n2,n3))
  # Sort by magnitude         
  shells = sorted(shells, key=lambda k: LA.norm(k))
  return shells 


# Find where new shells open up & # of eigs increases
def shell_breaks(E,L):
  out = []
  for shell in shell_list(E,L):
    out.append(Emin(shell,L))
  #out.append(E)
  return out


# Create list of all permutations & all perms w/ 1 negation
def perms_list(nnk):
  a=nnk[0]; b=nnk[1]; c=nnk[2]
  p_list = list(perms([a,b,c]))
  p_list += list(perms([a,b,-c]))
  p_list += list(perms([a,-b,c]))
  p_list += list(perms([-a,b,c]))
  
  p_list = [ list(p) for p in p_list ]
  return p_list


# Create list of all nnk in a given shell
def shell_nnk_list(shell):
  # 000
  if list(shell)==[0,0,0]:
    return [shell]

  # 00a
  elif shell[0]==shell[1]==0<shell[2]:
    a = shell[2]
    return [(0,0,a),(0,a,0),(a,0,0),(0,0,-a),(0,-a,0),(-a,0,0)]

  # aa0
  elif shell[0]==shell[1]>0==shell[2]:
    a = shell[0]
    return [(a,a,0),(a,0,a),(0,a,a),(a,-a,0),(a,0,-a),(0,a,-a),    (-a,-a,0),(-a,0,-a),(0,-a,-a),(-a,a,0),(-a,0,a),(0,-a,a)]

  # aaa
  elif shell[0]==shell[1]==shell[2]>0:
    a = shell[0]
    return [(a,a,a),(a,a,-a),(a,-a,a),(-a,a,a), (-a,-a,-a),(-a,-a,a),(-a,a,-a),(a,-a,-a)]
  
  # ab0
  elif 0==shell[2]<shell[0]<shell[1]:
    a = shell[0]; b = shell[1]
    return [
      (a,b,0),(b,a,0),(a,0,b),(b,0,a),(0,a,b),(0,b,a),
      (a,-b,0),(-b,a,0),(a,0,-b),(-b,0,a),(0,a,-b),(0,-b,a),
      (-a,-b,0),(-b,-a,0),(-a,0,-b),(-b,0,-a),(0,-a,-b),(0,-b,-a),
      (-a,b,0),(b,-a,0),(-a,0,b),(b,0,-a),(0,-a,b),(0,b,-a)
    ]

  # aab
  elif 0<shell[0]==shell[1]!=shell[2]>0:
    a = shell[0]; b = shell[2]
    return [
      (a,a,b),(a,b,a),(b,a,a),
      (a,a,-b),(a,-b,a),(-b,a,a),
      (a,-a,b),(a,b,-a),(b,a,-a),
      (-a,a,b),(-a,b,a),(b,-a,a),

      (-a,-a,-b),(-a,-b,-a),(-b,-a,-a),
      (-a,-a,b),(-a,b,-a),(b,-a,-a),
      (-a,a,-b),(-a,-b,a),(-b,-a,a),
      (a,-a,-b),(a,-b,-a),(-b,a,-a)
    ]

  # abc
  elif 0<shell[0]<shell[1]<shell[2]:
    return perms_list(shell)

  else:
    print('Error in shell_nnk_list: Invalid shell input')


# New nnk_list broken into shells
def list_nnk(E,L):
  nnk_list = []
  for shell in shell_list(E,L):
    nnk_list += shell_nnk_list(shell)
  return nnk_list


# Make list of all nkvecs that contribute; i.e., H(kvec)!=0 for each kvec = nkvec*2*pi/L
# TB: Copied from Fernando's F2_alt.py
def list_nnk_old(E,L):
  twopibyL = 2*pi/L
  nkmaxreal = kmax(E)/twopibyL;

  # maximum magnitude of nk that still contributes; i.e., H(k)!=0
  nkmax = int(np.floor(nkmaxreal))

  nklist = []
  for n1 in range(-nkmax,nkmax+1):
    for n2 in range(-nkmax,nkmax+1):
      for n3 in range(-nkmax,nkmax+1):
        if np.sqrt(n1**2+n2**2+n3**2)<nkmaxreal:
          nklist.append([n1,n2,n3])
  return nklist


####################################################################################
# Spherical harmonics
####################################################################################

# Complex spherical harmonics
def y2complex(kvec,m): # y2 = |kvec|**2 * sqrt(4pi) * Y2
  if m==2:
    return sqrt(15/8)*(kvec[0]+1j*kvec[1])**2
  elif m==-2:
    return sqrt(15/8)*(kvec[0]-1j*kvec[1])**2
  elif m==1:
    return -sqrt(15/2)*(kvec[0]+1j*kvec[1])*kvec[2]
  elif m==-1:
    return sqrt(15/2)*(kvec[0]-1j*kvec[1])*kvec[2]
  elif m==0:
    return sqrt(5/4)*(2*kvec[2]**2-kvec[0]**2-kvec[1]**2)
  else:
    print('Error: invalid m input in y2')

# Real spherical harmonics
def y2real(kvec,m): # y2 = sqrt(4pi) * |kvec|**2 * Y2
  if m==-2:
    return sqrt(15)*kvec[0]*kvec[1]
  elif m==-1:
    return sqrt(15)*kvec[1]*kvec[2]
  elif m==0:
    return sqrt(5/4)*(2*kvec[2]**2-kvec[0]**2-kvec[1]**2)
  elif m==1:
    return sqrt(15)*kvec[0]*kvec[2]
  elif m==2:
    return sqrt(15/4)*(kvec[0]**2-kvec[1]**2)
  else:
    print('Error: invalid m input in y2real')

# Spherical harmonics w/ flag for real vs. complex (default)
def y2(kvec,m,Ytype):
  if Ytype=='real' or Ytype=='r':
    return y2real(kvec,m)
  else:
    return y2complex(kvec,m)


####################################################################################
# Graveyard (old functions)
####################################################################################

# Convert (l'm',lm) matrix at fixed (p,k) from complex to real basis (not used anymore)
def cpx2real(cpx_mat):
  U = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1j/sqrt(2), 0, 0, 0, -1j/sqrt(2)],
    [0, 0, 1j/sqrt(2), 0, 1j/sqrt(2), 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1/sqrt(2), 0, -1/sqrt(2), 0],
    [0, 1/sqrt(2), 0, 0, 0, 1/sqrt(2)]
  ])
  Ui = U.conj().T
  
  # block size
  N = int(len(cpx_mat)/6)

  real_mat = np.zeros((len(cpx_mat),len(cpx_mat)),dtype=complex)
  for ip in range(N):
    for ik in range(N):
      mat_pk = cpx_mat[ip::N,ik::N]
      real_mat[ip::N,ik::N] = U @ mat_pk @ Ui

  if ((abs(real_mat.imag))>1e-15).any():
    print('Error: nonzero imaginary part in output of cpx2real')
  else:
    real_mat = real_mat.real

  return real_mat


####################################################################################
# Combine (l'm',lm) blocks into full matrix (old structure)
def full_matrix(M00,M20,M02,M22):
  return np.vstack((np.hstack((M00,M02)),np.hstack((M20,M22))))