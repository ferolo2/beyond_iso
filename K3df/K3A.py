from defns import omega, E2k, qst, y2
import numpy as np
sqrt=np.sqrt; pi=np.pi; conj=np.conjugate; LA=np.linalg
# 1j = imaginary number

#################################################################
# Want to compute K3A = sum(Delta_i**2 + Delta_i'**2)
# First compute omega2sum = sum(omega_i**2 + omega_i'**2)
#################################################################

# Note: input order for all functions is (E,outgoing momenta,incoming momenta)

def omega2sum(E,pvec,lp,mp,kvec,l,m,Ytype='r'):
  k=LA.norm(kvec); p=LA.norm(pvec)

  if (lp==0 and mp==0):
    if (l==0 and m==0):
      S1 = omega(k)**2 + ((E-omega(k))**2)/2 + 2/3*( k*qst(E,k) / E2k(E,k))**2
    elif l==2:
      #S1 = 4/15*(qst(E,k) / E2k(E,k))**2 * conj(y2(kvec,m,Ytype))
      S1 = 4/15 * (1/E2k(E,k))**2 * conj(y2(kvec,m,Ytype)) # TB, no q
    else:
      S1=0
  else:
    S1=0

  if (l==0 and m==0):
    if (lp==0 and mp==0):
      S2 = omega(p)**2 + ((E-omega(p))**2)/2 + 2/3*( p*qst(E,p) / E2k(E,p))**2
    elif lp==2:
      #S2 = 4/15*(qst(E,p) / E2k(E,p))**2 * y2(pvec,mp,Ytype)
      S2 = 4/15 * (1/E2k(E,p))**2 * y2(pvec,mp,Ytype) # TB, no q
    else:
      S2=0
  else:
    S2=0

  if np.imag(S1+S2)>10**-8:
    print('Error: imaginary part in omega2sum')
  return np.real(S1+S2)

# Compute K3A
def K3A(E,pvec,lp,mp,kvec,l,m,Ytype='r'):
  S = 4*E**2 * omega2sum(E,pvec,lp,mp,kvec,l,m,Ytype)
  if l==m==lp==mp==0:
    S-= 2*(E**2-3)*(E**2+9)
  return S
