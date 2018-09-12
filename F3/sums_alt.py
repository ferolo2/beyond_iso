import numpy as np
pi = np.pi
import math
import defns

from numpy.lib.scimath import sqrt


#from pathlib import Path
from numba import jit, autojit
from scipy.special import sph_harm
from scipy.special import erfi
from scipy.special import erfc
from scipy.optimize import fsolve

global alpH, aux1, aux2
alpH = -1.
aux1 = (1. + alpH)/4.
aux2 = (3. - alpH)/4.
xmax = 0.97
xmin = 0.01


@jit(nopython=True,fastmath=True) #FRL, this speeds up like 5-10%
def npsqrt(x):
    return np.sqrt(x)
 
@jit(nopython=True,fastmath=True) #FRL, this speeds up like 5-10%
def square(x):
    return x**2
 
 
@jit(nopython=True,fastmath=True)
def exp(x):
    return np.exp(x)
 
@jit(nopython=True,parallel=True,fastmath=True)
def mydot(x,y):
    res = 0.
    for i in range(3):
        res+=x[i]*y[i]
    return res

@jit(nopython=True, parallel=True,fastmath=True) #FRL this speeds up like 20%
def norm(nnk):
    nk=0.
    for i in nnk:
        nk += i**2
    return npsqrt(nk)


def jj( x):
    xmin = 0.01
    xmax = 0.97
    if xmin < x < xmax:
        return exp(-exp(-1/(1-x))/x)
    elif x >= xmax:
        return 1.
    else:
        return 0.

# E2k**2 / 4
def E2a2(e, a):
    return (1.+square(e))/4. - e*npsqrt(1.+square(a))/2

def hh(e, k):
    alpH = -1.
    aux1 = (1. + alpH)/4.
    aux2 = (3. - alpH)/4.
    return jj( (E2a2(e,k) - aux1)/aux2  )

def gam(e, k):
    return (e - npsqrt(1. + square(k)))/(2*npsqrt(E2a2(e, k)))

def xx2(e, L, k):
    return ( E2a2(e, k) - 1)*L*L/square(2*math.pi);




# TB: choose basis inside
def summand(e, L, nna, nnk, gamma, x2,l1,m1,l2,m2,alpha):

    nk = norm(nnk);

    nnA = np.array(nna)
    nnK = np.array(nnk)
    nnb = -1*nnA -1*nnK
    
    if(nk==0):
        rr=nnA
    else:
        rr = nnA + nnK/(2*gamma) + nnK*(1/gamma -1)*mydot(nnA,nnK)/square(nk)

    rr2 = mydot(rr, rr)
    twopibyL = 2*math.pi/L
    a = norm(nnA)*twopibyL
    b = norm(nnb)*twopibyL
    
    # TB: Choose spherical harmonic basis
    Ytype = 'r'  # 'r' for real, 'c' for complex
    Ylmlm=1
    if l1==2:
      Ylmlm = defns.y2(rr,m1,Ytype)
    if l2==2:
      Ylmlm = Ylmlm * defns.y2(rr,m2,Ytype)
    
    exponential = exp(alpha*(x2-rr2))
    
    out = Ylmlm*exponential/(x2 - rr2)
    if (Ytype=='r' or Ytype=='real') and abs(out.imag)>1e-15:
      print('Error in summand: imaginary part in real basis')
    else:
      out = out.real
    return out



# TB: May edit to try improving run-time at shell thresholds
def getnmax(cutoff,alpha,x2,gamma):
    eqn = lambda l : -cutoff + 2*math.pi*npsqrt(math.pi/alpha) * exp(alpha*x2)*erfc(npsqrt(alpha)*l)

    
    n0=1
    solution=fsolve(eqn, n0)
        
    return int(np.round(max(solution*gamma,1)+3))

    

def sum_nnk(e, L, nnk,l1,m1,l2,m2,alpha):
    nk = norm(nnk)
    if(E2a2(e, nk*2*math.pi/L)<=aux1):
        return 0.
    else:
        twopibyL = 2.*math.pi/L
        k = nk*twopibyL
        #nn0 = npsqrt((e**2 - alpH)**2/(4*e**2) - 1)/twopibyL;
        gamma = gam(e, k)
        x2 = xx2(e, L, k)
 #       nmax = math.floor(nn0);
        #hhk = hh(e, k)  # TB: put hhk in C

        cutoff=1e-9
        
        
        nmax = getnmax(cutoff,alpha,x2,gamma)

        ressum=0.
        for n1 in range(-nmax,nmax+1):
            for n2 in range(-nmax,nmax+1):
                for n3 in range(-nmax,nmax+1):
                    if norm([n1,n2,n3])<=nmax:
                      ressum += summand(e, L, [n1, n2, n3], nnk, gamma, x2,l1,m1,l2,m2,alpha) #TB
                      #ressum += hhk*summand(e, L, [n1, n2, n3], nnk, gamma, x2,l1,m1,l2,m2,alpha)
            
        # return (x2*twopibyL**2)**(-(l1+l2)/2)*ressum # FRL
        #return x2**(-(l1+l2)/2)*ressum # TB
        return (2*pi/L)**(l1+l2) * ressum # TB, no q


def int_nnk(e,L,nnk,l1,m1,l2,m2,alpha):
    nk = norm(nnk)
    twopibyL = 2.*math.pi/L
    k = nk*twopibyL
    gamma = gam(e, k)
    x2 = xx2(e, L, k)
    #hhk = hh(e, k) # TB: put hhk in C
    x_term = x2**(-(l1+l2)/2) # TB
    #q2s = (x2*twopibyL**2)**(-(l1+l2)/2) # FRL

    if(l1!=l2 or m1!=m2):
        return 0.

    elif(l1==l2==0):
        factor1 = -sqrt(math.pi/alpha)*0.5*exp(alpha*(x2))
        factor2 = 0.5*math.pi*sqrt(x2)*erfi(sqrt(alpha*x2))

        #out = q2s*hhk*4*math.pi*gamma*(factor1 + factor2) # FRL
        # out = x_term*4*math.pi*gamma*(factor1 + factor2) #TB
        out = 4*math.pi*gamma*(factor1 + factor2) #TB, no q

    elif(l1==l2==2):
        factor1 = -npsqrt(math.pi/alpha**5)*(3+2*alpha*x2+4*square(alpha*x2))*exp(alpha*(x2))/8
        factor2 = 0.5*math.pi*sqrt(x2**5)*erfi(sqrt(alpha*x2))

        #out = q2s*hhk*4*math.pi*gamma*(factor1 + factor2) # FRL
        #out = x_term*4*math.pi*gamma*(factor1 + factor2) #TB
        out = (2*pi/L)**4 * 4*math.pi*gamma*(factor1 + factor2) #TB, no q

    else:
      return 0.
    
    if abs(out.imag)>1e-15:
        print('Error in int_nnk: imaginary part in output')
    else:
      out = out.real
    return out
    
       
# Calculate F (this is really Ftilde=F/(2*omega))
def F2KSS(e,L,nnk,l1,m1,l2,m2,alpha):
    nk = norm(nnk)
    k = nk*2*math.pi/L
    omk = npsqrt(1. + square(k))
    hhk = hh(e, k)
    
    if hhk==0:
      return 0
    else:
      SUM = sum_nnk(e, L, nnk,l1,m1,l2,m2,alpha)
      INT = int_nnk(e,L,nnk,l1,m1,l2,m2,alpha)
      C = hhk/(32*omk*square(math.pi)*L*(e - omk)) #TB
      #C = 1/(32*omk*math.pi**2*L*(e - omk))
      return (SUM-INT)*C



#########################################################
# TB: Never touched/used this 
def sum_nnk_test(e, L, nnk,l1,m1,l2,m2,alpha,nmax2):
    nk = norm(nnk)
    if(E2a2(e, nk*2*math.pi/L)<=aux1):
        return 0.
    else:

        twopibyL = 2.*math.pi/L
        k = nk*twopibyL
        #nn0 = npsqrt((e**2 - alpH)**2/(4*e**2) - 1)/twopibyL;
        gamma = gam(e, k)
        x2 = xx2(e, L, k)
#        nmax = math.floor(nn0);
        hhk = hh(e, k)
        #cutoff=1e-9
        
        nmax=nmax2
#        nmax = getnmax(cutoff,alpha,x2,gamma)

        ressum=0.
        for n1 in range(-nmax,nmax+1):
            for n2 in range(-nmax,nmax+1):
                for n3 in range(-nmax,nmax+1):
                    ressum += hhk*summand(e, L, [n1, n2, n3], nnk, gamma, x2,l1,m1,l2,m2,alpha)


                    
        return (x2*square(twopibyL))**(-(l1+l2)/2)*ressum


