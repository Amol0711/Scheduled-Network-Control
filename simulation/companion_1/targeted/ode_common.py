"""Independent deterministic equations at the frozen six-node parameters."""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
N=6;T=2.0;K0=1.2;OMEGA=2.0;ALPHA=1.0;B=1.0;EPS=1e-5
J=np.array([[0.0,-1.0],[1.0,0.0]])
PHASES=np.array([-.75,-.42,-.15,.18,.48,.82])
RADII=np.array([.75,1.25,.90,1.15,.80,1.20])
X0=np.column_stack((RADII*np.cos(PHASES),RADII*np.sin(PHASES)))


def laplacian(kind: str,n:int=N) -> np.ndarray:
    A=np.zeros((n,n))
    if kind=='path':
        for i in range(n-1):A[i,i+1]=A[i+1,i]=1
    elif kind=='ring':
        for i in range(n):A[i,(i+1)%n]=A[i,(i-1)%n]=1
    elif kind=='complete':A=np.ones((n,n))-np.eye(n)
    else:raise ValueError(kind)
    return np.diag(A.sum(axis=1))-A
L=laplacian('ring')


def field(x:np.ndarray) -> np.ndarray:
    return ALPHA*x+OMEGA*(x@J.T)-B*np.sum(x*x,axis=-1)[...,None]*x


def flow(y:np.ndarray,delta:float|np.ndarray) -> np.ndarray:
    """Stable closed reduced flow on the verified collar, using expm1."""
    delta=np.asarray(delta);q=float(np.dot(y,y));u=np.expm1(2*ALPHA*delta)
    denom=ALPHA+B*q*u
    if np.any(denom<=0):raise ValueError('Outside the reduced-flow collar')
    amp=np.sqrt(ALPHA*q*(1+u)/denom)
    theta=np.arctan2(y[1],y[0])+OMEGA*delta
    return np.stack((amp*np.cos(theta),amp*np.sin(theta)),axis=-1)


def physical_inverse(x0,times,lap=L,k0=K0):
    # Integration starts at the first output time, as in the archive. The graph
    # protocol passes times beginning at 1.6, not at zero; this is audited below.
    times=np.asarray(times)
    def rhs(t,z):
        x=z.reshape(N,2);return (field(x)-k0/(T-t)*(lap@x)).ravel()
    sol=solve_ivp(rhs,(float(times[0]),float(times[-1])),x0.ravel(),
                  t_eval=times,method='Radau',rtol=1e-10,atol=1e-12)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y.T.reshape(-1,N,2),sol.nfev
