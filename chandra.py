import numpy as np
import numpy.polynomial.legendre as leg
from scipy.optimize import brentq

def coefficients(n):

    ## gaussian weights

    L = leg.Legendre([0 for i in range(2*n)] + [1]) # Leg([c_0,c_1,c_2,...]) = Sum(c_i*L_i). We just want one order
    mu = L.roots()

    # Simple formula for the weights #https://en.wikipedia.org/wiki/Gaussian_quadrature
    a = [2/(1-x**2)/L.deriv()(x)**2 for x in mu]

    # Keep only positive values, knowing that mu_-i=-mu_i, a_-i=a_i
    mu,a = mu[n:],a[n:]


    ## k

    # The k^2 satisfy an algebraic equation, Ch. III Eq. (7)
    f = lambda k2 : 1 - sum([a[j]/(1-mu[j]**2*k2) for j in range(n)]) # should be =0

    # There are 2n-2 non-zero roots, or n-1 positive roots (k_-i=k_i)
    # There's no obvious way to find all the roots at once, so I brute force by brentq'ing inside a moving window
    k2_roots = []
    x = 1e-3 # k2=0 is a solution but we ignore it
    while len(k2_roots)<n-1:
        if f(x)*f(x+0.1) < 0:
            root = brentq(f,x,x+0.1)

            # The equation is poorly behaved and brentq finds wrong roots, so check it
            if abs(f(root))<1e-3:
                k2_roots.append(root)
        x+=0.1

    k = np.sqrt(k2_roots)


    ## L
    P = lambda x : np.prod([x-mu[i] for i in range(n)]) # Eq. (55)
    def R(alpha,x): # Eq. (67)
        terms=[]
        for beta in range(n-1):
            if beta!=alpha:
                terms.append(1-k[beta]*x)
        return np.prod(terms)

    L = [(-1)**n * np.prod(k) * P(1/k[alpha])/R(alpha,1/k[alpha]) for alpha in range(n-1)] # Eq. (66)

    ## Q
    Q = sum(mu)-sum(1/k) # Eq. (77)

    return k,L,Q

if __name__ == "__main__":
    for n in range(1,6):
        k,L,Q=coefficients(n)
        print("n =",n)
        print("k: ",k)
        print("Q: ",Q)
        print("L: ",L,"\n")
