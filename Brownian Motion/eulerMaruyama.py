# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:59:36 2017

@author: JSULLIVAN
"""

import numpy as np
import matplotlib.pyplot as plt


mu, sigma = .12, .3  
So = 640
T = 1.
N = 2.**6               # increments

# Brownian Motion
def Brownian(seed, N):
    
    np.random.seed(seed)
    dt = 1./N
    b = np.random.normal(0.,1.,int(N)) * np.sqrt(dt)  # Brownian increments
    W = np.cumsum(b)                             # Discretized Brownian path
    return W, b 
 

    


# Exact Solution
def ExactSolution(So, mu, sigma, W, T, N):    
    t = np.linspace(0.,1.,N+1)
    S = []
    S.append(So)
    for i in xrange(1,int(N+1)):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i-1]
        S_temp = So*np.exp(drift + diffusion)
        S.append(S_temp)
    return S, t

W = Brownian(5, N)[0]                           # Discretized Brownian path
soln = ExactSolution(So, mu, sigma, W, T, N)    # Exact solution


# Plotting
X = soln[0]         # Exact solution
time = soln[1]      # Time increments
plt.plot(time, X, label = 'exact')





# Euler Maruyama Approximation
def EM(So, mu, sigma, b, T, N, M):
    dt = M * (1/N)  # EM step size
    L = N / M
    wi = [So]
    for i in xrange(0,int(L)):
        Winc = np.sum(b[(M*(i-1)+M):(M*i + M)])
        w_i_new = wi[i]+mu*wi[i]*dt+sigma*wi[i]*Winc
        wi.append(w_i_new)
    return wi, dt

# inputs
b = Brownian(5, N)[1]    # Brownian increments 
M = 2                    
L = N/M

EM_approx = EM(So, mu, sigma, b, T, N, M)[0]
time_EM = np.linspace(0.,1.,L+1)


plt.plot(time_EM, EM_approx, label = 'EM approx')
plt.legend(loc = 'upper left')
        
print "The time-step for the exact solution is:"
print 1./N
print "\n"
        
print "The time-step for the Euler-Maruyama approximation is:"
print EM(So, mu, sigma, b, T, N, M)[1]

        
        
    
    

        
