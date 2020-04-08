#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 19:17:22 2018

The script is divided in blocks, delimited by '# %%'
Please check each cell


'beta' is a vector of lenght 2




@author: rho
"""
import numpy as np
from scipy.stats import norm
import scipy
import pandas as pd
import datetime
from dateutil import parser
import time


import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


X0 = 100
t0 = 0
T = 15/365
r = 0.01
I = 120
N = 1000
SIGMA = 0.15
K_I = 120
BETA_1 = 15 
BETA_2 = 1
deltaT = T/N
deltaK = K_I/I

DATE = datetime.date(2014, 1, 1)

# %%


def sigmaDupire(t, K, beta1, beta2):
    """Compute the CEV volatility
        t is useless
    """

    return beta1 * pow(K, -beta2)


def A(i, t, K, beta1, beta2):

    sigma = sigmaDupire(t, K, beta1, beta2)
    return deltaT * 0.5 * (pow(i, 2) * pow(sigma, 2) + i * r)


def B(i, t, K, beta1, beta2):

    sigma = sigmaDupire(t, K, beta1, beta2)
    return 1 - deltaT * pow(i, 2) * pow(sigma, 2)


def C(i, t, K, beta1, beta2):

    sigma = sigmaDupire(t, K, beta1, beta2)
    return deltaT * 0.5 * (pow(i, 2) * pow(sigma, 2) - i * r)


# %%


def matrixA(n, beta1, beta2):
    "Compute de Matrix of the Ai, Bi, Ci, with respect to the time n"

    matrix = np.zeros((I, I))
    Tn = n * deltaT

    matrix[0, 0] = B(1, Tn, 1 * deltaK, beta1, beta2)  # B1
    matrix[0, 1] = C(1, Tn, 1 * deltaK, beta1, beta2)  # C1
    matrix[I - 1, -1] = B(I, Tn, I * deltaK, beta1, beta2)  # CI
    matrix[I - 1, -2] = A(I, Tn, I * deltaK, beta1, beta2)  # AI

    k = 0
    for j in range(1, I-1):  # from line 2 to I-1
        m = k
        matrix[j, m] = A(j + 1, Tn, j * deltaK, beta1, beta2)
        matrix[j, m + 1] = B(j + 1, Tn, j * deltaK, beta1, beta2)
        matrix[j, m + 2] = C(j + 1, Tn, j * deltaK, beta1, beta2)
        k += 1

    return matrix


def matrixB(n, beta1, beta2):

    Tn = n * deltaT

    matrix = np.zeros(I)
    matrix[0] = A(1, Tn, 1 * deltaK, beta1, beta2) * X0

    return matrix

# %%


def surfaceDupire(beta1, beta2):
    """Compute the surface with respect to beta 1 and beta 2"""

    V = np.full(I, X0) - np.arange(1, I + 1) * deltaK  # V0
    V = V.clip(min=0)
    V = np.expand_dims(V, axis=1)
    traj = V

    for n in range(0, N):
        Bn = np.expand_dims(matrixB(n, beta1, beta2), axis=1)  # Bn
        V = np.matmul(matrixA(n, beta1, beta2), V) + Bn  # Vn+1
        traj = np.concatenate((traj, V), axis=1)

    return traj


# %%

"""RUN THESE LINES -IF YOU WANT- TO COMPUTE A SURFACE """


start = time.time()

surface = surfaceDupire(BETA_1, BETA_2)

end = time.time()
print(end - start)


# %%

def blackScholes(t, x, T, K, sigma):
    """T : maturity in years !"""

    d1 = np.log(x / K) + (r + 0.5 * pow(sigma, 2)) * (T - t)
    d1 = d1 / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)

    return x * norm.cdf(d1) - np.exp(-r * (T - t)) * K * norm.cdf(d2)


def find_nearest(array, value):
    """This function returns the index of the closest element in 'array'
        given the variable 'value'
        It is useful to pick some V(T,K) on the Dupire surface
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def findCallValueOnSurface(surface, T, K):
    """
    Given a surface (matrixV) find a single call value V(T,K) on it.

    matrixV == the dupireSurface (a matrix)
    """

    tVect = np.arange(0, N + 1) * deltaT
    idxTime = find_nearest(tVect, T)
    KiVect = np.arange(1, I + 1) * deltaK
    idxK = find_nearest(KiVect, K)

    return surface[idxK, idxTime]

# %%



SnP500 = pd.read_csv('DataCallS&P500.tsv', sep='\t')
SnP500.sample(frac=1)

for i in range(len(SnP500["Date"])):
    date_i = parser.parse(SnP500["Date"].iloc[i]).date()
    delta = date_i - DATE
    delta = delta.days
    SnP500["Date"].iloc[i] = delta / 365
"""
"""
SnP500['Strike'] = pd.to_numeric(SnP500['Strike'])
listT_K = SnP500[['Date', 'Strike']].values.tolist()
listObs = SnP500["Prix"].values.tolist()





#%%

"""
RUN THESE LINES FOR EXTRACTION OF 48 V(T,K) (call value) on the surface.

Price values are stored in 'listObs'
Date and Strike values are stored in 'listT_K'
"""

callSimulated = pd.DataFrame(columns=["Date", "Strike", "Prix"])

i = 0

for K in [80, 85, 90, 95, 100, 110]:
    for T in [7, 8, 9, 10, 11, 12, 13, 14]:
        sigma = sigmaDupire(0, K, 15, 1)
        callSimulated.loc[i] = [T/365, K, findCallValueOnSurface(surface, T/365, K)]
        # callSimulated.loc[i] = [T/365, K, blackScholes(0, 100, T/365, K, sigma)]
        i += 1


listT_K = callSimulated[['Date', 'Strike']].values.tolist()
listObs = callSimulated["Prix"].values.tolist()


# %%


"""
RUN THESE LINES FOR EXTRACTION OF 20 V(T,K) (call value) on the surface.

Price values are stored in 'listObs'
Date and Strike values are stored in 'listT_K'
"""

callSimulated = pd.DataFrame(columns=["Date", "Strike", "Prix"])
i = 0
for K in [80, 90, 95, 100, 110]:
    for T in [8, 10, 12, 14]:
        sigma = sigmaDupire(0, K, 15, 1)
        callSimulated.loc[i] = [T/365, K, findCallValueOnSurface(surface, T/365, K)]
        # callSimulated.loc[i] = [T/365, K, blackScholes(0, 100, T/365, K, sigma)]
        i += 1


listT_K = callSimulated[['Date', 'Strike']].values.tolist()
listObs = callSimulated["Prix"].values.tolist()


# %%

"""RUN THESE LINES TO ADD NOISE TO V(T,K) VALUES (listObs) """

noise = 0.5

listObs = listObs + np.random.normal(0, noise, len(listObs))


# %%

def errorFunction(beta):
    """
    Given beta, compute the square of L2 distance between list of observed
    call prices 'listObs' and call prices from the surface:

    call_from_surface is computed inside this function

    YOU HAVE TO ALREADY HAVE COMPUTED YOUR OBSERVATIONS: 'listObs'
    AND THEIR ASSOCIATED MATURITY 'T' AND STRIKE 'K': 'listT_K'

        ('listObs' - 'call_from_surface')^2 is returned

    beta : vector of parameters B0 and B1

    """

    print("-----------------------------")
    print(f'betaValues({beta}) called \n')
    global historyBeta
    historyBeta.append((beta[0], beta[1]))
    global count
    global listT_K, listObs

    valuesList = np.array([])  # values from the matrix
    surface = surfaceDupire(beta[0], beta[1])

    for j in range(len(listT_K)):
        Tj = listT_K[j][0]
        Kj = listT_K[j][1]
        value = findCallValueOnSurface(surface, Tj, Kj)
        # value = blackScholes(0, X0, Tj, Kj, sigmaNew(0,Kj, beta1, beta2))
        valuesList = np.append(valuesList, value)

    listObs = np.array(listObs)
    dist = np.sqrt(np.square(valuesList - listObs).sum())  # * 100 # norm 2

    print(f"Error (L2 norm): {dist}")
    count += 1
    print(f"Iteration number: {count}")
    print("-----------------------------\n")
    print("")

    return dist


# %%

"""RUN THIS CELL BEFORE LAUNCHING optimisation() please
    it will reinitialize 'count' and 'historyBeta'
"""

""" 'count' is the number of iterations to optimise """
""" 'historyBeta' records all the betas values during the optimization """

count = 0
historyBeta = []


def optimisation(betaInit=[15, 1]):

    """
    betaInit[0] == beta1 ; betaInit[1] == beta 2
    Function which optimize over beta the L2 norm
    See scipy doc for more info
    The parametrization will deeply affect the results !
    """

    global count, historyBeta

    start = time.time()

    cons = ({'type': 'ineq', 'fun': lambda beta: beta[0]},
            {'type': 'ineq', 'fun': lambda beta: beta[1]})

    test = scipy.optimize.minimize(errorFunction,
                                   betaInit,
                                   constraints=cons,
                                   tol=0.00000001,
                                   bounds=((1, 100), (0.978, 1.023)))
                                   # method="COBYLA")

    end = time.time()
    print(f"--- Optim time: {end - start}s ---")

    return print("")


#%%

#plt.plot([a[0] for a in historyBeta], [i for i in range(len(historyBeta))], 'ro')

"""IF YOU WANT TO PLOT SOME THINGS """

import numpy as np
import matplotlib.pyplot as plt

x = [i for i in range(len(historyBeta))]
y = [a[0] for a in historyBeta]

import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.xlabel('# iterations', fontsize=14)
plt.ylabel('beta1', fontsize=14)

plt.plot(x, y)

plt.savefig('beta1convergenceNOISY.png')

plt.show()

