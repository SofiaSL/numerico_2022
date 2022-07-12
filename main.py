import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin, sqrt
import png
<<<<<<< HEAD
from random import normalvariate
=======

def qr_algo(A):
    n = 100
    L = A
    Q = np.identity(A.shape[0])
    for i in range(n):
        q, r = np.linalg.qr(L)
        L = r @ q
        Q = q
    return L,  Q
>>>>>>> b51cb487982ed619f1b9d7a9710aeecceadf4c09

f = png.Reader('gato.png')
pngdata = f.asDirect()
image_2d = np.vstack(map(np.uint16, pngdata[2]))
#w.write(f, s)

<<<<<<< HEAD
#image_2d = np.transpose(image_2d)

def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = randomUnitVector(min(n,m))
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs

p,q,r=svd(image_2d,k=100)

A = q@np.diag(p)@r

plt.imsave('out.png', A, cmap=cm.gray)
=======
#L = qr_algo(image_2d)

#print(L)

for i in range(2048):
    for j in range(2048):
        image_2d[j,i] = image_2d[i,j]
        
L , Q = qr_algo(image_2d/256)

compress = 256 * (np.transpose(Q) @ L @ Q)

print(image_2d - compress)

plt.imsave('gato3.png', (compress), cmap=cm.gray)
>>>>>>> b51cb487982ed619f1b9d7a9710aeecceadf4c09
