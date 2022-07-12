import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin, sqrt
import png
from random import normalvariate

f = png.Reader('gato.png')
pngdata = f.asDirect()

# matriz cujas entradas correspondem ao tom de cinza de cada pixel da imagem
image_2d = np.vstack(map(np.uint16, pngdata[2]))


# vetor unitario aleatorio com distribuicao uniforme
def vetor_aleatorio(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


# decomposicao SVD unidimensional
def svd_1d(A, tol=1e-10):

    n, m = A.shape
    x = vetor_aleatorio(min(n,m))
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

        if abs(np.dot(currentV, lastV)) > 1 - tol:
            print("converged in {} iterations!".format(iterations))
            return currentV


# decomposicao SVD da matriz A
# calculada usando metodo das potencias
# k eh o numero de valores singulares a serem computados
def svd(A, k=None, tol=1e-10):
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
            v = svd_1d(matrixFor1D, tol=tol)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, tol=tol)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs

p,q,r=svd(image_2d,k=20)

A = q@np.diag(p)@r

plt.imsave('out.png', A, cmap=cm.gray)