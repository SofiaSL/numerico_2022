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


# base ortonormal para a matriz
# baseada na decomposiao QR do numpy
def base_orto(M):
    Q, _ = np.linalg.qr(M)
    return Q


def iter_subspaco(A, Y0, n_iters):
    Q = base_orto(Y0)
    for i in range(n_iters):
        Z = base_orto(A.T @ Q)
        Q = base_orto(A @ Z)
    return Q


def find_range(A, n_samples, n_subspace_iters=None):
    m, n = A.shape
    O = np.random.randn(n, n_samples)
    Y = A @ O

    if n_subspace_iters:
        return iter_subspaco(A, Y, n_subspace_iters)
    else:
        return base_orto(Y)

def svd(A, rank, n_subspace_iters=None):
    n_samples = 2 * rank

    # estagio A.
    Q = find_range(A, n_samples, n_subspace_iters)

    # estagio B.
    B = Q.T @ A
    U_tilde, S, Vt = np.linalg.svd(B)
    U = Q @ U_tilde

    # truncar
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    return U, S, Vt

p,q,r=svd(image_2d,rank=100)

A = p@np.diag(q)@r

plt.imsave('out.png', A, cmap=cm.gray)