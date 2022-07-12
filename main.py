import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import pi, cos, sin, sqrt
import png
from random import normalvariate

f = png.Reader('gato.png')
pngdata = f.asDirect()

# matriz cujas entradas correspondem ao tom de cinza de cada pixel da imagem
image_2d = np.vstack(list(pngdata[2]))


# vetor unitario aleatorio com distribuicao uniforme
# gerado usando uma distribuicao uniforme em [0,1]^n e normalizando
def vetor_aleatorio(n):
    vec = [normalvariate(0, 1) for _ in range(n)]
    norma = sqrt(sum(x * x for x in vec))
    return [x / norma for x in vec]


# decomposicao SVD unidimensional
def svd_1d(A, tol):

    n, m = A.shape
    vec = np.zeros(min(n,m))
    prox_vec = vetor_aleatorio(min(n,m))

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    while abs(np.dot(prox_vec, vec)) < 1 - tol:
        vec = prox_vec
        prox_vec = np.dot(B, vec)
        prox_vec = prox_vec / np.linalg.norm(prox_vec)

    return prox_vec


# decomposicao SVD da matriz A
# calculada usando metodo das potencias
# o segundo argumento eh o numero de valores singulares a serem computados
def svd(A, k, tol=1e-10):
    A = np.array(A, dtype=float)
    n, m = A.shape
    out = [] # lista com as matrizes unitarias e os valores singulares

    # cada iteracao do loop adiciona um valor singular e aumenta em 1 o tamanho das matrizes
    for i in range(k):
        matriz_1d = A.copy()

        for valor_singular, u, v in out[:i]:
            matriz_1d -= valor_singular * np.outer(u, v)

        if n > m:
            v = svd_1d(matriz_1d, tol)  # proximo vetor singular
            u0 = np.dot(A, v)
            sigma = np.linalg.norm(u0)  # proximo valor singular
            u = u0 / sigma              # normalizar o vetor
        else:
            u = svd_1d(matriz_1d, tol) 
            v = np.dot(A.T, u)
            sigma = np.linalg.norm(v)
            v_normalizado = v / sigma

        out.append((sigma, u, v_normalizado))

    val_sing, us, vs = [np.array(x) for x in zip(*out)]
    return val_sing, us.T, vs

p,q,r=svd(image_2d, 20)

A = q@np.diag(p)@r

plt.imsave('out.png', A, cmap=cm.gray)