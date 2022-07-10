import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import pi, cos, sin
import png

def qr_algo(A):
    n = 100
    L = A
    Q = np.identity(A.shape[0])
    for i in range(n):
        q, r = np.linalg.qr(L)
        L = r @ q
        Q = q
    return L,  Q

f = png.Reader('gato.png')
pngdata = f.asDirect()
image_2d = np.vstack(map(np.uint16, pngdata[2]))
#w.write(f, s)

#L = qr_algo(image_2d)

#print(L)

for i in range(2048):
    for j in range(2048):
        image_2d[j,i] = image_2d[i,j]
        
L , Q = qr_algo(image_2d/256)

compress = 256 * (np.transpose(Q) @ L @ Q)

print(image_2d - compress)

plt.imsave('gato3.png', (compress), cmap=cm.gray)
