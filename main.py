import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import pi, cos, sin
import png


f = png.Reader('gato.png')
pngdata = f.asDirect()
image_2d = np.vstack(map(np.uint16, pngdata[2]))
#w.write(f, s)

image_2d = image_2d / 256

u,s,v = np.linalg.svd(image_2d)

k = 200

u = u[:,0:k]
s = np.diag(s[0:k])
v = v[0:k,:]

image_2d = 256* u @ s @ v

plt.imsave('gato2.png', (image_2d), cmap=cm.gray)
