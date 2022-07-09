import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from math import pi, cos, sin
import png


f = png.Reader('gato.png')
pngdata = f.asDirect()
image_2d = np.vstack(map(np.uint16, pngdata[2]))
#w.write(f, s)

image_2d = np.transpose(image_2d)

plt.imsave('test2.png', (image_2d), cmap=cm.gray)
