import matplotlib.pyplot as plt
import numpy as np
from math import pi, cos, sin
#import networkx as nx

n = 1000

A = np.random.rand(n,n)
A = A / np.transpose(A.sum(axis=0)[:,None])
print(A)

w, vl = np.linalg.eig(A)
autovet = vl[:, np.argmax(w)]

#print(autovet)
#print(A.dot(autovet))

pos_x = []
pos_y = []

for i in range(n):
    pos_x.append(cos(2 * pi * i / n))
    pos_y.append(sin(2 * pi * i / n))
    
plt.scatter(pos_x, pos_y, c = autovet)
plt.savefig("out.png")