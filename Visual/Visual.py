"""
=======================
Plot 2D data on 3D plot
=======================
 
Demonstrates using ax.plot's zdir keyword to plot 2D data on
selective axes of a 3D plot.
"""
 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
 
def randrange(n,vmin,vmax) :
  return (vmax-vmin)*np.random.rand(n) + vmin
fig = plt.figure()
ax = fig.add_subplot(110,projection='3d')
X = randrange(100,23,32)
Y = randrange(100,0,100)
Z = randrange(100,-50,-25)

ax.scatter(1,1,1)
plt.savefig("V.jpg")