#!/usr/bin/env python3
# Marcos del Cueto
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(xmin,xmax,Delta,noise):
    # Calculate f=sin(x1)+cos(x2)
    x1 = np.arange(xmin,xmax+Delta,Delta)   # generate x1 values from xmin to xmax
    x2 = np.arange(xmin,xmax+Delta,Delta)   # generate x2 values from xmin to xmax
    x1, x2 = np.meshgrid(x1,x2)             # make x1,x2 grid of points
    f = np.sin(x1) + np.cos(x2)             # calculate for all (x1,x2) grid
    # Add random noise to f
    random.seed(2020)                       # set random seed for reproducibility
    for i in range(len(f)):
        for j in range(len(f[0])):
            f[i][j] = f[i][j] + random.uniform(-noise,noise)  # add random noise to f(x1,x2)
    return x1,x2,f

x,y,f=generate_data(-10,10,0.1,0)
# Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 2
x1,y1,f1=generate_data(-10,10,1.0,2)

fig = plt.figure()
# Right subplot
ax = fig.add_subplot(1, 2,2)
ax.set(adjustable='box', aspect='equal')
ax.contourf(x, y, f, cmap='Greys',zorder=0)
points = ax.scatter(x1, y1, c=f1,cmap='viridis',s=40,zorder=1)
cbar=plt.colorbar(points)
cbar.set_label("$f(x_1,x_2)$",fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_xlabel('$x_1$',fontsize=16)
ax.set_xticks(np.arange(-10,12.5,2.5))
ax.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
ax.set_ylabel('$x_2$',fontsize=16)
ax.set_yticks(np.arange(-10,12.5,2.5))
ax.set_yticklabels(np.arange(-10,12.5,2.5),fontsize=14)
# Left subplot
ax1 = fig.add_subplot(1, 2, 1,projection='3d')
surf = ax1.plot_surface(x, y, f, rstride=1, cstride=1,linewidth=0, antialiased=False,cmap='Greys',zorder=0)
ax1.scatter(x1, y1, f1,c=f1,cmap='viridis',s=15,zorder=1)
ax1.set_xlabel('$x_1$',fontsize=16)
ax1.set_xticks(np.arange(-10,12.5,2.5))
ax1.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
ax1.set_ylabel('$x_2$',fontsize=16)
ax1.set_yticks(np.arange(-10,12.5,2.5))
ax1.set_yticklabels(np.arange(-10,12.5,2.5),fontsize=14)
ax1.set_zlabel('$f(x_1,x_2)$',fontsize=16)
ax1.set_zticks(np.arange(-3,4,1))
ax1.set_zticklabels(np.arange(-3,4,1),fontsize=14)
# Separation line
ax.plot([-0.30, -0.30], [0.0, 1.0], transform=ax.transAxes, clip_on=False,color="black")
# Plot
plt.subplots_adjust(wspace = 0.5)
fig = plt.gcf()
fig.set_size_inches(21.683, 9.140)
file_name = 'Figure1.png'
plt.savefig(file_name,format='png',dpi=600,bbox_inches='tight')
