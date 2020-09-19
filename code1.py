#!/usr/bin/env python3
# Marcos del Cueto
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-10,10.01,0.1)
y = np.arange(-10,10.01,0.1)
x, y = np.meshgrid(x, y)
f = np.sin(x) + np.cos(y)

# Points to make database
x1 = np.arange(-10,10.01,1.0)
y1 = np.arange(-10,10.01,1.0)
x1, y1 = np.meshgrid(x1, y1)
f1 = np.sin(x1) + np.cos(y1)

random.seed(2020)
for i in range(len(f1)):
    for j in range(len(f1[0])):
        rnd_number = random.uniform(-2,2)
        #print('RND:', rnd_number)
        f1[i][j] = f1[i][j] + rnd_number
#print(f)
#print(len(f), len(f[0]))


#######################################
## Print function
#fig = plt.figure()
#ax = fig.gca(aspect='equal')
#surf = ax.scatter(x1, y1, c=f1,cmap='viridis',s=10,zorder=4)
#ax.contourf(x, y, f, cmap='Greys',zorder=0)

#file_name = 'Figure1.png'
#plt.savefig(file_name,format='png',dpi=600)
#######################################

#fig, axs = plt.subplots(1,2,sharex=True,sharey=True)
#axs[0].set(adjustable='box', aspect='equal',projection='3d')
#axs[0].surface(x,y,f,c=f1,cmap='viridis',s=10,zorder=4)

#axs[1].set(adjustable='box', aspect='equal')
#axs[1].scatter(x1, y1, c=f1,cmap='viridis',s=10,zorder=4)
#axs[1].contourf(x, y, f, cmap='Greys',zorder=0)


#file_name = 'Figure1.png'
#plt.savefig(file_name,format='png',dpi=600)

fig = plt.figure()
# Right subplot
ax = fig.add_subplot(1, 2,2)
ax.set(adjustable='box', aspect='equal')
ax.contourf(x, y, f, cmap='Greys',zorder=0)
points = ax.scatter(x1, y1, c=f1,cmap='viridis',s=15,zorder=1)
cbar=plt.colorbar(points)
cbar.set_label("$f(x)$",fontsize=16)
cbar.ax.tick_params(labelsize=14)
ax.set_xlabel('$x$',fontsize=16)
ax.set_xticks(np.arange(-10,12.5,2.5))
ax.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
ax.set_ylabel('$y$',fontsize=16)
ax.set_yticks(np.arange(-10,12.5,2.5))
ax.set_yticklabels(np.arange(-10,12.5,2.5),fontsize=14)
# Left subplot
ax1 = fig.add_subplot(1, 2, 1,projection='3d')
surf = ax1.plot_surface(x, y, f, rstride=1, cstride=1,linewidth=0, antialiased=False,cmap='Greys',zorder=0)
ax1.scatter(x1, y1, f1,c=f1,cmap='viridis',s=15,zorder=1)
ax1.set_xlabel('$x$',fontsize=16)
ax1.set_xticks(np.arange(-10,12.5,2.5))
ax1.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
ax1.set_ylabel('$y$',fontsize=16)
ax1.set_yticks(np.arange(-10,12.5,2.5))
ax1.set_yticklabels(np.arange(-10,12.5,2.5),fontsize=14)
ax1.set_zlabel('$z$',fontsize=16)
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
