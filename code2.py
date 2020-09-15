#!/usr/bin/env python3
# Marcos del Cueto
import math
import numpy as np
import matplotlib.pyplot as plt

# Calculate underlying grid
x = np.arange(-5,5.01,0.01)
y = np.arange(-5,5.01,0.01)
x, y = np.meshgrid(x, y)
f = np.sin(x) + np.cos(y)

# Points to make database
x1 = np.arange(-5,5.01,1.0)
y1 = np.arange(-5,5.01,1.0)
x1, y1 = np.meshgrid(x1, y1)
f1 = np.sin(x1) + np.cos(y1)


#print(f)
#print(len(f), len(f[0]))
# Print function
fig = plt.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, f,linewidth=0, antialiased=False,cmap='viridis')
ax.scatter(x1,y1,f1,s=50,zorder=4)
file_name = 'Figure2.png'
#plt.savefig(file_name,format='png',dpi=600)
plt.show()
