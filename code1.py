#!/usr/bin/env python3
# Marcos del Cueto
import math
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5.01,0.01)
y = np.arange(-5,5.01,0.01)
x, y = np.meshgrid(x, y)
f = np.sin(x) + np.cos(y)
#print(f)
#print(len(f), len(f[0]))
# Print function
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, y, f,linewidth=0, antialiased=False,cmap='viridis')
file_name = 'Figure1.png'
plt.savefig(file_name,format='png',dpi=600)
