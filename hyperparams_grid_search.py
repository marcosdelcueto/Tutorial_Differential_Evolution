#!/usr/bin/env python3
# Marcos del Cueto
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

# Calculate underlying grid
x = np.arange(-5,5.01,0.01)
y = np.arange(-5,5.01,0.01)
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
        f1[i][j] = f1[i][j] + rnd_number
# Print function
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(x1,y1,f1,s=50,zorder=4)
#file_name = 'Figure2.png'
#plt.savefig(file_name,format='png',dpi=600)

X = []
for i in range(len(f1)):
    for j in range(len(f1)):
        X_term = []
        X_term.append(x1[i][j])
        X_term.append(y1[i][j])
        X.append(X_term)

y=f1.flatten()
X=np.array(X)
y=np.array(y)
#########################
graph_x = []
graph_y = []
graph_z = []
for alpha_value in np.arange(-5.0,2.2,0.2):
    alpha_value = pow(10,alpha_value)
    graph_x_row = []
    graph_y_row = []
    graph_z_row = []
    for gamma_value in np.arange(0.0,20.1,0.1):
        kf = KFold(n_splits=10,shuffle=True,random_state=2020)
        validation=kf.split(X)
        #alpha_value = 1.0
        #gamma_value = 1.0
        y_pred_total = []
        y_test_total = []
        for train_index, test_index in validation:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # scale data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            # fit KRR with (X_train, y_train), and predict X_test
            KRR = KernelRidge(kernel='rbf',alpha=alpha_value,gamma=gamma_value)
            y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
            y_pred_total.append(y_pred)
            y_test_total.append(y_test)
        y_pred_total = [item for sublist in y_pred_total for item in sublist]
        y_test_total = [item for sublist in y_test_total for item in sublist]
        rmse = np.sqrt(mean_squared_error(y_test_total, y_pred_total))
        r_pearson,_=pearsonr(y_test_total,y_pred_total)
        #print('alpha: %.6f . gamma: %.6f . rmse: %.3f .  r: %.3f' %(alpha_value,gamma_value,rmse,r_pearson))
        #print('%.20f %.20f %.12f %.12f' %(alpha_value,gamma_value,rmse,r_pearson))
        graph_x_row.append(alpha_value)
        graph_y_row.append(gamma_value)
        graph_z_row.append(rmse)
        #print('#################')
    print('new alpha:', alpha_value)
    graph_x.append(graph_x_row)
    graph_y.append(graph_y_row)
    graph_z.append(graph_z_row)

plt.xscale('log')
contour=plt.contourf(graph_x, graph_y, graph_z, levels=np.arange(1.0,2.0,0.05),cmap='Greys',vmin=1.1,vmax=2.0,extend='both',zorder=0)
contour_lines=plt.contour(graph_x, graph_y, graph_z, levels=np.arange(1.0,2.0,0.05),linewidths=1,colors='k',vmin=1.1,vmax=2.0,extend='both',zorder=1)
plt.clabel(contour_lines,levels=np.arange(1.0,1.7,0.1),inline=1,colors="C0",fontsize=8,fmt='%1.1f')
cbar=plt.colorbar(contour)
cbar.set_label("$RMSE$", fontsize=14)
plt.xlabel(r'$\alpha$',fontsize=14)
plt.ylabel(r'$\gamma$',fontsize=14)
file_name = 'Figure2.png'
plt.savefig(file_name,format='png',dpi=600)
#plt.show()
