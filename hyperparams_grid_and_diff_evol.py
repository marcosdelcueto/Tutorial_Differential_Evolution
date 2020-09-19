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
from scipy.optimize import differential_evolution

######################################################################################################
def KRR_function(hyperparams,X,y):
    alpha_value,gamma_value = hyperparams
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
    f_provi.write("%.20f %.20f %.12f %.12f\n" %(alpha_value,gamma_value,rmse,r_pearson))
    #print('#################')
    return rmse
######################################################################################################
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

# Perform differential evolution
KRR_alpha_lim = (0.00001,100.0)
KRR_gamma_lim = (0.00001,20.0)
bounds = [KRR_alpha_lim] + [KRR_gamma_lim]
mini_args = (X,y)
f_provi = open('provi.dat', 'w')
print('Starting differential evolution algorithm')
solver = differential_evolution(KRR_function,bounds,args=mini_args,popsize=15,tol=0.01,seed=2020)
best_hyperparams = solver.x
best_rmse = solver.fun
print("Best hyperparameters: %s" %(str(best_hyperparams)))
print(type(best_hyperparams), best_hyperparams[0])
print("Best rmse: %s" %(str(best_rmse)))
f_provi.close()
# Read intermediate values from differential evolution (for plotting)
read_x = []
read_y = []
read_z = []
f_provi = open('provi.dat', 'r')
for line in f_provi:
    read_x.append(float(line.split()[0].strip()))
    read_y.append(float(line.split()[1].strip()))
    read_z.append(float(line.split()[2].strip()))
# Read values from code2: grid search of hyperparams (for plotting)
f_results_grid = open('results_grid.dat', 'r')
graph_x = []
graph_y = []
graph_z = []
for line in f_results_grid:
    graph_x.append(float(line.split()[0].strip()))
    graph_y.append(float(line.split()[1].strip()))
    graph_z.append(float(line.split()[2].strip()))
graph_x = [graph_x[x:x+201] for x in range(0, len(graph_x), 201)]
graph_y = [graph_y[x:x+201] for x in range(0, len(graph_y), 201)]
graph_z = [graph_z[x:x+201] for x in range(0, len(graph_z), 201)]
# Plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel(r'$\alpha$',fontsize=14)
ax1.set_ylabel(r'$\gamma$',fontsize=14)
# Plot dense grid
plt.xscale('log')
contour=ax1.contourf(graph_x, graph_y, graph_z, levels=np.arange(1.0,2.0,0.05),cmap='Greys',vmin=1.1,vmax=2.0,extend='both',zorder=0)
contour_lines=ax1.contour(graph_x, graph_y, graph_z, levels=np.arange(1.0,2.0,0.05),linewidths=1,colors='k',vmin=1.1,vmax=2.0,extend='both',zorder=1)
plt.clabel(contour_lines,levels=np.arange(1.0,1.7,0.1),inline=1,colors="C0",fontsize=8,fmt='%1.1f')
# Plot genetic algorithm values
evolution=ax1.scatter(read_x,read_y,c=read_z,cmap='viridis',vmin=1.1,vmax=1.6,s=10,zorder=2)
cbar=fig.colorbar(evolution,ax=ax1)
cbar.set_label("$RMSE$", fontsize=14)
# Plot best combination of hyperparams
ax1.scatter(best_hyperparams[0],best_hyperparams[1],c='red',zorder=3)
# Save file
file_name = 'Figure3.png'
plt.savefig(file_name,format='png',dpi=600)
