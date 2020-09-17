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
#x1 = np.arange(-5,5.01,1.0)
#y1 = np.arange(-5,5.01,1.0)
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
#print(len(f1), len(f1[0]))
# Print function
fig = plt.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, f,linewidth=0, antialiased=False,cmap='viridis')
ax.scatter(x1,y1,f1,s=50,zorder=4)
file_name = 'Figure2.png'
plt.savefig(file_name,format='png',dpi=600)
#plt.show()

#X = np.concatenate((x1,y1),axis=0)
#X = np.meshgrid(x1, y1)

#print('X:')
#print(X)
#print('####')

#print('x1:')
#print(x1)
#print('y1:')
#print(y1)

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

#print('X:')
#print(X)
#print(len(X))
#print('####')

#print('y:')
#print(y)
#print(len(y))
#print('####')
#########################

#validation=loo.split(x1)

for alpha_value in [pow(10,-12),pow(10,-11),pow(10,-10),pow(10,-9),pow(10,-8),pow(10,-7),pow(10,-6),pow(10,-5),pow(10,-4),pow(10,-3),pow(10,-2),pow(10,-1),pow(10,0),pow(10,1)]:
    for gamma_value in np.arange(0.000,1.0,0.002):
#for alpha_value in np.arange(0.00000001,0.000001,0.00000001):
    #for gamma_value in np.arange(0.000,0.100,0.001):
        kf = KFold(n_splits=10,shuffle=True,random_state=None)
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
        print('%.20f %.20f %.12f %.12f' %(alpha_value,gamma_value,rmse,r_pearson))
        #print('#################')
        
