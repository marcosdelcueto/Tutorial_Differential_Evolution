#!/usr/bin/env python3
# Marcos del Cueto
import math
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
x1 = np.arange(-5,5.01,1.0)
y1 = np.arange(-5,5.01,1.0)
x1, y1 = np.meshgrid(x1, y1)
f1 = np.sin(x1) + np.cos(y1)


#print(f)
print(len(f1), len(f1[0]))
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

print('x1:')
print(x1)
print('y1:')
print(y1)

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

print('X:')
print(X)
print(len(X))
print('####')

print('y:')
print(y)
print(len(y))
print('####')
#########################
kf = KFold(n_splits=5,shuffle=True,random_state=0)
validation=kf.split(X)
#validation=loo.split(x1)
for train_index, test_index in validation:
    #print('train:', train_index)
    #print('test:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print('NEW kf cycle')
    #print('X_train')
    #print(X_train)
    #print('X_test')
    #print(X_train)
    # scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # fit KRR with (X_train, y_train), and predict X_test
    KRR = KernelRidge(kernel='rbf')
    y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
    print('NEW kf cycle')
    #print('test:',y_test)
    #print('pred:',y_pred)S
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_pearson,_=pearsonr(y_test,y_pred)
    print('rmse: %.3f.  r: %.3f' %(rmse,r_pearson))
    print('#################')

