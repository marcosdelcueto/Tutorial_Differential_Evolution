#!/usr/bin/env python3
# Marcos del Cueto
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
######################################################################################################
def main():
    # Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 2
    x1,x2,f = generate_data(-10,10,1.0,2)
    # Prepare X and y for KRR
    X,y = prepare_data_to_KRR(x1,x2,f)
    # Create hyperparams grid
    f_out = open('results_grid.dat', 'w')
    graph_x,graph_y,graph_z = create_hyperparams_grid(X,y,f_out)
    f_out.close()
    # Plot hyperparams_grid
    plot_hyperparams_grid(graph_x,graph_y,graph_z)
######################################################################################################
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
######################################################################################################
def prepare_data_to_KRR(x1,x2,f):
    X = []
    for i in range(len(f)):
        for j in range(len(f)):
            X_term = []
            X_term.append(x1[i][j])
            X_term.append(x2[i][j])
            X.append(X_term)
    y=f.flatten()
    X=np.array(X)
    y=np.array(y)
    return X,y
######################################################################################################
def KRR_function(hyperparams,X,y,f_out):
    # Assign hyper-parameters
    alpha_value,gamma_value = hyperparams
    # Split data into test and train: random state fixed for reproducibility
    kf = KFold(n_splits=10,shuffle=True,random_state=2020)
    y_pred_total = []
    y_test_total = []
    # kf-fold cross-validation loop
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Scale X0train and X_test
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        # Fit KRR with (X_train_scaled, y_train), and predict X_test_scaled
        KRR = KernelRidge(kernel='rbf',alpha=alpha_value,gamma=gamma_value)
        y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
        # Append y_pred and y_test values of this k-fold step to list with total values
        y_pred_total.append(y_pred)
        y_test_total.append(y_test)
    # Flatten lists with test and predicted values
    y_pred_total = [item for sublist in y_pred_total for item in sublist]
    y_test_total = [item for sublist in y_test_total for item in sublist]
    # Calculate error metric of test and predicted values: rmse
    rmse = np.sqrt(mean_squared_error(y_test_total, y_pred_total))
    #print('alpha: %.6f . gamma: %.6f . rmse: %.6f' %(alpha_value,gamma_value,rmse)) # Uncomment to print intermediate results
    # Print results to f_out file, since grid is expensive to recompute
    f_out.write("alpha: %.6f . gamma: %.6f . rmse: %.6f\n" %(alpha_value,gamma_value,rmse))
    return rmse
######################################################################################################
def create_hyperparams_grid(X,y,f_out):
    graph_x = []
    graph_y = []
    graph_z = []
    for alpha_value in np.arange(-5.0,2.2,0.2):
        alpha_value = pow(10,alpha_value)
        graph_x_row = []
        graph_y_row = []
        graph_z_row = []
        for gamma_value in np.arange(0.0,20.1,0.1):
            hyperparams = (alpha_value,gamma_value)
            rmse = KRR_function(hyperparams,X,y,f_out)
            graph_x_row.append(alpha_value)
            graph_y_row.append(gamma_value)
            graph_z_row.append(rmse)
        graph_x.append(graph_x_row)
        graph_y.append(graph_y_row)
        graph_z.append(graph_z_row)
    return graph_x,graph_y,graph_z
######################################################################################################
def plot_hyperparams_grid(graph_x,graph_y,graph_z):
    plt.xscale('log')
    contour=plt.contourf(graph_x, graph_y, graph_z, levels=np.arange(1.0,2.0,0.05),cmap='Greys',vmin=1.1,vmax=2.0,extend='both',zorder=0)
    contour_lines=plt.contour(graph_x, graph_y, graph_z, levels=np.arange(1.0,2.0,0.05),linewidths=1,colors='k',vmin=1.1,vmax=2.0,extend='both',zorder=1)
    plt.clabel(contour_lines,levels=np.arange(1.0,1.7,0.1),inline=1,colors="C0",fontsize=8,fmt='%1.1f')
    cbar=plt.colorbar(contour)
    cbar.set_label("$RMSE$", fontsize=14)
    plt.xlabel(r'$\alpha$',fontsize=14)
    plt.ylabel(r'$\gamma$',fontsize=14)
    file_name = 'Figure_hyperparams_grid.png'
    plt.savefig(file_name,format='png',dpi=600)
######################################################################################################
main()
