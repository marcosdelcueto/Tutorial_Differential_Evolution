#!/usr/bin/env python3
# Marcos del Cueto
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
######################################################################################################
def main():
    # Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 2
    x1,x2,f = generate_data(-10,10,1.0,2)
    # Prepare X and y for KRR
    X,y = prepare_data_to_KRR(x1,x2,f)
    # Set limits for Differential Evolution
    f_provi = open('provi.dat', 'w')
    KRR_alpha_lim = (0.00001,100.0)
    KRR_gamma_lim = (0.00001,20.0)
    boundaries = [KRR_alpha_lim] + [KRR_gamma_lim]
    extra_variables = (X,y,f_provi)
    # Set up Differential Evolution solver

    solver = differential_evolution(KRR_function,boundaries,args=extra_variables,strategy='best1bin',
                                    popsize=15,mutation=0.5,recombination=0.7,tol=0.01,seed=2020)
    # Calculate best hyperparameters and resulting rmse
    best_hyperparams = solver.x
    best_rmse = solver.fun
    # Print final results
    print("Converged hyperparameters: alpha= %.6f, gamma= %.6f" %(best_hyperparams[0],best_hyperparams[1]))
    print("Minimum rmse: %.6f" %(best_rmse))
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
    f_provi.close()
    # Read values from results_grid: grid search of hyperparams (for plotting)
    f_results_grid = open('results_grid.dat', 'r')
    graph_x = []
    graph_y = []
    graph_z = []
    for line in f_results_grid:
        graph_x.append(float(line.split()[0].strip()))
        graph_y.append(float(line.split()[1].strip()))
        graph_z.append(float(line.split()[2].strip()))
    f_results_grid.close()
    graph_x = [graph_x[x:x+201] for x in range(0, len(graph_x), 201)]
    graph_y = [graph_y[x:x+201] for x in range(0, len(graph_y), 201)]
    graph_z = [graph_z[x:x+201] for x in range(0, len(graph_z), 201)]
    plot_DE(read_x,read_y,read_z,graph_x,graph_y,graph_z,best_hyperparams)
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
def KRR_function(hyperparams,X,y,f_provi):
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
        # Scale X_train and X_test
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
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
    print('alpha: %.6f . gamma: %.6f . rmse: %.6f' %(alpha_value,gamma_value,rmse)) # Uncomment to print intermediate results
    f_provi.write("%.20f %.20f %.12f\n" %(alpha_value,gamma_value,rmse))
    return rmse
######################################################################################################
def plot_DE(read_x,read_y,read_z,graph_x,graph_y,graph_z,best_hyperparams):
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
######################################################################################################
main()
