from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

class reg():
    def __init__(self, n):
        self.n = n
        x = np.arange(0, 1, 0.05)
        y = np.arange(0, 1, 0.05)
        self.x, self.y = np.meshgrid(x,y)
        self.z = self.FrankeFunction(self.x, self.y)
        self.N = np.size(x)
        self.p = int((n+1)*(n+2)/2)
        self.X = self.X_design(x,y,n)

        beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.z.ravel()
        z_fit = self.X @ self.beta
        z_fit = z_fit.reshape(np.shape(z))
        plot_surf(x,y,z)
        plot_surf(x,y, z_fit)

    def FrankeFunction(self, x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def X_design(self, x,y,n):
        X = np.zeros((self.N,self.p))
        x = x.ravel()
        y = y.ravel()
        k = 0
        for i in range(n+1):
            for j in range(n+1-i):
                X[:,k] = x.ravel()**j * y.ravel()**i
                k += 1
        return X

    def plot_surf(self, x,y,z):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    def mean_squared_error(self, z,z_fit):
        n = np.size(z)
        MSE = (z-z_fit)**2
        return np.mean(MSE)

    def R_squared(self, z,z_fit):
        z_mean = np.mean(z)
        SStot = (z-z_mean)**2
        SSres = (z-z_fit)**2

        R_sq = 1 - np.sum(SSres)/np.sum(SStot)
        return R_sq

deg5 = reg(5)


R2 = R_squared(z,z_fit)
MSE = mean_squared_error(z,z_fit)
