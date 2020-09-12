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

class reg():
    def __init__(self, n):
        x = np.arange(0, 1, 0.05)
        y = np.arange(0, 1, 0.05)
        self.x, self.y = np.meshgrid(x,y)

        # Parameters
        self.n = n
        self.N = np.size(self.x)
        self.p = int((n+1)*(n+2)/2)

        # Makes design matrix and z
        self.X = self.X_design(self.x,self.y,self.n)
        self.z = self.FrankeFunction(self.x, self.y)

        # Split the data in test and training data
        X_train, X_test, self.ztrain, self.ztest = train_test_split(self.X, self.z.ravel(), test_size=0.2)

        # Make beta with train data
        self.beta = self.beta(X_train, self.ztrain)

        # Make prediction
        self.ztilde = X_train @ self.beta
        self.zpredict = X_test @ self.beta

        #self.MSE = self.mean_squared_error(self.z,self.zfit)
        #self.R2 = self.R_squared(self.z,self.zfit)

    def comp(self, X, beta, scale=True):
        """
        Funksjon som regner z med gitt X, beta og scaling.
        Så finner den MSE og R2.
        (Men må kunne sammenlikne shit også kanskje?
        Eller kan jo faktisk bare calle flere ganger.)

        if scale == True:
            scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        z = X @ beta
        """

    def __call__(self):
        pass

    def print_data(self):
        MSEtrain = self.mean_squared_error(self.ztrain, self.ztilde)
        MSEtest = self.mean_squared_error(self.ztest, self.zpredict)
        R2train = self.R_squared(self.ztrain, self.ztilde)
        R2test = self.R_squared(self.ztest, self.zpredict)
        print("Polynomial degree %i:\n\
        Training MSE: %.5f , Test MSE: %.5f \n\
        Training R2:  %.5f , Test R2: %.5f" %(self.n, MSEtrain, MSEtest, R2train, R2test))

    def beta(self, X, z):
        X, z = self.X, self.z
        beta = np.linalg.inv(X.T @ X) @ X.T @ z.ravel()

        return beta

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

    def mean_squared_error(self, z1,z2):
        n = np.size(z1)
        MSE = (z1-z2)**2
        return np.mean(MSE)

    def R_squared(self, z,zfit):
        zmean = np.mean(z)
        SStot = (z-zmean)**2
        SSres = (z-zfit)**2
        R2 = 1 - np.sum(SSres)/np.sum(SStot)
        return R2

objects = np.array([reg(i) for i in range(2,6)])

for ob in objects:
    ob.print_data()
