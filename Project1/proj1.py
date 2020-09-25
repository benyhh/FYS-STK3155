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
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
np.random.seed(100)

class reg_analysis():
    def __init__(self, n):
        #n grid points
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        self.x, self.y = np.meshgrid(x,y)

        #Parameters
        self.max_degree = 20
        self.noise = 0.01
        self.test_size = 0.2
        noise1 = self.noise * np.random.randn(np.size(self.x)).reshape(len(x),len(x))
        self.z = self.FrankeFunction(self.x, self.y) + noise1


    def bootstrap(self, n_bootstraps): #n_b #bootstraps
        MSE = np.zeros((self.max_degree,n_bootstraps))
        Bias = np.zeros((self.max_degree,n_bootstraps))
        X = self.X_design(self.x, self.y, self.max_degree)
        X_train, X_test, z_train, z_test = train_test_split(X, self.z.ravel(), test_size=self.test_size)
        X_test -= np.mean(X_test)

        degree = np.asarray([d+1 for d in range(self.max_degree)])
        for d in range(1, self.max_degree + 1):
            X = self.X_design(self.x, self.y, self.max_degree)
                

            for i in range(n_bootstraps):
                X_train, z_train = resample(X_train, z_train)
                X_train -= np.mean(X_test)
                beta = self.calc_beta(X)
                z_predict = X_test @ beta
                MSE[d-1][i] = np.mean((z_test-z_predict)**2)
                Bias[d-1][i] = np.mean((z_test - np.mean(z_predict))**2)

        MSE = np.mean(MSE, axis = 1)
        Bias = np.mean(Bias, axis = 1)
        plt.plot(degree, MSE, label = 'MSE')
        plt.plot(degree, Bias, label = 'Bias')
        plt.title("Bootstrap, n_b = %i" %(n_bootstraps))
        plt.legend()
        plt.show()

    def X_design(self, x,y, d):
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((d+1)*(d+2)/2) # Number of elements in beta
        X = np.ones((N,l))

        for i in range(1,d+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,q+k] = (x**(i-k))*(y**k)
        return X

    def calc_beta(self, X):
        beta = np.linalg.pinv(X.T @ X) @ X.T @ self.z.ravel()
        return beta

    def FrankeFunction(self, x, y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

test = reg_analysis(n = 40)
test.bootstrap(n_bootstraps = 200)
