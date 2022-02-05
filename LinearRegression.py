import numpy as np
class LinearRegression:
    m, n = 0, 0
    W = 0
    y_hat = 0
    grad = 0
    X = 0
    y = 0
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.m, self.n = self.X.shape
        self.X = np.append(self.X, np.ones((self.m,1)), axis=1)
        self.W = np.zeros(self.n+1)
        self.__calc_theta()
    
    def __calc_theta(self):
        #self.__calc_grad()
        self.W = (np.linalg.inv(self.X.T @ self.X) @ self.X.T) @ self.y
        #self.W = self.W-self.grad
    '''
    def __calc_grad(self):
        self.grad = self.X.T @ (self.y_hat - self.y)
    '''
    def predict(self, X):
        m= X.shape[0]
        X = np.append(X, np.ones((m,1)), axis=1)
        y_hat = X @ self.W
        return y_hat