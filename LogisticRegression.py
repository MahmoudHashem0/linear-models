import numpy as np
class LogisticRegression:
    m, n = 0, 0
    theta = 0
    yHat = 0
    grad = 0
    X = 0
    y = 0
    alpha = 0
    max_iter = 0
    def __init__(self, alpha, max_iter= 1000):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.m, self.n = self.X.shape
        self.X = np.append(self.X, np.ones((self.m,1)), axis=1)
        self.theta = np.zeros((self.n+1, 1))
        self.yHat = self.__calc_yHat(self.X)
        self.__calc_theta()
    
    def __calc_theta(self):
        for _ in range(self.max_iter):
            self.__calc_grad()
            new_theta = self.theta + self.alpha * self.grad
            if self.__converged(new_theta):
                break
            self.theta = new_theta
        else:
            print("doesn't converge")


    def __converged(self, new_theta):
        condition1 = abs(self.grad) <= 10**-2
        condition2 = abs(self.theta - new_theta) <= 10**-6
        return condition1.all() or condition2.all()

    def __calc_yHat(self, X):
        yHat =  self.__sigmoid(X @ self.theta)
        return yHat
    
    def __sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def __calc_grad(self):
        self.yHat = self.__calc_yHat(self.X)
        self.grad = self.X.T @ (self.y - self.yHat)
    
    def predict(self, X):
        m = X.shape[0]
        X = np.append(X, np.ones((m,1)), axis=1)
        yHat = self.__calc_yHat(X)
        return (yHat[:,0]+0.5).astype(int)