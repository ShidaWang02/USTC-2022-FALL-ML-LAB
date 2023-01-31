import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression():

    loss = []
    epoch = 0
    gamma = 0
    fit_intercept = 0

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.w = []
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.epoch = 0
        self.loss = []

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        z = 1 / (1 + np.exp(-x))
        return z

    def grad(self, X, y, w):
        if self.penalty == "l1":
            return (np.dot(np.transpose(X), self.sigmoid(np.dot(X, w))-y) + self.gamma * np.sign(w)) / X.shape[0]
        if self.penalty == "l2":
            return (np.dot(np.transpose(X), self.sigmoid(np.dot(X, w))-y) + self.gamma * w) / X.shape[0]

    def lossFunction(self, X, y, w):
        loss = 0
        if self.penalty == "l1":
            loss = - np.dot(np.transpose(y), np.dot(X, w)) + np.dot(self.temp, np.log(
                1+np.exp(np.dot(X, w)))) + \
                self.gamma * np.dot(np.ones([w.shape[1],w.shape[0]]),np.abs(w))
        if self.penalty == "l2":
            loss = - np.dot(np.transpose(y), np.dot(X, w)) + np.dot(self.temp, np.log(
                1+np.exp(np.dot(X, w)))) + self.gamma * np.dot(np.transpose(w), w) / 2
            '''
            loss = -(np.dot(np.transpose(y), np.log(self.sigmoid(np.dot(X, w)))) +
                     np.dot((1-np.transpose(y)), np.log(1-self.sigmoid(np.dot(X, w)))))

            for i in list(range(y.shape[0])):
                loss += (-y[i]*np.dot(X[i], w) + \
                         np.log(1 + np.exp(np.dot(X[i], w))))
            '''

        return loss / y.shape[0]

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent or other methods
        """
        self.temp = np.ones([1, y.shape[0]])
        if self.fit_intercept == True:
            self.w = np.zeros((X.shape[1] + 1, 1))  # 将偏置b并入w中
            X = np.column_stack((X, np.ones((X.shape[0], 1))))  # 同时在X中添加一列1向量
        else:
            self.w = np.zeros((X.shape[1], 1))
        self.epoch = 0
        while True:
            grad = self.grad(X, y, self.w)
            TempLoss = self.lossFunction(X, y, self.w)
            self.loss.append(TempLoss.A[0][0])
            self.w += - grad * lr
            self.epoch += 1
            if self.epoch > max_iter or np.abs(grad).max() < tol:
                #print('epoch:', self.epoch)
                # print('loss:', self.loss)
                return

    def ShowLoss(self):
        x = list(range(self.epoch))
        plt.figure(dpi=300)
        plt.plot(x, self.loss, 'r-', alpha=0.9)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot()
        plt.show()

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        if self.fit_intercept == True:
            X = np.column_stack((X, np.ones((X.shape[0], 1))))
        pred = self.sigmoid(np.dot(X, self.w))
        for i in list(range(len(pred))):
            if pred[i][0] >= 0.5:
                pred[i][0] = 1
            else:
                pred[i][0] = 0
        return pred



class SVM1:
    w = []
    b = 0
    C = 1
    epoch = 0

    def __init__(self, dim, C):
        """
        You can add some other parameters, which I think is not necessary
        """
        self.C = C

    def grad(self, X, y):
        wGrad = np.zeros([self.w.shape[0], 1])
        bGrad = 0
        for i in range(y.shape[0]):
            t = 0
            if y[i]*(np.dot(X[i], self.w)+self.b) < 1:
                t = 1
            else:
                t = 0
            wGrad += t*y[i][0]*np.transpose(X[i])
            bGrad += t
        wGrad = - wGrad / y.shape[0]
        bGrad = - bGrad / y.shape[0]
        wGrad += self.C*self.w
        return wGrad, bGrad

    def fit(self, X, y, lr=1, max_iter=500, tol=1e-4):
        """
        Fit the coefficients via your methods
        """
        X = np.asmatrix(X)
        self.w = np.ones([X.shape[1], 1])
        b = 1
        self.epoch = 0
        LR = lr
        while True:
            wGrad, bGrad = self.grad(X, y)
            self.w -= LR * wGrad
            self.b -= LR * bGrad
            self.epoch += 1
            if LR > 0.002:
                LR = lr / self.epoch
            else:
                LR=LR
            if self.epoch > max_iter or (np.abs(wGrad).max() < tol and bGrad < tol):
                return

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        pred = np.sign(np.dot(X, self.w) + self.b)
        return pred


class SVM2:
    w = []
    alpha = []
    C = 1
    epoch = 0

    def __init__(self, dim, C=1.0):
        """
        You can add some other parameters, which I think is not necessary
        """
        self.C = C

    def update(self, X, y, i):
        temp = self.alpha[i][0] - (y[i][0]*np.dot(X[i], self.w)-1) / \
            (np.dot(X[i], np.transpose(X[i])))
        if temp > self.C:
            return self.C
        elif temp <= self.C and temp >= 0:
            #return np.asarray(temp)[0][0]
            return temp
        else:
            return 0

    def fit(self, X, y, max_iter=1e6):
        """
        Fit the coefficients via your methods
        """
        X = np.column_stack((X, np.ones([X.shape[0], 1])))
        #X = np.asmatrix(X)
        #self.w = np.ones([X.shape[1], 1])
        self.alpha = np.zeros([X.shape[0], 1])
        for i in range(y.shape[0]):
            self.w = self.alpha[i][0]*y[i][0]*np.transpose(X[i])
        self.epoch = 0
        while True:
            for i in range(self.alpha.shape[0]):
                temp = self.update(X, y, i)
                self.w += (temp - self.alpha[i][0])*y[i][0]*np.transpose(X[i])
                self.alpha[i][0] = temp
                self.epoch += 1
                if self.epoch > max_iter:
                    return

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        X = np.column_stack((X, np.ones([X.shape[0], 1])))
        #X = np.asmatrix(X)
        pred = np.sign(np.dot(X, self.w))
        return pred
