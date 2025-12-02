import numpy as np

#linearregression class
class LinearRegression:
    def init(self, lr = 0.001, n_iters = 1000):
        self.lr = lr #learningrate
        self.n_iters = n_iters #qani angam e krknvum
        self.weights = None #modeli qashery
        self.bias = None #sheghumy

        def fit(self, X, y):
            n_samples, n_features = X.shape #orinaknery, hatkutyunnery
            self.weights = np.zeros(n_features)
            self.bias = 0

            for i in range(self.n_iters):
                y_pred = np.dot(X, self.weights) + self.bias

                dw = (1/n_samples) * np.dot(X.T, (y_pred-y)) #gradient descent
                db = (1/n_samples) * np.sum(y_pred-y)

                self.weights = self.weights - self.lr * dw
                self.bias = self.bias - self.lr * db

                def predict(self, X):
                    y_pred = np.dot(X< self.weights) + self.bias
                    return y_pred