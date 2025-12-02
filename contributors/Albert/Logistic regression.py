import numpy as np

#sigmoid funkcia
def sigmoid(x):
    return 1/(1+np.exp(-x))

#logisticreggresion class
class LogisticRegression():
    def init(self, lr=0.001, n_iters=1000):
        self.lr = lr #inchqan arag petq e sovori modely
        self.n_iters = iters #qani angam krknvi usucman cikly
        self.weights = None #qashery
        self.bias = None #sheghumy

        n_samples, n_features = X.shape #tvyalneri u hatkanishneri qanaky
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters): #gradient descent
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) #qasheri popoxutyun
            db = (1/n_samples) * np.sum(predictions - y) #bias-i popoxutyun

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            linear_pred = np.dot(X, self.weights) + self.bias #gcayin kanxatesum
            y_pred = sigmoid(linear_pred) #sigmoid havanakanutyun stanalu hamar

            class_pred = [0 if y <= 0.5 else 1 for y in y_pred] #havanakanutyuny dardznum e 0 kam 1