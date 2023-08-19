import numpy as np

class perseptron :
    def  __init__(self , input_size ,lr=0.0001 , epochs=10):
        self.w = np.random.rand(input_size)
        self.b = np.random.rand(1) 
        self.lr = lr
        self.epochs = epochs

    def fit(self ,X_train ,Y_train):
        losses =[]
        for epoch in range(self.epochs):
            for i in range(len(X_train)) :
                x = X_train[i]
                y = Y_train[i]
                y_pred = x @ self.w + self.b
                error = y - y_pred
                self.w = self.w +( error * x *self.lr )
                self.b = self.b + ( error * self.lr)
                loss = np.sum(np.abs(error))
                losses.append(loss)

    def predict(self,X_test):
       Y_pred = X_test @ self.w + self.b
       return Y_pred


    def evaluate(self,X_test,Y_test):
        Y_pred = self.predict(X_test)
        return mean_absolute_error(Y_pred , Y_test)