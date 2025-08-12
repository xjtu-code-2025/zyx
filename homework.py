import numpy as np

class linear_classfication:
    def __init__(self):
        pass

    def fit(self,X,Y):
        X =  np.column_stack([np.ones(X.shape[0]), X])
        self.beta_hat = np.linalg.inv(X.T @ X) @ (X.T) @ Y

    def predict(self,new_x):
        new_x = np.array(new_x).reshape(-1, 1) if np.isscalar(new_x) else np.array(new_x)
        new_x_with_intercept = np.column_stack([np.ones(new_x.shape[0]), new_x])
        return new_x_with_intercept @ self.beta_hat



X = np.array([[1,23], [2,32], [3,456], [4,5], [5,66]])
Y = np.array([3, 5, 7, 9, 11])           # 目标值

model = linear_classfication()
model.fit(X,Y)
print(model.beta_hat)
z = model.predict([[455,1000000]])
print(z)


