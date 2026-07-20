import numpy as np 
#import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = None
        self.r2score_ = None

    def fit(self, X, y):
        n = len(X)
        print(f'Number of samples: {n}')
        X_b = np.c_[np.ones((n, 1)), X] # Add bias term (intercept) to the input features

        self.coefficients_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # Calculate coefficients using the normal equation
        self.intercept_ = self.coefficients_[0] # Intercept is the first coefficient
        y_pred = X_b.dot(self.coefficients_) # Predicted values

        self.r2score_ = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)) # Calculate R-squared score
        self.y_pred_ = y_pred # Store predicted values

    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X] # Add bias term (intercept) to the input features
        return X_b.dot(self.coefficients_) # Return predicted values

X_simple = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y_simple = np.array([2,4,5,4,5,7,8,9,10,12])

model = LinearRegression()
model.fit(X_simple, y_simple)

print(f'Coefficients: {model.coefficients_}')
print(f'Intercept: {model.intercept_}')
print(f'R-squared: {model.r2score_}')

plt.scatter(X_simple, y_simple, color='blue', label='Data')
plt.plot(X_simple, model.predict(X_simple), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()