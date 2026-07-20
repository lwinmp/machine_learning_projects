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

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients_ = None 
        self.intercept_ = None     
        self.r2score_ = None    
        
    def fit(self, X, y):
        n = X.shape[0]
        X_b = np.c_[np.ones((n,1)), X]  
        self.coefficients_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = self.coefficients_[0]
        y_pred = X_b.dot(self.coefficients_)
        self.r2score_ = 1 - (np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2))
        self.y_pred_ = y_pred

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.coefficients_)

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
plt.title('Single Linear Regression')
plt.legend()
#plt.show()

np.random.seed(0)
X1 = np.random.randint(1, 11, 15)
X2 = np.random.randint(1, 11, 15)
X_multi = np.column_stack((X1, X2))
y_multi = 1 + 2*X1 + 3*X2 + np.random.randn(15) * 2

model_multi = MultipleLinearRegression()
model_multi.fit(X_multi, y_multi)

print(f'Coefficients: {model_multi.coefficients_}')
print(f'Intercept: {model_multi.intercept_}')
print(f'R-squared: {model_multi.r2score_}')

model_lr = MultipleLinearRegression()
model_lr.fit(X_multi, y_multi)
print(f'Coefficients: {model_lr.coefficients_}')
print(f'Intercept: {model_lr.intercept_}')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_multi[:,0], X_multi[:,1], y_multi, color='blue', label='Data')
x1_surf, x2_surf = np.meshgrid(np.linspace(X_multi[:,0].min(), X_multi[:,0].max(), 10),
                               np.linspace(X_multi[:,1].min(), X_multi[:,1].max(), 10))
pred_surf = model_lr.predict(np.c_[x1_surf.ravel(), x2_surf.ravel()]).reshape(x1_surf.shape)
ax.plot_surface(x1_surf, x2_surf, pred_surf, color='red', alpha=0.5, rstride=1, cstride=1)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plt.title('Multiple Linear Regression')
plt.legend()
plt.show()