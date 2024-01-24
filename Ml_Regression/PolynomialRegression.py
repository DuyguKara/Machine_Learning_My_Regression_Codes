import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.weights = None

    def create_polynomial_features(self, X):
        features = np.ones((len(X), 1))
        for i in range(1, self.degree + 1):
            features = np.concatenate((features, X ** i), axis=1)
        return features

    def fit(self, X, y):
        X_poly = self.create_polynomial_features(X)
        self.weights = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

    def predict(self, X):
        X_poly = self.create_polynomial_features(X)
        return X_poly.dot(self.weights)

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, -1].values  

poly_reg = PolynomialRegression(degree=4)
poly_reg.fit(X, y)

plt.scatter(X, y, color='blue')
plt.plot(X, poly_reg.predict(X), color='red')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

position_level = 6
predicted_salary = poly_reg.predict(np.array([[position_level]]))
print(f"Predicted salary for position level {position_level} is a ${predicted_salary[0]:.2f}")
