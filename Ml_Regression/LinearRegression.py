import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Salary_Data.csv', delimiter=',', skip_header=1)
X = data[:, 0]
y = data[:, 1] 

def fit(X, y, learning_rate=0.01, epochs=1000):
   
    m, b = 0, 0
    n = len(X)

    for epoch in range(epochs):
        
        y_pred = m * X + b

        error = np.mean((y_pred - y) ** 2)

        m -= learning_rate * (2 / n) * np.dot(X, (y_pred - y))
        b -= learning_rate * (2 / n) * np.sum(y_pred - y)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Error={error}')

    return m, b

m, b = fit(X, y)

def predict(X, m, b):
    return m * X + b

y_pred = predict(X, m, b)

plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()