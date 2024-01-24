import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RidgeRegression:

    def __init__(self, learning_rate, iterations, alpha):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.alpha = alpha

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.iterations):
            self.update_weights()
            print(f"Iteration {i + 1} - W: {self.W}, b: {self.b}")

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = (-2 * np.dot(self.X.T, (self.Y - Y_pred)) + 2 * self.alpha * self.W) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.W) + self.b

def main():
    df = pd.read_csv("student_scores.csv")
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    # Manually split the data into train and test sets
    split_ratio = 0.33
    split_index = int(len(X) * (1 - split_ratio))

    X_train, Y_train = X[:split_index], Y[:split_index]
    X_test, Y_test = X[split_index:], Y[split_index:]

    model = RidgeRegression(iterations=1000, learning_rate=0.01, alpha=18)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    print("Predicted values ", np.round(Y_pred[:3], 2))
    print("Real values      ", Y_test[:3])
    print("Trained W        ", round(model.W[0], 2))
    print("Trained b        ", round(model.b, 2))

    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred, color='red')
    plt.title('Scores vs Hours of Study')
    plt.xlabel('Hours of Study')
    plt.ylabel('Scores')
    plt.show()

if __name__ == "__main__":
    main()