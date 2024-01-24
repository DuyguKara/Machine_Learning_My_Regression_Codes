import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LassoRegression:

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

        for _ in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = (-2 * np.dot(self.X.T, (self.Y - Y_pred)) + self.alpha * np.sign(self.W)) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m
        self.W = self.W - self.learning_rate * dW
        self.W[self.W < 0] = 0 
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

    model = LassoRegression(iterations=1000, learning_rate=0.01, alpha=100)
    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    print("Training Predictions:", np.round(Y_pred_train[:3], 2))
    print("Real Values (Training):", Y_train[:3])
    print("Testing Predictions:", np.round(Y_pred_test[:3], 2))
    print("Real Values (Testing):", Y_test[:3])
    print("Trained W:", model.W)
    print("Trained b:", model.b)

    plt.scatter(X_test, Y_test, color='blue')
    plt.plot(X_test, Y_pred_test, color='red')
    plt.title('Lasso Regression: Actual vs Predicted (Test Data)')
    plt.xlabel('Hours of Study')
    plt.ylabel('Scores')
    plt.show()

if __name__ == "__main__":
    main()