import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyMultipleLinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)

        #  (X^T * X)^-1 * X^T * y
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.weights)

    def evaluate_performance(self, y_true, y_pred):
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true)**2)
        rmse = np.sqrt(mse)
        total_variance = np.sum((y_true - np.mean(y_true))**2)
        explained_variance = np.sum((y_pred - np.mean(y_true))**2)
        r2 = explained_variance / total_variance

        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Score': r2}

data = pd.read_csv('50_Startups.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

states = X[:, -1]
unique_states = np.unique(states)
encoded_states = np.zeros((len(states), len(unique_states)), dtype=float)

for i in range(len(states)):
    state_index = np.where(unique_states == states[i])[0][0]
    encoded_states[i, state_index] = 1

X = np.delete(X, -1, axis=1)
X = np.concatenate((X, encoded_states), axis=1)

X = X.astype(float)
y = y.astype(float)

def train_test_split_custom(X, y, test_size=0.2):
    split_index = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2)

model = MyMultipleLinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results_df)

performance_metrics = model.evaluate_performance(y_test, y_pred)
for metric, value in performance_metrics.items():
    print(f'{metric}: {value}')
