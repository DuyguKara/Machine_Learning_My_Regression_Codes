import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression:
    def __init__(self):
        self.opt_w1 = None
        self.opt_wo = None
        self.best_error = None

    def fit(self, x_list, y_list):
        n = len(x_list)

        mean_x = np.mean(x_list)
        mean_y = np.mean(y_list)

        cov_xy = np.sum((x_list - mean_x) * (y_list - mean_y)) / n
        var_x = np.sum((x_list - mean_x) ** 2) / n

        self.opt_w1 = cov_xy / var_x
        self.opt_wo = mean_y - self.opt_w1 * mean_x

        my_y_pred = self.predict(x_list)
        error = np.mean((my_y_pred - y_list) ** 2)

        if self.best_error is None or error < self.best_error:
            self.best_error = error
            opt_w1_temp = self.opt_w1
            opt_wo_temp = self.opt_wo

        print("Epoch - Best Error:", self.best_error)
        print("Optimal w1:", opt_w1_temp)
        print("Optimal w0:", opt_wo_temp)

        return opt_w1_temp, opt_wo_temp

    def predict(self, x):
        return self.opt_w1 * x + self.opt_wo

if __name__ == "__main__":
    my_data = np.genfromtxt('Salary_Data.csv', delimiter=',', skip_header=1)
    my_x = my_data[:, 0]  
    my_y = my_data[:, 1]  

    my_model = MyLinearRegression()
    my_model.fit(my_x, my_y)

    my_y_pred = my_model.predict(my_x)

    plt.scatter(my_x, my_y, color='blue')
    plt.plot(my_x, my_y_pred, color='red')
    plt.title('My Dataset Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()