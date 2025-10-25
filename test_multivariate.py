import pandas as pd
import numpy as np
data = pd.read_csv("test_data.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


m= [0.088, 0.07, 0.03, -0.07, 0.92, 0.01, -0.01, 0.11]
b=0.997

N=len(y)
d = len(x[0])

sum_sq_error = 0
for i in range(N):
    y_pred_i = 0.0
    for k in range(d):
        y_pred_i += x[i][k] * m[k]
    y_pred_i += b
    sum_sq_error += (y[i] - y_pred_i) ** 2

mse = sum_sq_error / N

ve = 1 - (mse / np.var(y))

print("Training MSE: ", mse)
print("Training VE: ",ve)