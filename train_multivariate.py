import pandas as pd
import numpy as np

data = pd.read_csv("train_data.csv")

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

N=len(y)
d = len(x[0])

m = [1 for j in range(d)]  
b = 1                      
alpha = 0.0000001
b_grad = 0.0
m_grad = [0 for j in range(d)]
max_iter=20000

for _ in range(max_iter):
    b_grad = 0.0
    m_grad = [0.0 for _ in range(d)]
    for i in range(N):
        y_pred_i = 0.0
        for k in range(d):
            y_pred_i += x[i][k] * m[k]
        y_pred_i += b
        err = y[i] - y_pred_i

        b_grad += (-2.0) * err
        for k in range(d):
            m_grad[k] += (-2.0) * x[i][k] * err
    b = b- alpha * (b_grad / N)
    for k in range(d):
        m[k] = m[k]- alpha * (m_grad[k] / N)



print("m:", m)
print("New bias (b):", b)

sum_sq_error = 0
for i in range(N):
    y_pred_i = sum(x[i][k] * m[k] for k in range(d)) + b
    sum_sq_error += (y[i] - y_pred_i)**2

mse = sum_sq_error / N

ve = 1 - (mse / np.var(y))

print("Training MSE: ", mse)
print("Training VE: ",ve)