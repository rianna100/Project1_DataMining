import pandas as pd
import numpy as np
data = pd.read_csv("train_data.csv")

# Extract x (Cement) and y (Strength)
x = data["Cement (component 1)(kg in a m^3 mixture)"].values
y = data["Concrete compressive strength(MPa, megapascals) "].values
m = 1
b = 1
N = len(x)
alpha = 0.000001 #select learning hyperparameter
max_iter=20000 #select iteration amount
# Compute gradients
for i in range(max_iter):  
    b_gradient= 0
    m_gradient = 0
    for i in range(N):
         y_calculated = x[i]*m + b  #calculate y predicted
         error = y[i] - y_calculated
         b_gradient += (-2)*error
         m_gradient += (-2)*x[i]*error
    b = b - alpha * b_gradient / N
    m = m - alpha * m_gradient / N


print("m:", m)
print("b:", b)

sum=0

for i in range(N):
    y_pred = m * x[i] + b
    difference= (y[i] - y_pred)**2
    sum += difference
mse = sum/N

ve = 1 - (mse / np.var(y))

print("Training MSE: ", mse)
print("Training VE: ",ve)
