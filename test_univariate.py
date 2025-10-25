import pandas as pd
import numpy as np
data = pd.read_csv("test_data.csv")

# Extract x (Cement) and y (Strength)
x = data["Age (day)"].values
print("Age (day)")
y = data["Concrete compressive strength(MPa, megapascals) "].values
m= 0.34
b = 1.79
N = len(x)
alpha = 0.1
sum=0

for i in range(N):
    y_pred = m * x[i] + b
    difference= (y[i] - y_pred)**2
    sum += difference
mse = sum/N

ve = 1 - (mse / np.var(y))

print("MSE: ", mse)
print("VE:", ve)
