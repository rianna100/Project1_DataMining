import pandas as pd
data = pd.read_csv("Concrete_Data.csv")

test_data = data.iloc[500:630]   # iloc is 0-based so data starts at 0 not column title, 630 is exclusive

train_data = data.drop(data.index[500:630])

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)


