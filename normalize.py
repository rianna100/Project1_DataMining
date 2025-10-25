import pandas as pd

data = pd.read_csv("test_data.csv")
min_values = {
    "Cement (component 1)(kg in a m^3 mixture)": 102,
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":0,
    "Fly Ash (component 3)(kg in a m^3 mixture)": 0,
    "Water  (component 4)(kg in a m^3 mixture)": 121.8,
    "Superplasticizer (component 5)(kg in a m^3 mixture)": 0, 
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": 801,
    "Fine Aggregate (component 7)(kg in a m^3 mixture)":594,
    "Age (day)": 1
}

max_values = {
    "Cement (component 1)(kg in a m^3 mixture)": 540,
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":305.3,
    "Fly Ash (component 3)(kg in a m^3 mixture)": 200.1,
    "Water  (component 4)(kg in a m^3 mixture)": 247,
    "Superplasticizer (component 5)(kg in a m^3 mixture)": 32.2, 
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": 1145,
    "Fine Aggregate (component 7)(kg in a m^3 mixture)":992.6,
    "Age (day)": 365
}


columns_to_normalize = data.columns[:-1]
for col in columns_to_normalize:
    data[col] = (data[col] - min_values[col]) / (max_values[col] - min_values[col])
    
data.to_csv("normalize_test_data.csv", index=False)

