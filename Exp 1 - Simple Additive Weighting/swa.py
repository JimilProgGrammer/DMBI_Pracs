"""
Main script that reads in the normalized data and applies
SWA method for multiple attribute decision making.

"""
import pandas as pd
import numpy as np

# Read in the normalized dataframe
data = pd.read_csv("normalized_data.csv", index_col = 0)
print(data.tail())
# Number of alternatives
m = len(data.columns)
# Number of attributes
n = len(data)

# Generate equal weights for all attributes
weights = np.linspace(0,1,m)

# Calculate pi as sum of attribute weights * normalized attribute values
pi = []
for index,row in data.iterrows():
    pi.append(np.sum(np.array(row.tolist()) * weights))

# Set PIs to the values calculate above
data["PIs"] = pi

# Set Rank according to pi
data["Rank"] = data["PIs"].rank(ascending=0)

# Reset index to have alternatives added back to the frame
data = data.reset_index()

# Sort data in ascending order of rank
data.sort_values(by='Rank', inplace=True)

# Store the result
data.to_csv("result.csv", index=False)

print("swa.py: COMPLETE!")