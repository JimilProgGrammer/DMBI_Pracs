import numpy as np
import pandas as pd

def normalize_data(df):
    """
    Utility function that takes in a dataframe and normalizes
    50% of it's attributes that are beneficial and other 50%
    that are non-beneficial.
    :param df: The dataframe to normalize
    """
    columns = df.columns
    
    """
    For Beneficial attributes, find max and normalize
    value using value/max
    """
    for i in range(0,int(len(columns)/2)):
        values = np.array(df[columns[i]].tolist())
        values = values / np.max(values)
        df[columns[i]] = values
        
    """
    For Non-beneficial attributes, find min and normalize
    value using min/value
    """
    for i in range(int(len(columns)/2),len(columns)):
        values = np.array(df[columns[i]].tolist())
        values = np.min(values) / values
        df[columns[i]] = values
    print(df.tail())
    return df

if __name__ == "__main__":
    data = pd.read_csv("data.csv", index_col=0)
    normalized_data = normalize_data(data)
    print("-------------------")
    print(normalized_data.tail())
    normalized_data.to_csv("normalized_data.csv")
    print("normalize.py: Normalization completed!")