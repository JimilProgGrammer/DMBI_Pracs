import numpy as np
import pandas as pd

def run_madm_approach(df, beneficial_attributes, non_beneficial_attributes, approach="saw"):
    """
    Utility method that normalizes the data passed as a DataFrame
    uses the beneficial and non-beneficial list to normalize

    :param df: data represented as a dataframe
    :param beneficial_attributes: list of columns that are beneficial
    :param non_beneficial_attributes: list of columns that are non-beneficial
    :param approach: the method to use for ranking alternatives; Allowed values -> ["saw","wpm"]
    """
    approach = str(approach).lower()
    
    if df is None:
        print("[ERROR]: Null dataframe passed. Please verify.")
    
    print(df.head())
    columns = df.columns
    if(len(columns) != len(beneficial_attributes)+len(non_beneficial_attributes)):
        print("[ERROR]: Data passed has more attributes than the beneficial and non-beneficial attributes specified.")

    for i in range(len(columns)):
        if columns[i] in beneficial_attributes:
            """
            For Beneficial attributes, find max and normalize
            value using value/max
            """
            values = np.array(df[columns[i]].tolist())
            values = values / np.max(values)
            df[columns[i]] = values
        else:
            """
            For Non-beneficial attributes, find min and normalize
            value using min/value
            """
            values = np.array(df[columns[i]].tolist())
            values = np.min(values) / values
            df[columns[i]] = values
    print("[INFO]: Normalization done.")

    """
    Weights are calculated by assigning relative importance to the
    attributes and then normalizing those attribute weights to 1.
    """
    weights = [0.7142, 0.1429, 0.1429]

    """
    Calculate performance as the weighted sum or weighted product
    depending upon the approach.
    """
    pi = []
    if approach == "saw":
        for index,row in data.iterrows():
            pi.append(np.sum(np.array(row.tolist()) * weights))
    elif approach == "wpm":
        for index,row in data.iterrows():
            pi.append(np.prod(np.array(row.tolist()) ** weights))
    print("[INFO]: Calculated performance")
    df["Performance"] = pi
    
    """
    Set rank according to the performance value, sort and
    store to disk.
    """
    df["Rank"] = df["Performance"].rank(ascending=0)
    df = df.reset_index()
    df.sort_values(by='Rank', inplace=True)
    print("[INFO]: Assigned and sorted rank.")
    df.to_csv("result_" + approach + ".csv", index=False)
    df = None

if __name__ == "__main__":
    # Read in the normalized dataframe
    data = pd.read_csv("data.csv", index_col = 0)
    print(data.head())
    print("----------------------------------------------------")
    print("[INFO]: Running SAW approach for MADM.")
    run_madm_approach(data, ["VC"], ["CF","PI"], "saw")
    print("[INFO]: SAW output generated.")
    print("----------------------------------------------------")
    data = None
    data = pd.read_csv("data.csv", index_col = 0)
    print("[INFO]: Running WPM approach for MADM.")
    run_madm_approach(data, ["VC"], ["CF","PI"], "wpm")
    print("[INFO]: WPM output generated.")
