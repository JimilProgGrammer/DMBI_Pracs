import numpy as np
import pandas as pd
import random

def generate_data(rn):
    """
    Takes a roll number as input and generates random dataset
    according to the required specifications.

    :param rn: roll number of the student to ensure randomness
                and uniqueness of data 
    """
    n = 10 + (rn%3)
    m = 7 + (rn%3)
    print("generate_data(): STARTING WITH " + str(n) + " ALTERNATIVES AND " + str(m) + " ATTRIBUTES.")
    data = {
        "alternatives": np.arange(1,n+1)
    }
    for i in range(1,m+1):
        data["Attribute " + str(i)] = list(np.random.randint(15-(rn%16), 30+(rn%16), n))
    pd.DataFrame(data).to_csv("data.csv", index=False)
    print("Step 1: Data Generation Complete.")

def generate_pairwise_comparison(data):
    """
    Creates a square matrix and fills in the upper triangle
    with a random value between 2 to 9 and the lower triangle
    with 1/random_value for each corressponding element.

    :param data: dataframe representing the original dataset
    """
    shape = data.shape
    pairwise = np.zeros((shape[1],shape[1]))
    for i in range(shape[1]):
        for j in range(shape[1]):
            value = np.random.randint(2,9,1)
            if i == j:
                pairwise[i][j] = 1
            else:
                if pairwise[j][i] == 0:
                    pairwise[j][i] = value
                if pairwise[i][j] == 0:
                    pairwise[i][j] = round(float(1/value),2)
    return pairwise

def normalize_data(df, beneficial_attributes):
    """
    Utility function that takes in a dataframe and normalizes
    it's attributes
    :param df: The dataframe to normalize
    """
    columns = df.columns
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
    return df

if __name__=="__main__":
    # Generate random dataset
    # generate_data(72)

    # Generate Pairwise Comparison Matrix
    # data = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7))
    data = np.loadtxt(open("data_workmaterials.csv", "rb"), delimiter=",", skiprows=1, usecols=(1,2,3))
    pairwise = generate_pairwise_comparison(data)
    m = 3
    attributes = ["VC","CF","PI"]
    # for i in range(m):
    #     attributes.append("Attribute " + str(i))
    pairwise_df = {
        "attributes": attributes
    }
    pairwise_df["VC"] = pairwise[0]
    pairwise_df["CF"] = pairwise[1]
    pairwise_df["PI"] = pairwise[2]
    # for i in range(m):
    #     pairwise_df["Attribute " + str(i)] = pairwise[i]
    pd.DataFrame(pairwise_df).to_csv("pairwise_workmaterial.csv", index=False)
    print("Step 2: Pairwise Comparison Matrix Generated.")
    print("\n")

    # Calculate Geometric Mean
    a1 = pd.read_csv("pairwise_workmaterial.csv", index_col=0)
    gm = []
    for index,row in a1.iterrows():
        mean = np.prod(np.array(row.values))**(1/m)
        gm.append(mean)
    sum_gm = np.sum(np.array(gm))
    a2 = np.array(gm) / sum_gm
    print("A1 = ")
    print(a1)
    print("\n")
    print("A2 = ")
    print(a2)
    print("Sum(A2) = " + str(np.sum(a2)))
    print("Step 3: A1 & A2 matrices have been calculated.")
    print("\n")

    # Calculate A3 = A1 * A2
    a3 = np.dot(a1, a2.reshape(a2.shape[0],1))
    a3 = a3.reshape(a3.shape[0]).tolist()
    print("A3 = ")
    print(a3)

    # Calculate A4 = A3 / A2
    a4 = (np.array(a3) / a2).tolist()
    print("A4 = ")
    print(a4)
    print("Step 4: A3 & A4 matrices have been calculated.")
    print("\n")

    # Calculate Consistency Ratio
    lambda_max = np.mean(a4)
    print("Lamda Max: " + str(lambda_max))
    ci = (lambda_max-m)/(m-1)
    print("Consistency Index: " + str(ci))
    consistency = {1:0, 2:0, 3:0.58, 4:0.9, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    ri = consistency[m]
    print("Random Index: " + str(ri))
    cr = ci/ri
    print("Consistency Ratio: " + str(cr))
    if cr < 0.10:
        print("Matrix is reasonably consistent.")
        print("Step 5: Matrix consistency checked.")
    else:
        print("Matrix is not reasonable.")
        print("Step 5: Matrix consistency checked.")
        exit()
    
    # Derive overall priorities
    print("\n")
    print("Overall percentage of weightage: ")
    print(a2 * 100)

    # Derive local preferences
    normalized = normalize_data(pd.read_csv("data_workmaterials.csv", index_col=0), ["VC"])
    normalized.to_csv("normalized_workmaterials.csv", index=False)
    print("Step 6: Normalization done.")

    # Calculate AI and rank
    data = pd.read_csv("data_workmaterials.csv", index_col=0)
    ai = []
    for index,row in data.iterrows():
        ai.append(np.sum(np.array(row.tolist()) * a2))
    data["AIs"] = ai
    data["Rank"] = data["AIs"].rank(ascending=0)
    data.sort_values(by='Rank', inplace=True)
    print(data)
