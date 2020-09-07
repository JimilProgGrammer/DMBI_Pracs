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
    return n,m

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
            value = np.random.randint(2,5,1)
            if i == j:
                pairwise[i][j] = 1
            else:
                if pairwise[j][i] == 0:
                    pairwise[j][i] = value
                if pairwise[i][j] == 0:
                    pairwise[i][j] = round(float(1/value),2)
    return pairwise

def normalize_data(df, root_square_list):
    """
    Utility function that takes in a dataframe and normalizes
    it's attributes
    :param df: The dataframe to normalize
    :param root_square_list:
    """
    columns = df.columns
    for i in range(len(columns)):
        df[columns[i]] = np.array(df[columns[i]].tolist()) / root_square_list[i]
    return df

def weighted_normalization(df, weights):
    """
    Utility function that takes in a normalized dataframe and
    recalculates the weighted normalized matrix
    :param df:
    :param weights:
    """
    columns = df.columns
    for i in range(len(columns)):
        df[columns[i]] = np.array(df[columns[i]].tolist()) * weights[i]
    return df

def calculate_ideal_best(df, beneficial):
    """
    Utility function that takes in a dataframe and returns the
    ideal best candidate value for each feature
    :param df:
    :param beneficial:
    """
    v_plus = []
    for col in df.columns:
        if col in beneficial:
            v_plus.append(np.max(np.array(df[col].tolist())))
        else:
            v_plus.append(np.min(np.array(df[col].tolist())))
    return v_plus

def calculate_ideal_worst(df, beneficial):
    """
    Utility function that takes in a dataframe and returns the
    ideal worst candidate value for each feature
    :param df:
    :param beneficial:
    """
    v_minus = []
    for col in df.columns:
        if col in beneficial:
            v_minus.append(np.min(np.array(df[col].tolist())))
        else:
            v_minus.append(np.max(np.array(df[col].tolist())))
    return v_minus

def calculate_distance_from_ideal_best(df, v_plus):
    """
    Utility function that returns the Euclidean distance of each
    alternative from the ideal best.
    :param df:
    :param v_plus:
    """
    si_plus = []
    columns = df.columns
    for index,row in df.iterrows():
        running_sum = 0
        for i in range(len(columns)):
            running_sum = running_sum + ((row[columns[i]]-v_plus[i])**2)
        si_plus.append(running_sum ** 0.5)
    return si_plus

def calculate_distance_from_ideal_worst(df, v_minus):
    """
    Utility function that returns the Euclidean distance of each
    alternative from the ideal worst.
    :param df:
    :param v_minus:
    """
    si_minus = []
    columns = df.columns
    for index,row in df.iterrows():
        running_sum = 0
        for i in range(len(columns)):
            running_sum = running_sum + ((row[columns[i]]-v_minus[i])**2)
        si_minus.append(running_sum ** 0.5)
    return si_minus

if __name__=="__main__":
    # Generate random dataset
    n,m = generate_data(72)

    # Generate Pairwise Comparison Matrix
    data = np.loadtxt(open("data.csv", "rb"), delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7))
    pairwise = generate_pairwise_comparison(data)
    attributes = []
    for i in range(m):
        attributes.append("Attribute " + str(i))
    pairwise_df = {
        "attributes": attributes
    }
    for i in range(m):
        pairwise_df["Attribute " + str(i)] = pairwise[i]
    pd.DataFrame(pairwise_df).to_csv("pairwise_data.csv", index=False)
    print("Step 2: Pairwise Comparison Matrix Generated.")
    print("\n")

    # Calculate Geometric Mean
    a1 = pd.read_csv("pairwise_data.csv", index_col=0)
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
    
    # TOPSIS Step 1
    data = pd.read_csv("data.csv", index_col=0)
    root_square = []
    for col in data.columns:
        values = np.array(data[col].tolist())
        root_square.append((np.sum(values**2))**(1/2))
    print("TOPSIS Step 1 done: Root square values calculated.")
    
    # TOPSIS Step 2: Normalize matrix
    data = normalize_data(data, root_square)
    print("TOPSIS Step 2 done: Data is normalized.")
    
    # TOPSIS Step 3: Calculate weighted-normalized matrix
    weighted_normalized = weighted_normalization(data, a2)
    print("TOPSIS Step 3 done: Weighted Normalization performed.")
    print(weighted_normalized)
    
    beneficial = []
    for i in range(int(m/2)):
        beneficial.append(attributes[i])
    
    # TOPSIS Step 4: Calculate ideal best and ideal worst
    v_plus = calculate_ideal_best(weighted_normalized, beneficial)
    v_minus = calculate_ideal_worst(weighted_normalized, beneficial)
    print("TOPSIS Step 4 done.")
    print("Ideal Best => " + str(v_plus))
    print("Ideal Worst => " + str(v_minus))
    
    # TOPSIS Step 5: Calculate distance of alternative from ideals
    si_plus = calculate_distance_from_ideal_best(weighted_normalized, v_plus)
    si_minus = calculate_distance_from_ideal_worst(weighted_normalized, v_minus)
    print("TOPSIS Step 5 done.")
    print("d(alternative,ideal_best) => " + str(si_plus))
    print("d(alternative,ideal_worst) => " + str(si_minus))
    
    # TOPSIS Step 6: Calculate performance scores
    pi = []
    for i in range(len(si_plus)):
        pi.append(si_minus[i]/(si_plus[i] + si_minus[i]))
    print("TOPSIS Step 6 done. PIs => " + str(pi))

    # TOPSIS Step 7: Assign ranks
    data = pd.read_csv("data.csv", index_col=0)
    data["PIs"] = pi
    data["Rank"] = data["PIs"].rank(ascending=0)
    data.sort_values(by='Rank', inplace=True)
    print("TOPSIS Result calculated.")
    print(data)
