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
    n = 93 + (rn%11)
    m = 70 + (rn%11)
    print("generate_data(): STARTING WITH " + str(n) + " ALTERNATIVES AND " + str(m) + " ATTRIBUTES.")
    data = {
        "alternatives": np.arange(1,n+1)
    }
    for i in range(1,m+1):
        data["Attribute " + str(i)] = list(np.random.randint(rn-(rn%11), rn+(rn%11), n))
    pd.DataFrame(data).to_csv("data.csv", index=False)
    print("generate_data(): DATA GENERATION COMPLETED; CHECK data.csv IN CWD.")

if __name__=="__main__":
    generate_data(72)