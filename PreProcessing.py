import pandas as pd
import numpy as np
import sys

#X is unprocessed data
#Y is processed data

df = pd.read_excel('Market Towns.xlsx')
headers = df.columns.tolist()
X = df.to_numpy()
Y = df.to_numpy()

##Following the steps on chapter 2.4 even though no categorical features in market towns table

def identify_categorical(Z):
    categorical_cols = []
   
    for i in range(1, len(Z[0])):
        try:
            float(Z[0][i])
        except:
            categorical_cols.append(i)
    return categorical_cols

def adding_categories(Z):
    categorical_cols = identify_categorical(Z)
    if not categorical_cols:
        return Z

    unique_vals = {}
    for col in categorical_cols:
        unique_vals[col] = set()
        for row in Z:
            unique_vals[col].add(row[col])

    new_Z = []
    for row in Z:
        new_row = []
        for i in range(len(row)):
            if i not in categorical_cols:
                new_row.append(row[i])
            else:
                for val in unique_vals[i]:
                    if row[i] == val:
                        new_row.append(1)
                    else:
                        new_row.append(0)
        new_Z.append(new_row)
    
    return np.array(new_Z)

def categorical_rescaling(Z, original_categorical_cols):
    for col in original_categorical_cols:
        dummy_count = 0
        unique_vals = set()
        for row in Z:
            unique_vals.add(row[col])
        dummy_count = len(unique_vals)

    start_col = col
    end_col = col + dummy_count
    for i in range(len(Z)):
        for j in range(start_col, end_col):
            Z[i][j] = Z[i][j] / np.sqrt(dummy_count)

    return Z


# Y_{iv} = (X_{iv} - av) / bv

###No error handling with forcing float, maybe look into later
def calculate_av(Z, feature_num):
    if feature_num == 0:
        return 0

    total = 0
    for i in Z:
        total += float(i[feature_num])
    return total / len(Z)
    

def calculate_bv(Z, feature_num):
    if feature_num == 0:
        return 1

    max = -sys.maxsize - 1
    min = sys.maxsize
    for i in Z:
        val = float(i[feature_num])
        if val > max:
            max = val
        if val < min:
            min = val
    return max - min

def shifting_origins(Z, feature_num, av, bv):
    if feature_num == 0:
        return Z

    for i in range(len(Z)):
        Z[i][feature_num] = round((float(Z[i][feature_num]) - av) / bv, 3)
    return Z

Y = adding_categories(X)
categorical_cols = identify_categorical(X) #Store for later :)
for feature in range(1, Y.shape[1]):
   Y = shifting_origins(Y, feature, calculate_av(Y, feature), calculate_bv(Y, feature))

if categorical_cols: #I know, genius
    Y = categorical_rescaling(Y, categorical_cols)

print(Y)

df_processed = pd.DataFrame(Y, columns=headers)
df_processed.to_excel('ProcessedData.xlsx', index=False)

##Tomorrow do Iris data to see if cat cols work.