import pandas as pd
import numpy as np
import sys

# Y_{iv} = (X_{iv} - av) / bv

def is_categorical(column):
    return column.dtype == 'object' or column.dtype.name == 'category' #NOTE: 'category' is for categorical & 'object' is for mixed data

def process_categorical_column(column):
    dummy_cols = pd.get_dummies(column, prefix=column.name) #Saved by panda again
    
    for col in dummy_cols.columns: #Mean centering
        p = dummy_cols[col].mean()
        dummy_cols[col] = (dummy_cols[col] - p) / 1
    
    return dummy_cols

def process_numerical_column(column):
    av = column.mean()
    bv = column.max() - column.min() #No using standard deviation said the book

    return (column - av) / bv

df = pd.read_excel('Market Towns.xlsx')
df = df.iloc[:, 1:] #For market towns only, or remove first col
processed_columns = []

for column in df.columns:
    if is_categorical(df[column]):
        dummy_cols = process_categorical_column(df[column])
        processed_columns.append(dummy_cols) #No need to convert because already a df from .get_dummies
    else:
        processed_col = process_numerical_column(df[column])
        processed_columns.append(pd.DataFrame(processed_col, columns=[column])) #.concat needs all inputs as df's

df_processed = pd.concat(processed_columns, axis=1)

df_processed.to_csv('ProcessedData.csv', index=False)