import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    data = pd.read_csv(os.path.join(os.getcwd(), 'data', 'breast-cancer.data'), header=None)
   
    # Read in the dataset from file
    data = pd.read_csv('./data/breast-cancer.data', header=None)

    # Set the column names
    data.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

    # Use pandas' get_dummies function to apply one-hot encoding to categorical variables
    data_encoded = pd.get_dummies(data, columns=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad'])
    data_encoded = data_encoded.iloc[:, 1:].join(data_encoded.iloc[:, 0])
    data_encoded['irradiat'] = data_encoded['irradiat'].map({'yes': 1, 'no': 0})


    # Save the encoded data to file
    data_encoded.to_csv('./data/breast-cancer-processed.csv', index=False)

