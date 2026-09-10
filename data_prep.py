import numpy as np


def create_sequences(data, target_col_index, seq_length):
    """
    Slides a window over your data to build (X, y) pairs for an LSTM.

    data :   your scaled/cleaned numpy array, shape (num_rows, num_features).
            Put train or test array here

    target_col_index :  which column[index] of data you're trying to predict.

    seq_length :  how many past time-steps the model looks at to make 1 prediction

    Returns: X and y
    
    """

    X = []
    y = []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_col_index])

    return np.array(X), np.array(y)