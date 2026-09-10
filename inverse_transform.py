"""
inverse_transform.py — undo StandardScaler/MinMaxScaler scaling for a single
target column when the scaler was fit on multiple feature columns.

"""
import numpy as np


def inverse_target(scaled_col, scaler, n_features, target_idx=0):
    """
    Inverse-transforms a single scaled column back to its original scale.

    scaled_col :  1D array-like (or column vector) of scaled target values.

    scaler :      the already-fitted scaler object

    n_features :  total number of columns/features the scaler

    target_idx :  which column index your target corresponds to

    Returns: 1D numpy array of the target values back in real/original units.
    """
    scaled_col = np.asarray(scaled_col).flatten()

    # Build a dummy array with the same width the scaler expects,
    # placing our values in the target column and zeros elsewhere.
    dummy = np.zeros((len(scaled_col), n_features))
    dummy[:, target_idx] = scaled_col

    return scaler.inverse_transform(dummy)[:, target_idx]