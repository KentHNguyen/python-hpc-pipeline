'''
This file is where you migrate all the functions that you made in data_process1 and data_process2. 
'''
import numpy as np
import pandas as pd
import csv
import time # Import the time module
from typeguard import typechecked

#data_process1.py methods
@typechecked
def load_dataset(filename: str) -> list[list[float]]:
    """
    Loads a dataset from a CSV file into a list of lists of floats.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list[list[float]]: A list of lists, where each inner list represents a row
                           of the dataset with float values.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: If you are stuck on this, I recommend looking through the python csv library.
    """
    GasProperties = []
    with open(filename , newline = '') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            GasProperties.append([float(value) for value in row])
    return GasProperties

    raise NotImplementedError()


@typechecked
def load_dataset_np(filename: str) -> np.ndarray:
    """
    Loads a dataset from a CSV file into a NumPy array.
    The first row (header) is skipped.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        np.ndarray: A NumPy array representing the dataset.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Numpy has a very useful csv file reader called genfromtxt.
    """

    GasProperties = np.genfromtxt(filename, delimiter = ',', skip_header = 1)

    return GasProperties

    raise NotImplementedError()

@typechecked
def normalize_array(arr: list[list[float]], out_file: str | None = None) -> int:
    """
    Normalizes the input array (list of lists) using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.

    Args:
        arr (list[list[float]]): The input dataset as a list of lists of floats.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    
    Note: Probably the most complicated function to write because you cant use numpy.
    I would spend some time on this to make sure that all the equations for metrics are correct.
    """
    num_rows = len(arr)
    num_cols = len(arr[0])

    means = []
    mins = []
    maxs = []
    stds = []

    for j in range(num_cols - 1):
        cols_values = [arr[i][j] for i in range(num_rows)]
        cols_mean = sum(cols_values) / num_rows
        cols_max = max(cols_values)
        cols_min = min(cols_values)
        cols_std = (sum((x - cols_mean) ** 2 for x in cols_values) / (num_rows - 1)) ** .5

        means.append(cols_mean)
        mins.append(cols_min)
        maxs.append(cols_max)
        stds.append(cols_std)

    result = []
    for i in range(num_rows):
        is_outlier = False
        for j in range(num_cols - 1):
            if abs(arr[i][j] - means[j]) > 2 * stds[j]:
                is_outlier = True
                break

        if is_outlier:
            continue  

        new_row = []
        for j in range(num_cols - 1):
            normalized_value = (arr[i][j] - means[j]) / (maxs[j] - mins[j])
            new_row.append(normalized_value)
        new_row.append(arr[i][num_cols - 1])  
        result.append(new_row)

    if out_file is not None:
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(result)

    return len(result)

    raise NotImplementedError()

@typechecked
def normalize_array_np(arr: np.ndarray, out_file: str | None = None) -> int:
    """
    Normalizes the input NumPy array using min-max normalization
    and filters out outliers based on standard deviation.
    The last column is assumed to be the target variable and is not normalized.
    Optionally writes the normalized data to a new CSV file.

    Args:
        arr (np.ndarray): The input dataset as a NumPy array.
        out_file (str, optional): The path to the output CSV file. Defaults to None.

    Returns:
        int: The number of rows processed after normalization and filtering.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: The same function as normalize_array but using numpy to calculate the metrics.
    This function should be almost copy and paste with numpy functions.
    """

    num_cols = arr.shape[1] 

    features = arr[:,:num_cols-1]

    means = np.mean(features, axis = 0)
    mins = np.min(features, axis = 0)
    maxs = np.max(features, axis = 0)
    stds = np.std(features, axis = 0)

    outlier_mask = np.abs(features - means) > 2 * stds
    row_is_outlier = np.any(outlier_mask, axis=1)
    clean_arr = arr[~row_is_outlier]

    clean_features = clean_arr[:, :num_cols - 1]
    idx_column = clean_arr[:, num_cols - 1:num_cols] 

    normalized_features = (clean_features - means) / (maxs - mins)

    result = np.hstack([normalized_features, idx_column])

    if out_file is not None:
        np.savetxt(out_file, result, delimiter=',')

    return result.shape[0]

    raise NotImplementedError()



#data_process2.py methods
@typechecked
def load_dataset_pd(filename: str) -> pd.DataFrame:
    """
    Loads normalized data from a CSV file into a Pandas DataFrame.

    Args:
        filename (str): The path to the CSV file containing normalized data.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the loaded data.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.

    Note: This is a 1-line solution, if you spend more than 10 mins on this, you may be overthinking.
    """
    return pd.read_csv(filename)

    raise NotImplementedError()


@typechecked
def split_xy(df: pd.DataFrame, y_axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits a Pandas DataFrame into feature (X) and target (Y) NumPy arrays.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        y_axis (int, optional): The column index of the target variable (Y).
                                Defaults to -1 (last column).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                       X (features) and Y (target).

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    Note: A 2-line solution but can be tricky if you are not familiar with pandas.
    """
    num_cols = df.shape[1]
    X_data = df.drop(df.columns[y_axis], axis=1).values
    Y_data = df.iloc[:, y_axis].values
    return (X_data, Y_data)

    raise NotImplementedError()

@typechecked
def split_training_test(
        X_data: np.ndarray, 
        Y_data: np.ndarray, 
        split: float = 0.8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the input feature (X) and target (Y) NumPy arrays into training and testing sets.

    Args:
        X_data (np.ndarray): The input feature array.
        Y_data (np.ndarray): The input target array.
        split (float, optional): The proportion of data to be used for training.
                                 Defaults to 0.8 (80% training, 20% testing).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing four NumPy arrays:
                                                                X_train, Y_train, X_test, Y_test.

    WARNING: Do not modify the type hints (parameter types or return type) of this function,
             as it will cause the autograder to fail, resulting in minimal credit.
    Note: If you are stuck, look into list slicing. 
    """
    num_rows = X_data.shape[0]
    cutoff = int(num_rows * split)

    X_train = X_data[:cutoff]
    Y_train = Y_data[:cutoff]
    X_test = X_data[cutoff:]
    Y_test = Y_data[cutoff:]

    return (X_train, Y_train, X_test, Y_test)

    raise NotImplementedError()