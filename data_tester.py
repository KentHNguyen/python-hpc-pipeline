import time
from data_loader import *


if __name__ == "__main__":
    '''
    You can write your filled out functions in here to test them out.
    '''

    
    from data_loader import *
import time


if __name__ == "__main__":

    # 1(a) - Performance comparison: plain Python vs NumPy

    data_list = load_dataset("GasProperties.csv")
    data_np = load_dataset_np("GasProperties.csv")

    start = time.time()
    rows_plain = normalize_array(data_list, "GasProperties_norm.csv")
    time_plain = time.time() - start

    start = time.time()
    rows_np = normalize_array_np(data_np, "GasProperties_norm_np.csv")
    time_np = time.time() - start

    print(f"Plain Python normalize: {time_plain:.4f} seconds ({rows_plain} rows)")
    print(f"NumPy normalize:        {time_np:.4f} seconds ({rows_np} rows)")
    print(f"NumPy was {time_plain / time_np:.1f}x faster")
    print()

    # 1(b) - Load, split into X/Y, split into train/test

    df = load_dataset_pd("GasProperties_norm.csv")
    X_data, Y_data = split_xy(df)
    X_train, Y_train, X_test, Y_test = split_training_test(X_data, Y_data)

    print("X_train:", X_train.shape, "| X_test:", X_test.shape)
    print("Y_train:", Y_train.shape, "| Y_test:", Y_test.shape)