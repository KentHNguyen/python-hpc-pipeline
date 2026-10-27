from vector_product import *
from data_loader import load_dataset_np
import numpy as np
import time

if __name__ == "__main__":
    data_np = load_dataset_np("GasProperties.csv")
    X_data = data_np[:, :4]
    Y_data = data_np[:, 4]

    X_columns_np = X_data.T          # each row = one feature column
    X_columns_list = X_columns_np.tolist()
    Y_data_list = Y_data.tolist()

    # 2(a) - plain Python
    start = time.time()
    idx_py = find_largest_dot_product_py(X_columns_list, Y_data_list)
    time_py = time.time() - start
    print(f"Python loop: index {idx_py}, {time_py:.4f} seconds")

    # 2(b) - NumPy
    start = time.time()
    idx_np = find_largest_dot_product_np(X_columns_np, Y_data)
    time_np = time.time() - start
    print(f"NumPy: index {idx_np}, {time_np:.4f} seconds")
    print(f"NumPy was {time_py / time_np:.1f}x faster")

    # 2(c) - 32-bit vs 64-bit matmul (subset to avoid memory overflow)
    X_data_small = X_data[:20000] 

    X_32 = X_data_small.astype(np.float32)
    X_64 = X_data_small.astype(np.float64)

    start = time.time()
    mat_mul_np(X_32, X_32.T)
    time_32 = time.time() - start

    start = time.time()
    mat_mul_np(X_64, X_64.T)
    time_64 = time.time() - start

    print(f"32-bit matmul: {time_32:.4f} seconds")
    print(f"64-bit matmul: {time_64:.4f} seconds")