# Python Data Pipeline Performance

A data processing pipeline built three ways (pure Python, NumPy, and Pandas) to compare how much speed you get from vectorization. It runs on a 420,000-row gas properties dataset.

## What it does

**Data pipeline (`data_loader.py`, `data_tester.py`)**
- Loads the CSV with Python's `csv` module, NumPy's `genfromtxt`, and Pandas
- Filters outliers (any row more than 2 standard deviations from a column's mean) and normalizes each feature column
- Splits the data into features and target, then into 80% training and 20% testing sets

**Vector operations (`vector_product.py`, `vector_tester.py`)**
- Dot products and "find the row with the largest dot product" in pure Python and NumPy
- Matrix multiplication with NumPy and PyTorch tensors
- Compares 32-bit and 64-bit floating point matrix multiplication

## Results

| Operation | Pure Python | NumPy | Speedup |
| --- | --- | --- | --- |
| Normalize + outlier filter (420K rows) | 1.79 s | 0.13 s | ~14x |
| Largest dot product (4 x 420K) | 0.083 s | 0.009 s | ~9x |

| Matrix multiply (20K x 4 by 4 x 20K) | Time |
| --- | --- |
| float32 | 1.64 s |
| float64 | 2.80 s |

32-bit floats were about 1.7x faster than 64-bit, since they use half the memory and move through the CPU faster.

Timings are from a single Linux machine and will vary by hardware. Normalization times leave out writing the output CSV, which takes about the same time either way.

## Running it

```
pip install numpy pandas torch typeguard
python data_tester.py
python vector_tester.py
```

Both scripts expect `GasProperties.csv` in the same folder. It isn't included in the repo because of its size (about 19 MB).
