"""
LMDB utilities for reading array data.
"""

import numpy as np


def retrieve_row_from_lmdb(lmdb_env, array_name: str, dtype, shape: tuple, row_index: int):
    """
    Retrieve a specific row from an array stored in LMDB.

    Args:
        lmdb_env: LMDB environment
        array_name: Name of the array
        dtype: NumPy dtype
        shape: Shape of each row
        row_index: Row index to retrieve

    Returns:
        NumPy array with the row data
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    array = np.frombuffer(row_bytes, dtype=dtype)

    if len(shape) > 0:
        array = array.reshape(shape)
    return array


def get_array_shape_from_lmdb(lmdb_env, array_name: str) -> tuple:
    """
    Get the shape of an array stored in LMDB.

    Args:
        lmdb_env: LMDB environment
        array_name: Name of the array

    Returns:
        Shape tuple
    """
    with lmdb_env.begin() as txn:
        shape_str = txn.get(f"{array_name}_shape".encode()).decode()
        shape = tuple(map(int, shape_str.split()))

    return shape
