import numpy as np

def label_encode_ordinal(values: list, order: list) -> list:
    order_arr = np.array(order)
    values_arr = np.array(values)
    
    # 1. Sort alphabetically and search
    sorter = np.argsort(order_arr)
    encoded_indices = np.searchsorted(order_arr, values_arr, sorter=sorter)
    
    # 2. Check for out-of-bounds indices
    valid_mask = encoded_indices < len(order_arr)
    
    # 3. Map safely to original positions
    actual_indices = np.zeros(len(values_arr), dtype=int)
    actual_indices[valid_mask] = sorter[encoded_indices[valid_mask]]
    
    # 4. Check if the string actually matches (filters out unknown words)
    valid_mask &= (order_arr[actual_indices] == values_arr)
    
    # 5. Use -1 for anything that is invalid/unknown
    result = np.where(valid_mask, actual_indices, -1)
    
    return result.tolist()
