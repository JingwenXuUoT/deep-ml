def label_encode_ordinal(values: list, order: list) -> list:
    # Build a lookup map of category -> rank index
    rank_map = {category: index for index, category in enumerate(order)}
    
    # .get(v, -1) returns -1 if the value isn't found in the order list
    return [rank_map.get(v, -1) for v in values]

