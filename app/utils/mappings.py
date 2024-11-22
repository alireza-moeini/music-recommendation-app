def create_mapping(values):
    unique = sorted(set(values))
    idx_map = {v: i for i, v in enumerate(unique)}
    rev_map = {i: v for v, i in idx_map.items()}
    return idx_map, rev_map
