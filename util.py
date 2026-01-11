def clip (val, bounds):
    return min(max(val, bounds[0]), bounds[1])
