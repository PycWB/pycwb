def parse_id_string(id_string: str) -> list[int]:
    result = []
    # Split the string by commas to handle multiple segments
    parts = id_string.split(',')
    for part in parts:
        if '-' in part:
            # It's a range, split it into start and end
            start, end = map(int, part.split('-'))
            # Extend result with the full range of numbers
            result.extend(range(start, end + 1))
        else:
            # It's a single number, convert to int and add to the list
            result.append(int(part))
    return result


def parse_vars(var_string):
    variables = {}
    for item in var_string.split():
        if '=' in item:
            key, value = item.split('=', 1)
            variables[key] = value
    return variables


def parse_lag_string(lag_string: str) -> list[list[float]]:
    """Parse a lag specification string into a lag array.

    Each lag vector is semicolon-separated, and within each vector the
    per-IFO shifts are comma-separated.  For example ``"0,0;1,0;0,1"``
    produces ``[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]``.
    """
    lag_array = []
    for entry in lag_string.split(';'):
        entry = entry.strip()
        if not entry:
            continue
        lag_array.append([float(v) for v in entry.split(',')])
    if not lag_array:
        raise ValueError("Empty lag specification")
    width = len(lag_array[0])
    for i, row in enumerate(lag_array):
        if len(row) != width:
            raise ValueError(
                f"Lag vector {i} has {len(row)} values, expected {width}"
            )
    return lag_array