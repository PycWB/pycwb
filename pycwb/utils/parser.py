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