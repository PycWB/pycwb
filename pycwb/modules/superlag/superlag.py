import itertools
import random


def generate_slags(num_ifos, slag_min, slag_max, slag_off=0, slag_size=None, shuffle=True):
    """Generate superlag shift tuples for a detector network.

    Parameters
    ----------
    num_ifos : int
        Detector count; the reference detector at index zero has no shift.
    max_shift : int
        Maximum absolute shift for the other detectors.
    slag_min : int
        Minimum accepted superlag distance.
    slag_max : int
        Maximum accepted superlag distance.
    slag_off : int
        Number of combinations to skip before selecting output.
    slag_size : int
        Maximum number of combinations to return after the offset.

    Returns
    -------
    list of tuple of int
        Shift tuples of length ``num_ifos``, with first element zero.
    """
    
    # Generate all possible shifts for ifos except ifo[0]
    shifts = list(itertools.product(range(-slag_max, slag_max + 1), repeat=num_ifos - 1))

    # Remove shifts with any zero except all zeros
    shifts = [shift for shift in shifts if not any(s == 0 for s in shift)]

    # Remove shifts contains same values in one shift
    shifts = [shift for shift in shifts if len(set(shift)) == len(shift)]

    # Add shifts with all zeros
    shifts.append((0,) * (num_ifos - 1))

    # Calculate slag distance (excluding ifo[0]) and sort combinations by distance
    slag_with_distance = [(sum(abs(shift) for shift in combination), (0,) + combination) for combination in shifts]
    slag_with_distance.sort(key=lambda x: (x[0], x[1]))

    # Filter by slag distance range
    filtered_slags = [slag for slag in slag_with_distance if slag_min <= slag[0] <= slag_max]

    # Apply slag offset
    offset_slags = filtered_slags[slag_off:]

    # Select slag size
    if slag_size:
        selected_slags = offset_slags[:slag_size]
    else:
        selected_slags = offset_slags

    # randomize the list
    if shuffle:
        random.seed(0)
        random.shuffle(selected_slags)

    # Return the selected slags without the distance value
    return [slag[1] for slag in selected_slags]