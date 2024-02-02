import itertools
import random


def generate_slags(num_ifos, slag_min, slag_max, slag_off=0, slag_size=None, shuffle=True):
    """
     Generate a list of super lag (slag) combinations for a given number of interferometers (ifos),
     considering a specified range of shift values, slag distance range, offset, and size.

     Parameters:
     - num_ifos (int): The number of interferometers. ifo[0] is considered the reference and always has a shift of 0.
     - max_shift (int): The maximum absolute shift value for each ifo (except ifo[0]).
                        The shifts range from -max_shift to max_shift for each ifo.
     - slag_min (int): The minimum slag distance to be considered. Combinations with slag distance
                       less than slag_min are excluded from the result.
     - slag_max (int): The maximum slag distance to be considered. Combinations with slag distance
                       greater than slag_max are excluded from the result.
     - slag_off (int): The offset for slag combinations. The first 'slag_off' combinations are skipped
                       in the final list. Useful for pagination or skipping certain combinations.
     - slag_size (int): The number of slag combinations to be included in the final list.
                        If the number of available combinations after applying the offset is less than slag_size,
                        the resulting list may be shorter than slag_size.

     Returns:
     List[Tuple[int]]: A list of tuples, where each tuple represents a combination of shifts for the ifos.
                       The length of each tuple is equal to 'num_ifos', and the first element of each tuple
                       (representing ifo[0]) is always 0. The subsequent elements are the shifts for ifos[1] to ifos[n-1].

     Note:
     - The function ensures that there are no zero shifts for ifos apart from the [0, 0, ..., 0] combination.
     - The shifts are sorted primarily by the slag distance and secondarily by the order of the shifts themselves.
     - The function does not return the slag distances themselves, only the combinations of shifts.

     Example:
     num_ifos = 3
     max_shift = 2
     slag_min = 3
     slag_max = 3
     slag_off = 2
     slag_size = 2
     slags = generate_slags(num_ifos, max_shift, slag_min, slag_max, slag_off, slag_size)
     print(slags)  # Output would be: [(0, -1, 2), (0, -2, -1)]
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