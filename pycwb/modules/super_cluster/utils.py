import numpy as np
from numba.typed import List
from numba import njit, types


@njit(cache=True)
def find(parent, x):
    """Path compression find."""
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


@njit(cache=True)
def union(parent, rank, x, y):
    """Union by rank."""
    rootX = find(parent, x)
    rootY = find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1


@njit(cache=True)
def aggregate_clusters(links):
    """Aggregate clusters based on provided links."""
    n = np.max(links) + 1  # Assuming links are 0-indexed.
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.int32)

    # Perform unions
    for x, y in links:
        union(parent, rank, x, y)

    # Find all unique representatives
    for i in range(n):
        parent[i] = find(parent, i)

    # Create an empty list for each unique parent
    clusters = List()
    for i in range(n):
        clusters.append(List.empty_list(types.int32))

    # Aggregate clusters
    for i in range(n):
        root = find(parent, i)
        clusters[root].append(i)

    # Filter out single-element clusters
    filtered_clusters = List()
    for cluster in clusters:
        if len(cluster) > 1:
            filtered_clusters.append(cluster)

    return filtered_clusters



@njit
def remove_duplicates_sorted(arr):
    # Assuming arr is a sorted 2D numpy array of shape (n, 2)
    n = len(arr)
    if n == 0:
        return arr  # Early return for empty array

    # Preallocate an array of the same size to store unique elements
    unique = np.empty_like(arr)
    unique[0] = arr[0]
    count = 1  # Initialize count with 1 as the first element is always unique

    for i in range(1, n):
        # Since arr is sorted, just check if the current element is different from the last unique element added
        if not np.array_equal(arr[i], unique[count-1]):
            unique[count] = arr[i]
            count += 1

    # Trim the unique array to the correct number of unique elements found
    return unique[:count]
