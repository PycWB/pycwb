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