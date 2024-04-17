import numpy as np
from numba import jit, int32, float64

# Using Numba's JIT decorator to compile the function
@jit(nopython=True)
def are_pixels_close_numba(pixel1, pixel2, dt, df):
    time_distance = abs(pixel1[1] - pixel2[1])
    frequency_distance = abs(pixel1[0] - pixel2[0])
    time_edge_distance = time_distance - (pixel1[3] / 2) - (pixel2[3] / 2)
    frequency_edge_distance = frequency_distance - (pixel1[2] / 2) - (pixel2[2] / 2)
    return (time_edge_distance <= dt) and (frequency_edge_distance <= df)

@jit(nopython=True)
def find_numba(parent, i):
    if parent[i] == i:
        return i
    return find_numba(parent, parent[i])

@jit(nopython=True)
def union_numba(parent, rank, x, y):
    xroot = find_numba(parent, x)
    yroot = find_numba(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

@jit(nopython=True)
def cluster_pixels_numba(pixels, dt, df):
    n = pixels.shape[0]
    parent = np.arange(n)
    rank = np.zeros(n, dtype=np.int32)

    for i in range(n):
        for j in range(i + 1, n):
            if are_pixels_close_numba(pixels[i], pixels[j], dt, df):
                union_numba(parent, rank, i, j)

    for i in range(n):
        parent[i] = find_numba(parent, i)

    return parent  # Returning parent array for simplicity

