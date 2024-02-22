import pickle

with open('test_data.pkl', 'rb') as f:
    data = pickle.load(f)

import copy

fragment_clusters = data['fragment_clusters']

clusters = []

for fragment_cluster in fragment_clusters:
    clusters += fragment_cluster.clusters


from pycwb.modules.super_cluster.super_cluster import supercluster, defragment
from timeit import timeit

superclusters = supercluster(clusters, 'L', data['config'].gap, data['e2or'], data['config'].nIFO)

print(f"Total number of superclusters: {len(superclusters)}")
for i, c in enumerate(superclusters):
    n_pix = len(c.pixels)
    print(f'supercluster {i} has {n_pix} pixels and {"rejected" if c.cluster_status != 0 else "accepted"}')

# timeit, print one call in seconds
time = timeit("supercluster(clusters, 'L', data['config'].gap, data['e2or'], data['config'].nIFO)", globals=globals(), number=100)
print(f"Time: {time/100} seconds per call")

# profiler
# import cProfile
# cProfile.run("supercluster(clusters, 'L', data['config'].gap, data['e2or'], data['config'].nIFO)", sort='cumtime')

new_superclusters = defragment([sc for sc in superclusters if sc.cluster_status == 0],
                               data['config'].Tgap, data['config'].Fgap, data['config'].nIFO)

print(f"Total number of new superclusters: {len(new_superclusters)}")
for i, c in enumerate(new_superclusters):
    n_pix = len(c.pixels)
    print(f'supercluster {i} has {n_pix} pixels')

time = timeit("defragment([sc for sc in superclusters if sc.cluster_status == 0], data['config'].Tgap, data['config'].Fgap, data['config'].nIFO)", globals=globals(), number=100)
print(f"Time: {time/100} seconds per call")