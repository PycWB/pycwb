import pickle

with open('test_data.pkl', 'rb') as f:
    data = pickle.load(f)

import copy

fragment_clusters = data['fragment_clusters']

cluster = copy.deepcopy(fragment_clusters[0])
if len(fragment_clusters) > 1:
    for fragment_cluster in fragment_clusters[1:]:
        cluster.clusters += fragment_cluster.clusters


from pycwb.modules.super_cluster.super_cluster import supercluster

superclusters = supercluster(cluster, 'L', data['config'].gap, data['e2or'], data['config'].nIFO)

print(f"Total number of superclusters: {len(superclusters)}")
for i, c in enumerate(superclusters):
    n_pix = len(c.pixels)
    print(f'supercluster {i} has {n_pix} pixels and {"rejected" if c.cluster_status != 0 else "accepted"}')

# timeit, print one call in seconds
from timeit import timeit
time = timeit("supercluster(cluster, 'L', data['config'].gap, data['e2or'], data['config'].nIFO)", globals=globals(), number=100)
print(f"Time: {time/100} seconds per call")

# profiler
import cProfile
cProfile.run("supercluster(cluster, 'L', data['config'].gap, data['e2or'], data['config'].nIFO)", sort='cumtime')
