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

aggregated_clusters = supercluster(cluster, data['config'].gap, 2)

# timeit, print one call in seconds
from timeit import timeit
time = timeit('supercluster(cluster, data["config"].gap, 2)', globals=globals(), number=100)
print(time/100)

# profiler
import cProfile
cProfile.run('supercluster(cluster, data["config"].gap, 2)', sort='cumtime')
