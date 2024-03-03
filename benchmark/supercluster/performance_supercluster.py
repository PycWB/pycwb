import pickle

from pycwb.modules.super_cluster.sub_net_cut import sub_net_cut

with open('test_data.pkl', 'rb') as f:
    data = pickle.load(f)



fragment_clusters = data['fragment_clusters']
ml = data['ml']
FP = data['FP']
FX = data['FX']
acor = data['acor']
n_ifo = data['n_ifo']
n_sky = data['n_sky']
n_loudest = data['n_loudest']
lag = 0
subnet = data['subnet']
subcut = data['subcut']
subnorm = data['subnorm']
subrho = data['subrho']
xtalk_coeff = data['xtalk_coeff']
xtalk_lookup_table = data['xtalk_lookup_table']
layers = data['layers']
nRes = data['nRes']
Tgap = data['Tgap']
Fgap = data['Fgap']
gap = data['gap']
e2or = data['e2or']

clusters = []

# for fragment_cluster in fragment_clusters:
#     clusters += fragment_cluster.clusters

clusters = fragment_clusters.clusters

from pycwb.modules.super_cluster.super_cluster import supercluster, defragment
from timeit import timeit
from time import perf_counter
# import cProfile
#
# cProfile.run("supercluster(clusters, 'L', data['gap'], data['e2or'], data['n_ifo'])", sort='cumtime')
start_time_all = perf_counter()
start_time = perf_counter()
superclusters = supercluster(clusters, 'L', gap, e2or, n_ifo)
print(f"Time taken for full supercluster: {perf_counter() - start_time}")


print(f"Total number of superclusters: {len(superclusters)}")
for i, c in enumerate(superclusters):
    n_pix = len(c.pixels)
    print(f'supercluster {i} has {n_pix} pixels and {"rejected" if c.cluster_status != 0 else "accepted"}')

start_time = perf_counter()
new_superclusters = defragment([sc for sc in superclusters if sc.cluster_status == 0],
                               Tgap, Fgap, n_ifo)
print(f"Time taken for defragment: {perf_counter() - start_time}")

print(f"Total number of defragment superclusters: {len(new_superclusters)}")

start_time_1 = perf_counter()
for i, c in enumerate(new_superclusters):
    # sort pixels by likelihood
    c.pixels.sort(key=lambda x: x.likelihood, reverse=True)
    # downselect config.loud pixels
    c.pixels = c.pixels[:n_loudest]
    sub_net_cut(c.pixels, ml, FP, FX, acor, e2or, n_ifo, n_sky, subnet, subcut, subnorm, subrho,
                xtalk_coeff, xtalk_lookup_table, layers)

print(f"Time taken for sub_net_cut: {perf_counter() - start_time_1}")

print(f"Total time taken: {perf_counter() - start_time_all}")

# profiler
# import cProfile
# cProfile.run("supercluster(clusters, 'L', data['config'].gap, data['e2or'], data['config'].nIFO)", sort='cumtime')

time = timeit("supercluster(clusters, 'L', gap, e2or, n_ifo)", globals=globals(), number=100)
print(f"Time supercluster: {time/100} seconds per call")

time = timeit("defragment([sc for sc in superclusters if sc.cluster_status == 0],data['Tgap'], data['Fgap'], data['n_ifo'])", globals=globals(), number=100)
print(f"Time defragment: {time/100} seconds per call")