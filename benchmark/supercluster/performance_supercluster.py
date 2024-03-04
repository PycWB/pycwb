import pickle

from pycwb.modules.likelihood import likelihood
from pycwb.modules.super_cluster.sub_net_cut import sub_net_cut

# with open('/Users/yumengxu/GWOSC/catalog/GWTC-1-confident/GW150914/pycWB/test_data_1.pkl', 'rb') as f:
#     data = pickle.load(f)
with open('./test_data.pkl', 'rb') as f:
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
subrho = data['subrho'] if data['subrho'] > 0 else data['netrho']
netrho = data['netrho']
xtalk_coeff = data['xtalk_coeff']
xtalk_lookup_table = data['xtalk_lookup_table']
layers = data['layers']
nRes = data['nRes']
Tgap = data['Tgap']
Fgap = data['Fgap']
gap = data['gap']
e2or = data['e2or']

print(subrho)
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


total_pixels = 0
for i, c in enumerate(superclusters):
    total_pixels += len(c.pixels)
    # print(f'supercluster {i} has {n_pix} pixels and {"rejected" if c.cluster_status > 0 else "accepted"}')

# filter out the rejected superclusters
accepted_superclusters = [sc for sc in superclusters if sc.cluster_status <= 0]
print(f"Total number of superclusters: {len(superclusters)}, total pixels: {total_pixels}, "
      f"accepted clusters: {len(accepted_superclusters)}")

start_time = perf_counter()
new_superclusters = defragment(accepted_superclusters, Tgap, Fgap, n_ifo)
print(f"Time taken for defragment: {perf_counter() - start_time}")

total_pixels = 0
for i, c in enumerate(superclusters):
    total_pixels += len(c.pixels)
    # print(f'supercluster {i} has {n_pix} pixels and {"rejected" if c.cluster_status != 0 else "accepted"}')
print(f"Total number of superclusters after defragment: {len(new_superclusters)}, total pixels: {total_pixels}")

start_time_1 = perf_counter()
for i, c in enumerate(new_superclusters):
    # sort pixels by likelihood
    c.pixels.sort(key=lambda x: x.likelihood, reverse=True)
    # downselect config.loud pixels
    results = sub_net_cut(c.pixels[:n_loudest], ml, FP, FX, acor, e2or, n_ifo, n_sky, subnet, subcut, subnorm, subrho,
                xtalk_coeff, xtalk_lookup_table, layers)
    # update cluster status and print results
    if results['subnet_passed'] and results['subrho_passed'] and results['subthr_passed']:
        print(f"Cluster {i} passed subnet, subrho, and subthr cut")
        c.cluster_status = -1
    else:
        log_output = f"Cluster {i} failed "
        if not results['subnet_passed']:
            log_output += f"subnet cut condition: {results['subnet_condition']}, "
        if not results['subrho_passed']:
            log_output += f"subrho cut condition: {results['subrho_condition']}, "
        if not results['subthr_passed']:
            log_output += f"subthr cut condition: {results['subthr_condition']}, "
        # print(log_output)
        c.cluster_status = 1

# fragment_clusters.clusters = new_superclusters
fragment_clusters.clusters = [c for c in new_superclusters if c.cluster_status <= 0]

total_pixels = 0
for i, c in enumerate(fragment_clusters.clusters):
    total_pixels += len(c.pixels)
print(f"Total number of superclusters after sub_net_cut: {len(fragment_clusters.clusters)}, total pixels: {total_pixels}")

for c in fragment_clusters.clusters:
    for p in c.pixels:
        p.core = 1
        p.td_amp = None

print(f"Time taken for sub_net_cut: {perf_counter() - start_time_1}")

print(f"Total time taken: {perf_counter() - start_time_all}")

likelihood(data['config'], data['network'], [fragment_clusters])

# profiler
# import cProfile
# cProfile.run("supercluster(clusters, 'L', data['config'].gap, data['e2or'], data['config'].nIFO)", sort='cumtime')

time = timeit("supercluster(clusters, 'L', gap, e2or, n_ifo)", globals=globals(), number=100)
print(f"Time supercluster: {time/100} seconds per call")

time = timeit("defragment([sc for sc in superclusters if sc.cluster_status <= 0], Tgap, Fgap, n_ifo)", globals=globals(), number=100)
print(f"Time defragment: {time/100} seconds per call")