import random 
from time import perf_counter
n_jobs = 60000
n_events = 1e6

start_time = perf_counter()

sample_event = [{
    "nevent": 1,
    "ndim": 2,
    "run": 0,
    "rho": [
        random.randint(10, 100),
        random.randint(10, 100)
    ],
    "netcc": [
        0,
        0,
        0
    ],
    "neted": [
        0,
        0,
        0,
        0,
        0
    ],
    "gnet": random.randint(0, 100),
    "anet": random.randint(0, 100),
    "inet": 0.0,
    "ecor": 779.2445068359375,
    "norm": 6.138232707977295,
    "ECOR": 0.0,
    "penalty": 0,
    "likelihood": 785.892578125,
    "factor": 0.0,
    "range": [
        0
    ],
    "chirp": [
        0,
        24.399999618530273,
        1.4213848114013672,
        0.9872614741325378,
        0.845714271068573,
        0.8970630168914795
    ],
    "eBBH": [],
    "usize": 0.0,
    "ifo_list": [],
    "eventID": [
        1,
        0
    ],
    "type": [
        1
    ],
    "name": [],
    "log": [],
    "rate": [
        0,
        0
    ],
    "volume": [
        432,
        178
    ],
    "size": [
        random.randint(10, 100),
        random.randint(10, 100)
    ],
    "gap": [
        0,
        0
    ],
    "lag": [
        0.0,
        0.0,
        0.0,
        0.0
    ],
    "slag": [
        0,
        0,
        0
    ],
    "strain": [
        6.880148223850065e-44
    ],
    "phi": [
        257.34375,
        0,
        38.66607516538352,
        259.453125
    ],
    "theta": [
        random.randint(10, 100),
        0,
        random.randint(10, 100),
        49.37981414794922
    ],
    "psi": [
        0.0
    ],
    "iota": [
        0.0
    ],
    "bp": [
        0.6385361548699656,
        -0.6341107153528613
    ],
    "bx": [
        -0.7364173133789823,
        0.7327122618773034
    ],
    "time": [
        1126259595.6280518,
        1126259595.6281734
    ],
    "gps": [
        1126259253.0,
        1126259253.0
    ],
    "right": [
        76.75,
        76.75
    ],
    "left": [
        342.25,
        342.25
    ],
    "duration": [
        0.04741816414320488,
        1.0
    ],
    "start": [
        1126259595.25,
        1126259595.25
    ],
    "stop": [
        1126259596.25,
        1126259596.25
    ],
    "frequency": [
        149.12086486816406,
        116.93898770299204
    ],
    "low": [
        28.0,
        28.0
    ],
    "high": [
        416.0,
        416.0
    ],
    "bandwidth": [
        83.89502942724967,
        388.0
    ],
    "hrss": [
        1.8618007566313623e-22,
        1.8476596457294163e-22
    ],
    "noise": [
        0.0078125,
        9.809437292968938e-24
    ],
    "erA": [],
    "Psm": [],
    "null": [
        36.03498458862305,
        28.38075065612793
    ],
    "nill": [
        27.054473876953125,
        16.872833251953125
    ],
    "mass": [],
    "spin": [],
    "snr": [
        411.7388610839844,
        464.72845458984375
    ],
    "xSNR": [
        382.6258544921875,
        447.1940612792969
    ],
    "sSNR": [
        355.5713806152344,
        430.32122802734375
    ],
    "iSNR": [],
    "oSNR": [],
    "ioSNR": [],
    "Deff": [],
    "injection": {
        "mass1": 30,
        "mass2": 20,
        "spin1z": 0,
        "spin2z": 0,
        "distance": 500,
        "inclination": 0.38915451612735563,
        "polarization": 1.140241362410411,
        "coa_phase": 0,
        "t_start": -2,
        "t_end": 0.5,
        "ra": 117.29125184362876,
        "dec": -16.65142046281714,
        "gps_time": 1126259595.6808321,
        "trail_idx": 0,
        "start_time": 1126259593.6808321,
        "end_time": 1126259596.1808321,
        "approximant": "IMRPhenomXPHM",
        "delta_t": 0.00006103515625,
        "f_lower": 16.0
    },
    "job_id": 2
} for _ in range(int(n_events))]

print(f"Time taken to generate events: {perf_counter() - start_time}")

start_time = perf_counter()
sample_job = [{
            "index": index,
            "ifos": [
                "L1",
                "H1"
            ],
            "start_time": start_time,
            "end_time": start_time + 4096,
            "sample_rate": 16384,
            "seg_edge": 10,
            "shift": None,
            "channels": [
                "L1:GWOSC-4KHZ_R1_STRAIN",
                "H1:GWOSC-4KHZ_R1_STRAIN"
            ],
            "frames": [
                {
                    "ifo": "L1",
                    "path": "./input/frames/L1_frames/L-L1_GWOSC_4KHZ_R1-1126257415-4096.gwf",
                    "start_time": start_time,
                    "duration": 4096
                },
                {
                    "ifo": "H1",
                    "path": "./input/frames/H1_frames/H-H1_GWOSC_4KHZ_R1-1126257415-4096.gwf",
                    "start_time": start_time,
                    "duration": 4096
                }
            ],
            "noise": None,
            "injections": [
                {
                    "mass1": random.randint(10, 50),
                    "mass2": random.randint(10, 50),
                    "spin1z": 0,
                    "spin2z": 0,
                    "distance": 500,
                    "inclination": 0.7398940058902834,
                    "polarization": 2.075067879167926,
                    "coa_phase": 0,
                    "t_start": -2,
                    "t_end": 0.5,
                    "ra": 88.10490293781076,
                    "dec": 5.552833857492445,
                    "gps_time": 1126258958.353076,
                    "trail_idx": 0,
                    "start_time": 1126258956.353076,
                    "end_time": 1126258958.853076
                },
                {
                    "mass1": random.randint(10, 50),
                    "mass2": random.randint(10, 50),
                    "spin1z": 0,
                    "spin2z": 0,
                    "distance": 500,
                    "inclination": 1.5264970455989935,
                    "polarization": 5.314055176762984,
                    "coa_phase": 0,
                    "t_start": -2,
                    "t_end": 0.5,
                    "ra": 180.56665839632885,
                    "dec": 33.499932631513005,
                    "gps_time": 1126258974.8114996,
                    "trail_idx": 1,
                    "start_time": 1126258972.8114996,
                    "end_time": 1126258975.3114996
                },
                {
                    "mass1": 30,
                    "mass2": 20,
                    "spin1z": 0,
                    "spin2z": 0,
                    "distance": 500,
                    "inclination": 1.1398165180527853,
                    "polarization": 3.650515995932227,
                    "coa_phase": 0,
                    "t_start": -2,
                    "t_end": 0.5,
                    "ra": 339.56772904070647,
                    "dec": -16.653710106043892,
                    "gps_time": 1126259148.1039732,
                    "trail_idx": 0,
                    "start_time": 1126259146.1039732,
                    "end_time": 1126259148.6039732
                },
                {
                    "mass1": 30,
                    "mass2": 20,
                    "spin1z": 0,
                    "spin2z": 0,
                    "distance": 500,
                    "inclination": 3.086063534554786,
                    "polarization": 0.1691368327017097,
                    "coa_phase": 0,
                    "t_start": -2,
                    "t_end": 0.5,
                    "ra": 5.16541812779769,
                    "dec": -19.671364176272075,
                    "gps_time": 1126259167.123404,
                    "trail_idx": 1,
                    "start_time": 1126259165.123404,
                    "end_time": 1126259167.623404
                }
            ],
            "trail_idx": random.randint(0, 10),
        } for start_time in range(1126258863, 1126259263, n_jobs) for index in range(n_jobs)]

print(f"Time taken to generate jobs: {perf_counter() - start_time}")

generated_output = {
    "config": {},
    "jobs": sample_job,
    "events": sample_event
}

# save as compressed json

import orjson
import gzip

start_time = perf_counter()
with gzip.open("data.json.gz", 'wb') as f:
    f.write(orjson.dumps(generated_output, option=orjson.OPT_SERIALIZE_NUMPY))

print(f"Time taken to write to file: {perf_counter() - start_time}")


# save as parquet
import pyarrow as pa
import pyarrow.parquet as pq

# Parquet write test
start_time = perf_counter()

# Convert data to Arrow Tables
jobs_table = pa.Table.from_pylist(sample_job)
events_table = pa.Table.from_pylist(sample_event)

# Write to Parquet files
pq.write_table(jobs_table, 'jobs.parquet')
pq.write_table(events_table, 'events.parquet')

print(f"Time taken to write Parquet files: {perf_counter() - start_time}")

# Parquet read test
start_time = perf_counter()

# Read from Parquet files
jobs_table = pq.read_table('jobs.parquet')
events_table = pq.read_table('events.parquet')

# Convert back to Python objects if needed (optional)
# jobs_data = jobs_table.to_pylist()
# events_data = events_table.to_pylist()

print(f"Time taken to read Parquet files: {perf_counter() - start_time}")