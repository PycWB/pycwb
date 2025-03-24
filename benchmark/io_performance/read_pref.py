import orjson
import gzip
from time import perf_counter
for i in range(3):
    start_time = perf_counter()
    # load compressed json
    with gzip.open("data-small.json.gz", 'rb') as f:
        data = orjson.loads(f.read())

    print(f"Time taken to read from file: {perf_counter() - start_time}")
    del data
