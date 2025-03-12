import pyarrow.parquet as pq
import pyarrow.compute as pc
from time import perf_counter

# Parquet read test
start_time = perf_counter()

# Read from Parquet files
jobs_table = pq.read_table('jobs.parquet')
events_table = pq.read_table('events.parquet')

# Convert back to Python objects if needed (optional)
# jobs_data = jobs_table.to_pylist()
# events_data = events_table.to_pylist()

print(f"Time taken to read Parquet files: {perf_counter() - start_time}")

print(f'number of rows in jobs_table: {jobs_table.num_rows}')
print(f'number of rows in events_table: {events_table.num_rows}')


start_time = perf_counter()
first_event = events_table.slice(0, 1).to_pylist()[0]

print("First event:", first_event)
print(f"Time taken to convert to Python object: {perf_counter() - start_time}")

start_time = perf_counter()
# Get first element of rho arrays
rho_first = pc.list_element(events_table['rho'], index=0)

# Create boolean mask for filtering
mask = pc.greater(rho_first, 90)

# Apply filter to get matching rows
filtered_table = events_table.filter(mask)

# Get indexes (if needed) - this converts to numpy array
indices = pa.compute.field('index')  # If you have an index column
filtered_indices = indices.filter(mask).to_numpy()

# Convert to Python objects if needed
filtered_events = filtered_table.to_pylist()

print(f"Found {len(filtered_events)} events with rho[0] > 90")
print("First matching event:", filtered_events[0])
print(f"Time taken to filter: {perf_counter() - start_time}")