# Working with the pycWB Catalog: Analysis & Plotting Guide

This guide demonstrates how to load, query, filter, and visualise triggers
from a pycWB Arrow/Parquet catalog.  All examples use the
`Catalog` class and standard Python data-science libraries
(`pandas`, `matplotlib`, `numpy`).

> **Sample file used below:** `catalog.parquet` (SGE injection run,
> ~470 triggers, 2 jobs, HL network).  Adjust the path to your own catalog.

---

## 0. Setup

```python
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycwb.modules.catalog import Catalog

cat = Catalog.open("catalog.parquet")
print(cat)                       # Catalog('…', triggers=471)
```

---

## 1. Basic Trigger Table

```python
table = cat.triggers()           # pyarrow.Table
df = table.to_pandas()           # pandas DataFrame – one row per trigger
df.head()
```

### Expand the injection struct into readable columns

The `injection` column is a nested Arrow struct.  Background triggers
have `injection = None`, so filter those out before building the DataFrame:

```python
inj_rows = [r for r in table.column("injection").to_pylist() if r is not None]
inj_df = pd.DataFrame(inj_rows)
inj_df.head()
```

---

## 2. 1-D Histogram of `rho`

```python
fig, ax = plt.subplots()
ax.hist(df["rho"], bins=50, edgecolor="k", alpha=0.7)
ax.set_xlabel("rho (effective correlated SNR)")
ax.set_ylabel("Count")
ax.set_title("Distribution of rho")
ax.set_yscale("log")
plt.tight_layout()
plt.show()
```

---

## 3. 2-D Scatter of Q_veto vs Q_factor

`q_veto` and `q_factor` are network quality factors
stored directly on each trigger:

```python
fig, ax = plt.subplots()
sc = ax.scatter(df["q_veto"], df["q_factor"],
                c=df["rho"], cmap="viridis", s=10, alpha=0.6)
fig.colorbar(sc, ax=ax, label="rho")
ax.set_xlabel("Qveto")
ax.set_ylabel("Qfactor")
ax.set_title("Qveto vs Qfactor coloured by rho")
plt.tight_layout()
plt.show()
```

---

## 4. Filter Injections by Name

### Using `Catalog.filter` (Python expression)

```python
# filter() works on top-level columns
sgq9 = cat.filter("rho > 5")
```

### Using `Catalog.query` with DuckDB (struct sub-fields)

```python
# Filter by injection.name using DuckDB dot notation
q9_table = cat.query("""
    SELECT *
    FROM   triggers
    WHERE  injection IS NOT NULL
      AND  injection.name LIKE 'SGE_Q9_%'
""")
print(f"Q=9 triggers: {q9_table.num_rows}")
q9_df = q9_table.to_pandas()
```

### Plot rho of the filtered triggers

```python
fig, ax = plt.subplots()
ax.hist(q9_df["rho"], bins=30, edgecolor="k", alpha=0.7, label="Q=9")
ax.set_xlabel("rho")
ax.set_ylabel("Count")
ax.set_title("rho distribution for Q=9 injections")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## 5. Filter Injections by Q and Frequency from the JSON Parameters

The full injection dict is stored as a JSON string in
`injection.parameters`.  Use DuckDB's `json_extract` to query
waveform-specific fields (Q, frequency, bandwidth, …):

```python
# Select triggers with Q=100 and frequency > 500 Hz
high_freq_q100 = cat.query("""
    SELECT id, rho, injection.name, injection.hrss,
           json_extract(injection.parameters, '$.Q')::FLOAT         AS Q,
           json_extract(injection.parameters, '$.frequency')::FLOAT AS freq
    FROM   triggers
    WHERE  injection IS NOT NULL
      AND  json_extract(injection.parameters, '$.Q')::FLOAT = 100
      AND  json_extract(injection.parameters, '$.frequency')::FLOAT > 500
""")
print(high_freq_q100.to_pandas())
```

### Plot rho of the filtered subset

```python
filtered_df = high_freq_q100.to_pandas()
fig, ax = plt.subplots()
ax.hist(filtered_df["rho"], bins=20, edgecolor="k", alpha=0.7)
ax.set_xlabel("rho")
ax.set_ylabel("Count")
ax.set_title("rho for Q=100, freq > 500 Hz")
plt.tight_layout()
plt.show()
```

---

## 6. 2-D Heatmap of Injected Q vs Frequency

Extract Q and frequency from `injection.parameters` for all recovered
triggers and plot a 2-D heatmap of recovery counts:

```python
# Extract Q and frequency via DuckDB
qf_table = cat.query("""
    SELECT json_extract(injection.parameters, '$.Q')::FLOAT         AS Q,
           json_extract(injection.parameters, '$.frequency')::FLOAT AS freq
    FROM   triggers
    WHERE  injection IS NOT NULL
""")
qf_df = qf_table.to_pandas()

# Count occurrences per (Q, freq) bin
counts = qf_df.groupby(["Q", "freq"]).size().reset_index(name="count")
pivot = counts.pivot(index="Q", columns="freq", values="count").fillna(0)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(pivot.values, aspect="auto", origin="lower",
               extent=[pivot.columns.min(), pivot.columns.max(),
                       pivot.index.min(), pivot.index.max()])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Q")
ax.set_title("Recovered trigger count by injected (Q, Frequency)")
fig.colorbar(im, ax=ax, label="Count")
plt.tight_layout()
plt.show()
```

For a discrete heatmap (recommended when Q/frequency values are categorical):

```python
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.pcolormesh(
    range(len(pivot.columns) + 1),
    range(len(pivot.index) + 1),
    pivot.values, cmap="YlOrRd", edgecolors="white", linewidth=0.5,
)
ax.set_xticks([i + 0.5 for i in range(len(pivot.columns))],
              [f"{int(f)}" for f in pivot.columns], rotation=45)
ax.set_yticks([i + 0.5 for i in range(len(pivot.index))],
              [f"{q:.0f}" for q in pivot.index])
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Q")
ax.set_title("Recovered trigger count by injected (Q, Frequency)")
fig.colorbar(im, ax=ax, label="Count")
plt.tight_layout()
plt.show()
```

---

## 7. Sky Position of Injected Parameters

Compare the injected sky position with the reconstructed sky position:

```python
inj_table = cat.query("""
    SELECT injection.ra  AS inj_ra,
           injection.dec AS inj_dec,
           ra            AS rec_ra,
           dec           AS rec_dec,
           rho
    FROM   triggers
    WHERE  injection IS NOT NULL
""")
sky_df = inj_table.to_pandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Injected sky positions
ax = axes[0]
sc = ax.scatter(np.degrees(sky_df["inj_ra"]), np.degrees(sky_df["inj_dec"]),
                c=sky_df["rho"], cmap="viridis", s=10, alpha=0.6)
fig.colorbar(sc, ax=ax, label="rho")
ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dec (deg)")
ax.set_title("Injected sky position")

# Reconstructed sky positions
ax = axes[1]
sc = ax.scatter(np.degrees(sky_df["rec_ra"]), np.degrees(sky_df["rec_dec"]),
                c=sky_df["rho"], cmap="viridis", s=10, alpha=0.6)
fig.colorbar(sc, ax=ax, label="rho")
ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dec (deg)")
ax.set_title("Reconstructed sky position")

plt.tight_layout()
plt.show()
```

### Sky position error

```python
# Angular separation (small-angle approximation)
sky_df["delta_ra"]  = np.degrees(sky_df["rec_ra"]  - sky_df["inj_ra"])
sky_df["delta_dec"] = np.degrees(sky_df["rec_dec"] - sky_df["inj_dec"])

fig, ax = plt.subplots()
sc = ax.scatter(sky_df["delta_ra"], sky_df["delta_dec"],
                c=sky_df["rho"], cmap="viridis", s=10, alpha=0.6)
fig.colorbar(sc, ax=ax, label="rho")
ax.axhline(0, color="grey", ls="--", lw=0.5)
ax.axvline(0, color="grey", ls="--", lw=0.5)
ax.set_xlabel("ΔRA (deg)")
ax.set_ylabel("ΔDec (deg)")
ax.set_title("Sky position residual (reconstructed − injected)")
plt.tight_layout()
plt.show()
```

---

## 8. Job Information and Injections Within a Job

The catalog stores the full list of job segments (including **all**
scheduled injections) in the Parquet metadata.  This includes injections
that were **not** recovered — essential for computing detection efficiency.

```python
jobs = cat.jobs                  # list of job dicts
print(f"Number of jobs: {len(jobs)}")

for j in jobs:
    print(f"  Job {j['index']}: {j['analyze_start']} – {j['analyze_end']}, "
          f"{len(j.get('injections', []))} injections scheduled")
```

### Access injections for a specific job

```python
job = jobs[0]
job_inj = pd.DataFrame(job["injections"])
print(f"Scheduled injections in job {job['index']}: {len(job_inj)}")
job_inj.head()
```

---

## 9. Compare Injected vs Recovered: Finding Missing Injections

For each job, compare the full injection schedule (from metadata) with the
triggers actually recovered. This reveals which (Q, frequency)
combinations the pipeline missed.

```python
job = jobs[0]
job_id = job["index"]

# --- All scheduled injections in this job ---
sched_df = pd.DataFrame(job["injections"])
sched_counts = sched_df.groupby(["Q", "frequency"]).size().reset_index(name="scheduled")

# --- Recovered triggers for this job ---
rec_table = cat.query(f"""
    SELECT json_extract(injection.parameters, '$.Q')::FLOAT         AS Q,
           json_extract(injection.parameters, '$.frequency')::FLOAT AS freq
    FROM   triggers
    WHERE  injection IS NOT NULL
      AND  job_id = {job_id}
""")
rec_df = rec_table.to_pandas()
rec_counts = rec_df.groupby(["Q", "freq"]).size().reset_index(name="recovered")
rec_counts.rename(columns={"freq": "frequency"}, inplace=True)

# --- Merge ---
merged = sched_counts.merge(rec_counts, on=["Q", "frequency"], how="left").fillna(0)
merged["recovered"] = merged["recovered"].astype(int)
merged["missed"] = merged["scheduled"] - merged["recovered"]
merged["efficiency"] = merged["recovered"] / merged["scheduled"]
print(merged.to_string(index=False))
```

### Heatmap: scheduled vs recovered vs efficiency

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col, title, cmap in zip(
    axes,
    ["scheduled", "recovered", "efficiency"],
    ["Scheduled injections", "Recovered triggers", "Recovery efficiency"],
    ["Blues", "Greens", "RdYlGn"],
):
    pivot = merged.pivot(index="Q", columns="frequency", values=col).fillna(0)
    im = ax.pcolormesh(
        range(len(pivot.columns) + 1),
        range(len(pivot.index) + 1),
        pivot.values, cmap=cmap, edgecolors="white", linewidth=0.5,
    )
    ax.set_xticks([i + 0.5 for i in range(len(pivot.columns))],
                  [f"{int(f)}" for f in pivot.columns], rotation=45, fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(len(pivot.index))],
                  [f"{q:.0f}" for q in pivot.index])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Q")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

### Which (Q, frequency) had zero recoveries?

```python
missing = merged[merged["recovered"] == 0]
if len(missing):
    print("Completely missed (Q, frequency) combinations:")
    print(missing[["Q", "frequency", "scheduled"]].to_string(index=False))
else:
    print("All (Q, frequency) combinations had at least one recovery.")
```

---

## 10. Per-hrss Detection Efficiency Curve

For injection studies it is common to plot the fraction of recovered
triggers as a function of injected hrss for each waveform family:

```python
job = jobs[0]
job_id = job["index"]

# Scheduled: count per (name, hrss)
sched_df = pd.DataFrame(job["injections"])
sched_by_hrss = sched_df.groupby(["name", "hrss"]).size().reset_index(name="scheduled")

# Recovered: count per (name, hrss)
rec_hrss = cat.query(f"""
    SELECT injection.name,
           injection.hrss
    FROM   triggers
    WHERE  injection IS NOT NULL
      AND  job_id = {job_id}
""").to_pandas()
rec_by_hrss = rec_hrss.groupby(["name", "hrss"]).size().reset_index(name="recovered")

eff = sched_by_hrss.merge(rec_by_hrss, on=["name", "hrss"], how="left").fillna(0)
eff["recovered"] = eff["recovered"].astype(int)
eff["efficiency"] = eff["recovered"] / eff["scheduled"]

# Plot one curve per waveform name
fig, ax = plt.subplots(figsize=(8, 5))
for name, grp in eff.groupby("name"):
    grp = grp.sort_values("hrss")
    ax.plot(grp["hrss"], grp["efficiency"], "o-", ms=4, label=name)

ax.set_xscale("log")
ax.set_xlabel("Injected hrss")
ax.set_ylabel("Recovery efficiency")
ax.set_title("Detection efficiency vs hrss")
ax.legend(fontsize=7, ncol=3, loc="lower right")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Quick Reference: Useful DuckDB Queries

```python
# All injection names and counts
cat.query("""
    SELECT injection.name, COUNT(*) AS n
    FROM   triggers
    WHERE  injection IS NOT NULL
    GROUP BY injection.name
    ORDER BY n DESC
""")

# Top-10 loudest injection triggers
cat.query("""
    SELECT id, rho, injection.name, injection.hrss, injection.approximant
    FROM   triggers
    WHERE  injection IS NOT NULL
    ORDER BY rho DESC
    LIMIT 10
""")

# Average rho per (Q, frequency) bin
cat.query("""
    SELECT json_extract(injection.parameters, '$.Q')::FLOAT         AS Q,
           json_extract(injection.parameters, '$.frequency')::FLOAT AS freq,
           AVG(rho)  AS avg_rho,
           COUNT(*)  AS n
    FROM   triggers
    WHERE  injection IS NOT NULL
    GROUP BY Q, freq
    ORDER BY Q, freq
""")
```
