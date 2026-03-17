# PyCWB Online Workflow — Implementation

This document describes the implemented online gravitational-wave search workflow:
what was built, where each piece lives, how they connect, and how to configure and
run the pipeline.

---

## Overview

The online workflow is a long-running process that:

1. **Acquires data** continuously from a live data stream (NDS2/gwpy, with stubs for
   Kafka and shared memory)
2. **Buffers** each IFO channel in a per-IFO in-memory ring buffer
3. **Emits analysis segments** on a sliding window (`duration=60 s`, `stride=20 s`
   by default), reducing worst-case latency to `stride + processing_time ≈ 50 s`
4. **Analyses each segment** in a worker process using the existing PyCWB pipeline
   with intra-segment parallelism via `ThreadPoolExecutor`
5. **Deduplicates** triggers from overlapping windows before performing significance
   assignment
6. **Dispatches** finalized triggers to GraceDB, a local catalog, and/or a webhook

The design philosophy is **clean separation**: all analysis modules are called
unchanged through their existing per-item functions.  All orchestration and
parallelism lives in the new workflow layer.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      OnlineSearchManager                             │
│  (Main process — owns lifecycle of all components)                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────┐      ┌────────────────────┐                  │
│  │ DataAcquisition    │ ──→  │  SegmentQueue       │                 │
│  │ Thread             │      │  (bounded,          │                 │
│  │                    │      │   backpressure)      │                 │
│  │ • Poll 1 s chunks  │      └──────────┬──────────┘                 │
│  │ • RingBuffer / IFO │                 │                            │
│  │ • Snapshot → queue │                 ▼                            │
│  └────────────────────┘     ┌───────────────────────────┐            │
│                             │  AnalysisWorkerPool        │            │
│                             │  (ProcessPoolExecutor)     │            │
│                             │                            │            │
│                             │  Per segment (lag=0 only): │            │
│                             │  ├ check_and_resample ║    │            │
│                             │  ├ data_conditioning  ║    │            │
│                             │  ├ setup_coherence ─┐ ║    │            │
│                             │  ├ build_td_cache   ─┤ parallel        │
│                             │  ├ setup_supercluster┘ ║   │            │
│                             │  ├ setup_likelihood     │   │           │
│                             │  ├ coherence_single_lag │   │           │
│                             │  ├ supercluster_single  │   │           │
│                             │  ├ likelihood           │   │           │
│                             │  └ reconstruction + Qveto   │           │
│                             └──────────┬──────────────────┘           │
│                                        │                              │
│                                        ▼                              │
│                             ┌───────────────────────────┐             │
│                             │  TriggerHandler Thread     │            │
│                             │  • Deduplication           │            │
│                             │  • XGBoost significance    │            │
│                             │  • Local catalog           │            │
│                             │  • GraceDB upload          │            │
│                             │  • Webhook callback        │            │
│                             └───────────────────────────┘             │
│                                                                       │
│  ┌────────────────────┐                                               │
│  │ LatencyMonitor     │  queue depth · latency · staleness           │
│  └────────────────────┘                                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## File Inventory

### New Files

| File | Description |
|------|-------------|
| `pycwb/types/online.py` | `OnlineSegment` and `OnlineTrigger` dataclasses |
| `pycwb/modules/online/__init__.py` | Package init |
| `pycwb/modules/online/data_source.py` | `DataSource` ABC + `NDS2DataSource` + stubs |
| `pycwb/modules/online/ring_buffer.py` | Thread-safe per-IFO ring buffer |
| `pycwb/modules/online/data_acquisition.py` | `DataAcquisitionManager` daemon thread |
| `pycwb/modules/online/significance.py` | XGBoost model loading + IFAR assignment |
| `pycwb/modules/online/deduplication.py` | `TriggerDeduplicator` — sliding-window dedup |
| `pycwb/modules/online/trigger_handler.py` | `TriggerHandler` daemon thread |
| `pycwb/modules/online/latency_monitor.py` | `LatencyMonitor` daemon thread |
| `pycwb/modules/online/background.py` | `BackgroundManager` placeholder (deferred) |
| `pycwb/workflow/online.py` | `OnlineSearchManager` orchestrator |
| `pycwb/workflow/subflow/process_online_segment.py` | Parallel online segment processor |
| `pycwb/vendor/online/online_schema_extension.yaml` | JSON-schema extension for all `online_*` params |
| `pycwb/vendor/online/user_parameters_online.yaml` | Example online config |
| `pycwb/cli/online.py` | CLI entry point (`pycwb online`) |

### Modified Files

| File | Change |
|------|--------|
| `pycwb/utils/td_vector_batch.py` | Extracted `_build_td_inputs_single_level` (per-level callable for parallel use) |
| `pycwb/modules/gracedb/gracedb.py` | Added `upload_online_event()`, `upload_skymap()`, `write_log()` |
| `bin/pycwb` | Registered `online` subcommand |

### NOT Modified

All analysis modules (`cwb_coherence`, `data_conditioning`, `super_cluster`,
`likelihoodWP`, `reconstruction`, `qveto`) are called unchanged through their
existing per-item functions.  `pycwb/config/config.py` and the default schema
are also untouched — new parameters are delivered via the `pycwb_schema`
extension mechanism.

---

## Data Types

### `OnlineSegment` (`pycwb/types/online.py`)

Carries pre-read detector data together with GPS metadata and wall-clock timing.
The `data_payload` spans `[segment_gps_start - seg_edge, segment_gps_end + seg_edge]`
so analysis modules see the same edge-padded window they expect from the
file-based pipeline.

```python
@dataclass
class OnlineSegment:
    index: int              # monotonically increasing counter
    ifos: List[str]         # detector names (same order as config.ifo)
    segment_gps_start: float
    segment_gps_end: float
    seg_edge: float         # wavelet boundary padding (seconds)
    sample_rate: float
    data_payload: list      # one array per IFO, includes seg_edge padding
    wall_time_received: float
    stride: float           # seconds of NEW data in this segment
    overlap_frac: float     # fraction shared with previous segment
```

### `OnlineTrigger` (`pycwb/types/online.py`)

```python
@dataclass
class OnlineTrigger:
    event: Event
    cluster: object
    sky_stats: object
    segment_index: int
    segment_gps: float
    wall_time_done: float
```

---

## Intra-Segment Parallelism

`process_online_segment()` runs inside a `ProcessPoolExecutor` worker and uses
`ThreadPoolExecutor` for true concurrency (Numba `@prange`, JAX `jit/vmap`, and
BLAS all release the GIL).

```
STEP 1 — Parallel resample          ThreadPoolExecutor(nIFO)
STEP 2 — Parallel conditioning      ThreadPoolExecutor(nIFO)
STEP 3 — Overlapped setup (3 tasks) ThreadPoolExecutor(3)
          ├─ coherence setup          inner ThreadPoolExecutor(nRES)
          ├─ TD cache build           inner ThreadPoolExecutor(nLevels)
          └─ supercluster + xtalk
STEP 4 — Likelihood setup           (sequential, fast)
STEP 5 — lag=0 analysis             (coherence → supercluster → likelihood)
          └─ per-cluster postprocess
               ├─ 4× reconstruction  ThreadPoolExecutor(4)
               └─ Q-veto             ThreadPoolExecutor(2×nIFO)
```

The processing function is a **drop-in counterpart** of
`process_job_segment_native.process_job_segment()`.  Differences:

| Offline step | Online adaptation |
|---|---|
| `read_from_job_segment()` | **Skipped** — data already in `OnlineSegment.data_payload` |
| Trail loop | **Removed** — single pass, `trail_idx=0` |
| Lag loop | **Removed** — single pass, `lag=0` (zero-lag only) |
| MDC / injection steps | **Skipped** |
| `save_trigger()` | **Deferred** — returns `list[OnlineTrigger]` to caller |

`OnlineSegment` is converted to a minimal `WaveSegment` (the bridge function
`_online_seg_to_wave_seg()`) so all downstream modules receive the interface they
expect.

---

## Data Acquisition

### Sliding Window

```
duration = 60 s,  stride = 20 s

GPS:  0       20      40      60      80
Seg 0: |............60 s.............|
Seg 1:         |............60 s.............|
Seg 2:                 |............60 s.............|
```

- A new segment is emitted every `stride` seconds of wall-clock time
- Segments overlap by `duration - stride = 40 s`
- Worst-case latency: `stride + processing_time`
- `duration / stride` = 3× more segments to process → requires `≥ 3` workers

### `RingBuffer` memory budget

```
capacity ≥ duration + stride + 2 × segEdge
         = 60 + 20 + 16 = 96 s at 16384 Hz ≈ 12 MB / IFO
```

### GPS gap handling

- Gaps < 1 sample: silently absorbed
- Gaps > 1 sample: warning logged, ring buffer reset; `DataAcquisitionManager`
  will re-fill before emitting the next segment
- Reconnection on `read_chunk()` failure: exponential backoff (1 s → 60 s max)

---

## Trigger Deduplication

With `stride < duration`, the same astrophysical event is detected in up to
`ceil(duration / stride)` consecutive segments.  `TriggerDeduplicator` merges
duplicates before they reach GraceDB.

**Algorithm:**
1. When a trigger arrives, check all `pending` triggers for GPS + sky match
2. Match condition: `|GPS_a - GPS_b| < gps_window` **and** `angular_distance < sky_tolerance`
3. Keep the trigger with the higher `ranking_statistic`
4. Flush triggers whose `wall_time_done` exceeds `flush_delay = duration + 2 × stride`
   — enough time for all overlapping duplicates to arrive

**No-op when `stride = duration`:** deduplication still runs but the flush delay
is minimal and no merges occur.

---

## Significance Assignment

XGBoost model + IFAR lookup table are loaded **once** at `TriggerHandler` startup:

```python
model, ifar_table = load_significance_model(model_path, ifar_file)
```

Per trigger:
```python
assign_significance(event, model, ifar_table, feature_columns)
# sets: event.ranking_statistic, event.ifar
```

Default features: `["rho", "netcc", "penalty", "ecor", "qveto", "qfactor"]`

If `model_path` or `ifar_file` is empty, significance assignment is silently
skipped (events are still saved locally).

---

## Alert Dispatch

`TriggerHandler` processes each finalized, deduplicated trigger in order:

1. **Assign significance** (XGBoost + IFAR)
2. **Save locally** — `save_trigger()` + `add_event_to_catalog()` under
   `working_dir/triggers/seg_NNNNNN/`
3. **GraceDB upload** — if `online_alert.gracedb = true` and
   `event.ifar ≥ gracedb_ifar_threshold`
4. **Webhook** — POST JSON payload to `online_alert.webhook_url`

Errors in alerting are logged but never crash the pipeline.

---

## GraceDB Extensions

Three new functions in `pycwb/modules/gracedb/gracedb.py`:

| Function | Description |
|----------|-------------|
| `upload_online_event(event, group, pipeline, search)` | Create a GraceDB event; returns `graceid` |
| `upload_skymap(graceid, skymap_data, filename)` | Upload a HEALPix skymap to an event |
| `write_log(graceid, message, tag_name)` | Attach a log message to an event |

---

## Configuration

All online parameters are defined in the schema extension file.  Users opt in by
adding a `pycwb_schema` block to their YAML:

```yaml
pycwb_schema:
  mode: extend
  schema_file: /path/to/pycwb/vendor/online/online_schema_extension.yaml
```

The existing `Config` dataclass is **not modified**.  All `online_*` attributes
are accessed via `getattr(config, 'online_*', default)`.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `online_segment_duration` | 60 | Analysis window length (seconds) |
| `online_segment_stride` | 20 | Slide step (seconds); set equal to duration to disable overlap |
| `online_poll_interval` | 1 | Data acquisition poll interval (seconds) |
| `online_n_workers` | 4 | Analysis worker processes |
| `online_max_queue_depth` | 8 | Bounded segment queue size |
| `online_channels` | `[]` | Channel names, one per IFO, same order as `config.ifo` |
| `online_data_source.type` | `"nds2"` | Adapter: `"nds2"`, `"kafka"` (stub), `"shm"` (stub) |
| `online_latency_threshold` | 30 | Warning threshold for processing latency (seconds) |
| `online_dedup_window` | 0.5 | GPS coincidence window for deduplication (seconds) |
| `online_dedup_sky_tolerance` | 5.0 | Sky coincidence tolerance for deduplication (degrees) |
| `online_significance.model_path` | `""` | XGBoost model file |
| `online_significance.ifar_file` | `""` | IFAR lookup table (CSV or `.npz`) |
| `online_alert.gracedb` | `false` | Enable GraceDB upload |
| `online_alert.gracedb_ifar_threshold` | `0.0` | Minimum IFAR (years) to upload |
| `online_alert.local_catalog` | `true` | Save triggers to local catalog |
| `online_alert.webhook_url` | `""` | HTTP POST webhook URL |
| `online_state_file` | `"online_state.json"` | GPS state file for crash recovery |

A complete annotated example is at `pycwb/vendor/online/user_parameters_online.yaml`.

---

## Running

```bash
pycwb online my_config.yaml --work-dir /path/to/output --n-workers 4
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `config_file` | (required) | Path to YAML config |
| `--work-dir` / `-d` | `.` | Output directory |
| `--n-workers` / `-n` | from config | Override `online_n_workers` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Send `SIGINT` or `SIGTERM` to initiate a graceful shutdown:  in-flight segments
are allowed to complete, the trigger handler drains its queue, and the last
processed GPS is saved to `online_state.json` for crash recovery.

---

## Crash Recovery

On every successfully analysed segment, the GPS end time is persisted:

```json
{ "last_processed_gps": 1234567890.0 }
```

On restart, `DataAcquisitionManager` resumes data acquisition from that GPS.
The state file path is configurable via `online_state_file` (relative to
`--work-dir`).

---

## Health Monitoring

`LatencyMonitor` wakes every `online_monitor_interval` seconds (default 5) and
logs:

| Metric | Warning condition |
|--------|------------------|
| Segment queue depth | > `max_queue_depth / 2` |
| Per-IFO data staleness | > `2 × online_segment_duration` |
| Processing latency (avg + max) | max > `online_latency_threshold` |

---

## Latency Budget

With default settings (`duration=60 s`, `stride=20 s`, `processing≈18 s`):

| Phase | Time |
|-------|------|
| Wait for new stride data | ≤ 20 s |
| Parallel resample | ~0.5 s |
| Parallel conditioning | ~3 s |
| Overlapped setup (coherence + TD + SC) | ~6 s |
| lag=0 analysis (coherence → SC → likelihood) | ~7 s |
| Reconstruction + Q-veto | ~2 s |
| **Total analysis** | **~18.5 s** |
| **Wall-clock latency (worst case)** | **~38.5 s** |

---

## Background Estimation (Deferred)

`BackgroundManager` (`pycwb/modules/online/background.py`) reserves the interface
for future automatic background job management (Condor/Slurm submission, result
harvesting, XGBoost retraining).  The initial release uses pre-trained model +
IFAR files.

---

## Dependencies

| Library | Usage |
|---------|-------|
| `gwpy` | `NDS2DataSource.read_chunk()` — `TimeSeriesDict.get()` |
| `xgboost` | `significance.py` — ranking statistic prediction |
| `numpy` | Ring buffer operations, IFAR interpolation |
| `requests` | Webhook POST in `TriggerHandler` |
| `ligo.gracedb` | GraceDB client in `gracedb.py` |

All are optional at import time: missing libraries produce a log warning rather
than a crash at startup.
