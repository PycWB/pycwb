# PyCWB Online Search — Shared-Memory (llhoft) Test Run

Self-contained run directory for testing the PyCWB online workflow reading
1-second GWF files from the LIGO low-latency shared-memory ring buffer.

## Directory contents

| File | Purpose |
|------|---------|
| `user_parameters.yaml` | Full run configuration (edit before running) |
| `online_schema_extension.yaml` | YAML schema for online extension params |
| `run.sh` | Continuous live launch script |
| `debug_run.sh`, `user_parameters_debug.yaml` | Real-time local fake-stream example |
| `fake_data_generator.py` | PyCBC noise/CBC generator and GWF writer |
| `_test_integration.py`, `_test_pipeline.py` | Bounded local frame-read and worker checks |

## Expected data layout on the server

```
/dev/shm/kafka/
    H1/
        H-H1_llhoft-1457805590-1.gwf
        H-H1_llhoft-1457805591-1.gwf
        ...
    L1/
        L-L1_llhoft-1457805590-1.gwf
        L-L1_llhoft-1457805591-1.gwf
        ...
```

The filename format is parsed as:
```
{site}-{ifo}_{stream}-{gps_start}-{duration}.gwf
```
Only the `gps_start` and `duration` fields are used. Files with any `duration`
value are supported, but the standard llhoft files are 1-second (`duration=1`).

## Channels

The config reads:
- `H1:GDS-CALIB_STRAIN_CLEAN_C00`
- `L1:GDS-CALIB_STRAIN_CLEAN_C00`

Edit `online_channels` in `user_parameters.yaml` if your llhoft frames carry
different channel names (e.g. `H1:GDS-CALIB_STRAIN` for older frames).

## Running

```bash
# From this directory:
bash run.sh

# Override number of workers:
bash run.sh --workers 8

# With debug logging:
bash run.sh --log-level DEBUG

# Or call pycwb directly:
pycwb online user_parameters.yaml --work-dir output --n-workers 4
```

## Local fake-data checks

Install PyCBC in the active PycWB environment (`python -m pip install pycbc`).
The GWF writer used by GWpy must also be available. From this directory, use a
fresh location for the fake frames:

```bash
python fake_data_generator.py --gps-start 1257894000 --duration 120 \
  --shm-base ./fake-stream --no-realtime
python _test_integration.py --shm-base ./fake-stream --gps-start 1257894000
python _test_pipeline.py --shm-base ./fake-stream --gps-start 1257894000 --segments 2
```

The checks read padding on both sides of the analysis windows. They raise errors
for missing data or failed stages. The worker check exercises two segments and
returns candidates locally; it does not start the continuous acquisition manager.
For that path, `bash debug_run.sh --duration 120` starts a real-time generator
and the online manager. The manager keeps polling after the generator finishes;
stop it with Ctrl-C. `PYCWB_PYTHON` and `PYCWB_BIN` optionally select explicit
executables; otherwise the active environment is used. The debug script clears
its fixed `/tmp/fake_stream` test directory before starting.

## Output

All output lands in `output/` (created automatically):
- `output/catalog/catalog.parquet` — local trigger catalog (updated continuously)
- `output/online_state.json` — crash-recovery checkpoint
- `output/triggers/seg_*/<event-hash>/` — cluster and sky-statistics JSON for each retained trigger

The online worker reconstructs waveforms for quality statistics but does not
currently persist the offline search's `wave.h5` products. Use the offline
workflow when those waveform files are needed.

## Key tuning parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `online_segment_duration` | 60 s | Full analysis window passed to CWB |
| `online_segment_stride` | 8 s | Slide interval; lower = less latency |
| `segEdge` | 8 s | Wavelet edge padding stripped each end |
| `online_n_workers` | 4 | Parallel segment workers |
| `online_data_source.timeout` | 30 s | Wait time before error if file missing |
| `netRHO` | 4.0 | CWB coherent SNR threshold |
| `netCC` | 0.5 | Network cross-correlation threshold |

## GraceDB alerts

Set `online_alert.gracedb: true` and ensure `LIGO_ACCOUNTING` / GraceDB
credentials are available in the environment before starting.
