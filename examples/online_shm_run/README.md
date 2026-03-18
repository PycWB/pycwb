# PyCWB Online Search — Shared-Memory (llhoft) Test Run

Self-contained run directory for testing the PyCWB online workflow reading
1-second GWF files from the LIGO low-latency shared-memory ring buffer.

## Directory contents

| File | Purpose |
|------|---------|
| `user_parameters.yaml` | Full run configuration (edit before running) |
| `online_schema_extension.yaml` | YAML schema for online extension params |
| `run.sh` | Convenience launch script |

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

## Output

All output lands in `output/` (created automatically):
- `output/catalog.json` — local trigger catalog (appended continuously)
- `output/online_state.json` — crash-recovery checkpoint
- `output/` — per-trigger JSON + waveforms if `save_waveform: true`

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
