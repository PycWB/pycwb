# pycWB Coherent Search Animation

This example renders a didactic animation of the pycWB coherent burst search,
from detector projection to a time-slide significance estimate. Every panel is
computed from simulated data by `search_model.py`; nothing is painted by hand.

| # | Scene | What is computed | pycWB counterpart |
|---|-------|------------------|-------------------|
| 1 | projection | `h+` projected into H1/L1 with PyCBC: `F+`, `Fx`, arrival delay | injection (`modules/injection`) |
| 2 | whitening | aLIGO-shaped coloured noise; per-layer noise RMS `sqrt(0.7191 * median(00^2 + 90^2))` at M = 256; whitened series = mean of 00/90 inverses | `data_conditioning/whitening.py` |
| 3 | WDM | the whitened data at M = 16, 32, 64, 128 | `coherence_native/setup.py` (`l_low..l_high`) |
| 4 | selection | per-detector energy maximised over +-light-travel-time shifts, network sum, threshold `E_o` from `bpp` on noise-only data, neighbour support | `coherence_native` max-energy and pixel selection |
| 5 | clustering | 8-neighbour clusters per resolution | `coherence_native/clustering.py` |
| 6 | supercluster | TF-gap links across resolutions, size and sub-network cuts, defragment (`Tgap`, `Fgap`) | `super_cluster_native` |
| 7 | sky loop | HEALPix scan: sky delays (on a grid 4x finer than the sample rate, like `upTDF = 4`) select time-delayed pixel amplitudes; DPF rotation, regulated projection, `Lo`, `Ec`, null, `cc`, statistic `Lo * cc` | `likelihoodWP/sky_scan.py` |
| 8 | reconstruction | whitened waveform and null stream at the best sky point | `get_network_MRA_wave` |
| 9 | background | 31 circular time slides of L1, `(rho, netcc)` cuts, FAR bound | lag loop in `process_job_segment_native.py` |

The model is a compact version of the cWB 2G statistic, not the production
`likelihoodWP` module. It keeps the dominant polarisation frame, a regulator
on the weak polarisation, coherent and null energies, `cc = Ec / (|Ec| + N)`
and `rho = sqrt(Ec * cc / (nIFO - 1))`. A two-detector network uses the hard
constraint (only the dominant polarisation is fitted; fitting both would
absorb any data and leave no null stream). Larger networks weight the cross
polarisation by `g = |fx|^2 / (|fx|^2 + delta |f+|^2)`.
It leaves out the xtalk/MRA corrections, packet patterns, chirp and Q-veto.
Energies from overlapping resolutions are normalised by the effective number
of resolutions, in the spirit of cWB's `norm`. The sub-network check is
simplified to the energy fraction outside the loudest detector.

With only H1 and L1 the sky map is a ring of constant arrival-time difference.
The recovered H1-L1 delay matches the injected one to within one time-delay
step (-6.50 ms vs -6.52 ms), but the position along the ring is not
constrained. Render with `--ifos H1 L1 V1` to see the V1 delay ring cross the
L1 ring at the source (recovered within 0.4 deg here). Three detectors
switch the default sky grid from nside 16 to 64, because the pixel must match
two delays at once; that run is slower, so use `--lags 8` or fewer.

## Running

Run from the repository root with the development environment:

```bash
conda run -n pycwb-dev-py13 python examples/search_animation/render_search_animation.py
```

The search runs in about a minute, most of it on the 31 time slides.
Rendering the default 60 s video takes about three minutes more.

The default output goes to `examples/search_animation/output/`:

- `pycwb_search_animation.mp4`: 1280x720, 24 fps, 60 s master video
- `pycwb_search_animation.gif`: 854 px wide, 10 fps web preview
- `animation_data.npz`: the window time series, per-resolution TF maps and selections, sky arrays, and background triggers
- `metadata.json`: configuration, per-resolution thresholds and counts, event parameters, reconstruction overlaps and background summary

Quick checks:

```bash
# PNG stills of every scene at 40% and 100% progress, no video
conda run -n pycwb-dev-py13 python examples/search_animation/render_search_animation.py \
  --format --stills 0.4 1.0 --lags 6 --out examples/search_animation/output_stills

# only the sky-loop and background scenes, short and coarse
conda run -n pycwb-dev-py13 python examples/search_animation/render_search_animation.py \
  --scenes sky background --duration 10 --fps 12 --format mp4 \
  --out examples/search_animation/output_tiny
```

Useful options: `--ifos`, `--levels` (must include 64, the display
resolution), `--bpp`, `--noise-sigma`, `--nside`, `--lags`, `--lag-step`,
`--duration`, `--scenes`, `--frames-limit`.

### Home-page hero loop

`--hero` renders a separate 12 s seamless loop for the documentation home
page: whitening, coherent pixel selection and the sky scan, with larger panels
and no stage bar. It skips the time slides, so it takes about 30 s:

```bash
conda run -n pycwb-dev-py13 python examples/search_animation/render_search_animation.py \
  --hero --out docs/source/_static/media
```

This writes `pycwb_hero.mp4` (about 0.45 MB, muted H.264) and
`pycwb_hero_poster.png` (the final frame, shown instead of the video when the
reader prefers reduced motion).

The default signal is `tests/logo/cWB_logo_waveform.txt`, used as `h+` with
`hx = 0` and sky position `(ra, dec) = (1.4, 0.3)` rad. It is placed in an
8 s segment, coloured together with unit-variance white noise (noise level
`--noise-sigma 0.09` relative to the projected peak), and whitened again by
the search. Thresholds are estimated from the noise-only part of the segment.
