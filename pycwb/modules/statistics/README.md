# PycWB Statistics Module

Tools for reading, fitting, and plotting cWB detection efficiency results.

## Overview

This module provides two complementary sets of functionality:

| Module | Purpose | Usage |
|--------|---------|-------|
| `sigmoid_fit.py` | **Numerical sigmoid fitting** (active pipeline) | Used by `postprocess/efficiency_metrics.py` and `postprocess/plot_efficiency.py` via `iminuit` |
| `eff.py` | **cWB results parsing toolkit** (standalone) | Read injection lists, fit-parameter files, compute hrss percentiles, generate efficiency plots |
| `eff_plot.py` | **Standalone bar-plot helper** | Grouped bar chart from pre-loaded hrss50 data |
| `merge.py` | **Chunk merging utility** | Merge cWB `eff_*.txt` chunks into per-waveform DataFrames |

## Module Details

### `sigmoid_fit.py` — Vectorized Sigmoid Fitting

Numerically optimized sigmoid fitting for cWB efficiency curves using `iminuit.Minuit`.

- **`logNfit(x, par0, par1, par2, par3, par4)`** — Vectorized sigmoid evaluation. Computes detection efficiency at given hrss values. Fully vectorized over `x`.
- **`fit(xdata, ydata, debug=False)`** — Fit a sigmoid to efficiency data using Minuit. Tries both sigmoid orientations (`par4=0,1`) and returns the best-fit parameters `[chi2, hrss50, hrssEr, sigma, betam, betap, flag]`.
- **`estimate_hrss(params, xlim, target_dp)`** — Estimate hrss at a target detection probability by inverting the fitted sigmoid via Brent's method.

### `eff.py` — cWB Results Parsing Toolkit

Standalone toolkit for reading and plotting cWB efficiency results. The `logNfit` function in this module is **deprecated** — use `sigmoid_fit.logNfit` instead. The reading and plotting functions remain active.

- **`read_inj_type(file_name)`** — Parse cWB injection-type definition file (ASCII).
- **`read_fit_parameters(filepath)`** — Parse cWB `fit_parameters_*.txt` file.
- **`logNfit(x, ...)`** — ⚠️ Deprecated. Use `sigmoid_fit.logNfit`.
- **`get_hrss_from_percentile(percentile, ...)`** — Compute hrss at a given efficiency percentile via root-finding.
- **`read_hrss_for_mdc(run_dir, ...)`** — Aggregate hrss10/50/90 for all injection sets in an MDC run.
- **`plot_hrss_from_mdc(run_dirs, tags, ...)`** — Log-log hrss50 vs frequency plot with error bars.
- **`barplot_hrss_from_mdc(run_dirs, tags, ...)`** — Grouped bar chart comparing hrss50 across runs.
- **`sort_key(s)`** — Sort key for waveform names (e.g. `"SG4Q9"`).

### `eff_plot.py` — Standalone Bar Plot

- **`hrss50_bar_plot(data_sets, ...)`** — Grouped bar chart from pre-loaded data (does not read from disk). Accepts `list[tuple[dict, str]]` where each dict maps waveform name → `[hrss50, hrss50_err]`.

### `merge.py` — Chunk Merging

- **`read_data_file(file_path, i)`** — Read a single cWB `eff_*.txt` chunk into a DataFrame.
- **`get_evt_vs_inj(chunks, wf_selections)`** — Merge multiple chunks, compute `evt_total`/`inj_total` columns per waveform.

## Dependencies

- **Pipeline (sigmoid_fit)**: `numpy`, `scipy`, `iminuit`
- **Toolkit (eff, eff_plot, merge)**: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`

## Example

```python
from pycwb.modules.statistics.sigmoid_fit import fit, estimate_hrss

# Fit efficiency curve
xdata = [1e-22, 2e-22, 5e-22, 1e-21, 2e-21]
ydata = [0.0, 0.1, 0.5, 0.9, 1.0]
result = fit(xdata, ydata)
chi2, hrss50, hrssEr, sigma, betam, betap, flag = result

# Estimate hrss at 90% detection probability
hrss90 = estimate_hrss([hrss50, sigma, betam, betap, flag],
                       xlim=(-25, -18), target_dp=0.9)
```
