"""
PycWB Statistics Module
========================

Tools for reading, fitting, and plotting cWB detection efficiency results.

**Pipeline (active):**
    - ``sigmoid_fit`` — Vectorized sigmoid fitting via ``iminuit``, used by
      ``postprocess/efficiency_metrics.py`` and ``postprocess/plot_efficiency.py``.

**Toolkit (standalone):**
    - ``eff`` — Read injection lists and fit-parameter files; compute hrss
      percentiles; generate efficiency plots.
    - ``eff_plot`` — Standalone grouped bar chart from pre-loaded data.
    - ``merge`` — Merge cWB ``eff_*.txt`` chunks into per-waveform DataFrames.

Note: ``eff.logNfit`` is deprecated — use ``sigmoid_fit.logNfit`` instead.
"""