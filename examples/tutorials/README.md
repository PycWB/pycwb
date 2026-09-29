# Research tutorial inputs

From the repository root, in an installed PycWB environment:

```bash
python examples/tutorials/prepare.py
pycwb run tutorial-work/patch.yaml --work-dir tutorial-work/runs/patch
```

The preparation script creates a new directory and refuses to overwrite an
existing one. Use `--output another-directory` for an independent experiment.
It requires PyYAML, NumPy and healpy. Searches use the dependencies of
`examples/demo`, including `burst-waveform` and a WDM cross-talk catalog.
Inputs containing file references record their absolute locations; regenerate
them after moving the exercise to a different machine.

Configurations share the demo's seeded noise, source and inexpensive sky grid.
`fixed`, `patch`, `offset`, and `custom` vary only the search mask. `hlv` and
`custom_network` add a detector with the same default analytic noise model;
these are geometry experiments, not forecasts for actual instrument sensitivity.
`population` has four separated signals of different amplitudes. `gated` enables
time-veto diagnostics. `bounded` selects a resource-bounded execution profile.
`custom_waveform` uses the local illustrative polarization generator.

`open_data` requires `pycwb gwosc-data tutorial-work/open_data.yaml --work-dir
tutorial-work` before running. `background` reuses those frames with a few
nonzero time slides. This is a small educational background, not an estimate
of a published event's significance.

See the documentation's Tutorials section for the exercises and interpretation.
