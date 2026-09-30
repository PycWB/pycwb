# cWB reference conversion and classifier tools

Run these scripts in a PycWB environment with `uproot`, pandas and XGBoost.
Supply your own cWB `waveburst` ROOT files; the O4 filenames in the CLI defaults
are illustrative study inputs and are not bundled. Explicit paths are preferred:

```bash
python convert_root_to_parquet.py --bkg /path/to/background.root \
  --sim /path/to/simulation.root --nifo 2 \
  --bkg-out bkg_xgb.parquet --sim-out sim_xgb.parquet
python train_xgb.py --bkg bkg_xgb.parquet --sim sim_xgb.parquet \
  --model xgb_model.ubj --config xgb_config.py
```

This flat feature-table format serves the classifier tools; it is not a native
PycWB catalog. Inspect `--help` for each script. Background consistency checks
also require livetime files and a separate Python executable with ROOT/cWB
available (`--root-python`). They read the reference files and write reports;
they do not require changes to any cWB source repository or remote.
