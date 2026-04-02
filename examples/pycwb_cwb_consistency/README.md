```bash
python examples/pycwb_cwb_consistency/compare_pycwb_vs_cwb.py \
  --parquet tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/catalog/catalog4.M1.parquet \
  --progress tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/catalog/progress4.M1.parquet \
  --root tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/catalog/wave_O4b0_BCK_C00_LH_BurstLF_SIM0_run1.M1.root \
  --live tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/catalog/live_O4b0_BCK_C00_LH_BurstLF_SIM0_run1.M1.root \
  --log tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/log4 \
  --ref_events tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/Events.csv \
  --ifo L1 H1 --tol 0.05
```

## O4b0 MDC short0 (SIM0)

```bash
python examples/pycwb_cwb_consistency/compare_pycwb_vs_cwb.py \
  --parquet tests/postprod/O4b0_MDC/short0/catalog4.M1.parquet \
  --progress tests/postprod/O4b0_MDC/short0/progress4.M1.parquet \
  --root tests/postprod/O4b0_MDC/short0/wave_O4b0_BCK_C00_LH_BurstLF_SIM0_run1.M1.root \
  --live tests/postprod/O4b0_MDC/short0/live_O4b0_BCK_C00_LH_BurstLF_SIM0_run1.M1.root \
  --log tests/postprod/O4b0_MDC/short0/log4 \
  --out report_consistency_short0.pdf \
  --csv matched_triggers_short0.csv \
  --ifo L1 H1 --tol 0.05
```

## O4b0 MDC short1 (SIM1)

```bash
python examples/pycwb_cwb_consistency/compare_pycwb_vs_cwb.py \
  --parquet tests/postprod/O4b0_MDC/short1/catalog.M1.parquet \
  --progress tests/postprod/O4b0_MDC/short1/progress.M1.parquet \
  --root tests/postprod/O4b0_MDC/short1/wave_O4b0_BCK_C00_LH_BurstLF_SIM1_run1.M1.root \
  --live tests/postprod/O4b0_MDC/short1/live_O4b0_BCK_C00_LH_BurstLF_SIM1_run1.M1.root \
  --log tests/postprod/O4b0_MDC/short1/log \
  --out report_consistency_short1.pdf \
  --csv matched_triggers_short1.csv \
  --ifo L1 H1 --tol 0.05
```


