```bash
python examples/pycwb_cwb_consistency/compare_pycwb_vs_cwb.py --parquet tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/catalog/catalog.M1.parquet \
  --root    tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/catalog/wave_O4b0_BCK_C00_LH_BurstLF_SIM0_run1.M1.root \
  --log tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/log \
  --ref_events tests/postprod/O4_K21b0_C00_BurstLF_LH_SIM_MDC_short0_test5/Events.csv \
  --ifo L1 H1 --tol 0.05
```