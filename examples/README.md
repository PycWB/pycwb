# Examples for the current PycWB API

Install this source checkout in the Python environment used by the CLI or
notebook kernel (`python -m pip install .` from the repository root). Start with
[demo](demo/README.md) or [the tutorial suite](tutorials/README.md). The settings
are educational; successful execution does not establish search sensitivity or
statistical significance.

For a directory-based example, copy the entire directory to a fresh work area,
run commands from that copy, and inspect `pycwb run CONFIG --list-jobs` first.
Keep the parameter generators, schema files and `input/` files together. Full
population and time-slide configurations can be expensive: `--jobs`,
`--trial-idx` and `--lags` select a bounded first run. Use `pycwb progress
--work-dir RUN` to confirm completion. A compatible WDM cross-talk catalog is
required and can be downloaded by configuration loading.

## Choose an example

| Directory | Purpose and prerequisites |
| --- | --- |
| `demo` | Compact, seeded synthetic search, recovery check and plots; no detector-data download |
| `tutorials` | Compact synthetic, sky-mask, population, custom-network/waveform, public-data, background and postprocessing lessons |
| `injection`, `lvk_sep_2023` | Current native pipeline stages in notebooks; compact CBC injection and reconstruction |
| `colab` | Three equivalent notebook entry points for GW150914 on O1 LOSC V1 public data; about 240 MB of downloads |
| `autoencoder` | Native injection notebook plus historical BLIP classifier; optional TensorFlow (checked with TensorFlow 2.21/Keras 3) |
| `waveform_reconstruction` | Self-contained search saving a cluster, then reconstruction from that saved cluster |
| `injection_with_coordinate_system` | Parameter-only notebook for time and sky distributions; no search is run |
| `batch_injection` | Ten standard CBC waveforms; private hyperbolic extension documented separately |
| `multiple_injection` | Two CBC signals in one seeded-noise segment; run `pycwb run user_parameters_injection.yaml` |
| `pyseobnr_injection` | Optional SEOBNRv5HM generator; see its environment instructions |
| `new_injection_infra_with_gaussian_noise` | Scheduled CBC population with analytic detector PSDs |
| `new_injection_infra_with_LHV` | Three-detector population; bundled O4 PSD files; analysis notebook consumes the resulting Parquet catalog |
| `new_injection_infra_with_sky_patch` | Population in a sky patch; analysis notebook consumes a Parquet catalog |
| `new_injection_infra_with_real_data`, `data_injection` | CBC injections into public strain; download the configured interval first |
| `sine_gaussian_injection`, `white_noise_burst_injection` | Large burst populations in public strain; `burst-waveform` generators and GWOSC downloads |
| `GW190521_search`, `gwosc` | Public-event searches; download frames and DQ intervals first |
| `mesa_testing` | Native MESA whitening with optional `memspectrum`; real-data configuration and custom schema example |
| `custom_workflow` | Custom native segment processor and schema |
| `catalog` | Parquet queries, angular residuals and unique injection-recovery accounting; SGE study products needed for Q/frequency plots |
| `postproduction` | Workflow templates requiring completed background/simulation studies, truth summaries and configured paths |
| `performance` | Execution-profile YAML fragment to merge into a complete search configuration |
| `online_shm_run` | Local fake-frame checks and continuous shared-memory search; PyCBC, GWpy and a GWF writer for the fake generator |
| `cwb_results_conversion` | ROOT-to-Parquet and XGBoost tools; external reference/study files required |
| `pycwb_cwb_consistency` | Compare supplied cWB ROOT and PycWB Parquet results; read its input requirements |
| `search_animation` | Educational animation and still renderer; run from the repository root |

## Notebooks

Launch Jupyter from this checkout and select the environment where this version
of PycWB is installed. Synthetic notebooks use a compact 128-second lesson even
where the adjacent YAML shows a larger campaign. Each run creates a fresh
`notebook-runs/` directory; choose a new parent with `PYCWB_EXAMPLES_WORK_DIR`
when repeating a lesson. `PYCWB_EXAMPLES_XTALK` can point to an existing compatible
cross-talk catalog. No notebook requires ROOT. Catalog-analysis notebooks use
`PYCWB_EXAMPLE_RUN` to locate a completed search directory.

The Colab notebooks require the matching source checkout too: clone it, install
it, change into it, then execute the lesson. Optional integrations must be
installed in the same kernel environment. Outputs are cleared in Git so the
figures and statistics always come from the current execution.

## Validation

`python -m pytest tests/test_example_inputs.py` checks every complete example
configuration, injection parameter loading and the GWOSC path/channel contracts.
It does not execute a full scientific campaign. Follow the individual README or
notebook for an actual search and inspect its recorded progress and products.
The numbered Markdown files at the top of `docs/` are archived ROOT-era design
notes; current executable documentation is under `docs/source/`.
