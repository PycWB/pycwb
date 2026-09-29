.. _tutorial_population:

Design an Injection Population
==============================

**Question:** Which sources are recovered, and which are missed?
Prepare the inputs in :doc:`tutorial_signals`. ``population.yaml`` places four
150-Hz signals at separated times with source amplitudes
``1e-23``, ``3e-22``, ``6e-22`` and ``1e-21`` in the same noise interval.
The weak source lets you inspect missed-event accounting alongside loud
recoveries; determine the actual recovered set from the output.

Run and match to truth
----------------------

.. code-block:: bash

   pycwb run tutorial-work/population.yaml \
     --work-dir tutorial-work/runs/population
   pycwb simulation-summary --work-dir tutorial-work/runs/population
   pycwb match-simulations tutorial-work/runs/population/catalog/catalog.parquet \
     tutorial-work/runs/population/catalog/simulations.parquet \
     --how right --output tutorial-work/matched_population.parquet

.. code-block:: python

   import pandas as pd

   truth = pd.read_parquet("tutorial-work/runs/population/catalog/simulations.parquet")
   matched = pd.read_parquet("tutorial-work/matched_population.parquet")
   print(truth.columns.tolist())
   print(truth)
   print(matched.columns.tolist())
   print(matched[["sim_sim_idx", "sim_hrss", "id", "rho"]])
   print("Unmatched sources:", matched.loc[matched["id"].isna(), "sim_sim_idx"].tolist())

A right match retains all scheduled simulations, including those without a
trigger. The matcher resolves competing candidates into unique associations.
See :ref:`reading_simulation_matches` for the output columns and join choices,
and :doc:`postproduction_efficiency` for eligibility and recovery rules.

In this example, the right match retained all four sources: three had a
recovered trigger and the weakest had an empty ``id``.

Change one population choice
----------------------------

Copy the YAML into a new input and change one dimension: amplitude, frequency,
sky position or noise seed. Use a fresh output directory. Four signals are
enough to inspect the workflow but not to fit a reliable efficiency curve.
Repeat across independent trials before following :doc:`postproduction_study`
with your own study catalogs.

For larger populations, use ``parameters_from_python`` with explicit
``sky_distribution`` and ``time_distribution`` settings. The scheduling
reference in :doc:`injection_infrastructure` covers rate/Poisson timing,
repeated trials, real-data injection and supported sky distributions.
The ``examples/multiple_injection`` directory retains the earlier parameter-list
example.

Amplitude and time support
--------------------------

Fixed ``hrss`` controls source waveform amplitude according to the generator's
convention. CBC ``distance`` is a waveform-model parameter. ``target_snr``
requests normalization against detector noise; these are different experiments.
When switching the population to target SNR, remove conflicting amplitude
normalization choices and follow :ref:`tutorial_injection` for the supported
resampling convention. Keep target-SNR and fixed-hrss sources in separate
trials when using the cWB-compatible path.

``t_start`` and ``t_end`` bound the waveform relative to its reference epoch.
Do not shorten them to force a long signal into a job. As a boundary exercise,
move one signal close to a segment edge and inspect its simulation-summary
eligibility and recovered waveform. ``analyze_injection_only`` is a simulation
shortcut, not a way to estimate unbiased background exposure.

Waveform families and stored signals
------------------------------------

Use ``examples/sine_gaussian_injection`` and ``examples/white_noise_burst_injection``
for burst families, and the CBC examples for parameterized binary waveforms.
:doc:`tutorial_customized_wf_gen` shows the generator contract and how to use
stored detector strain. :doc:`tutorial_search_families` explains which search
settings must be revisited when signal duration or bandwidth changes.

**Result to keep:** a table of scheduled sources, eligibility, recovery and
selected statistics, including every missed source.
