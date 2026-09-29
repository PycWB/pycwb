.. _tutorial_search_families:

Adapt a Search to Signal Morphology
===================================

**Question:** Which search choices must change when the signal's duration or
bandwidth changes? Start from a recovered injection in
:doc:`tutorial_population`; keep one source population fixed while comparing
search configurations.

Choose the scientific change
----------------------------

.. list-table::
   :header-rows: 1

   * - Scenario
     - Inspect and change together
     - Observable result
   * - A compact low-frequency burst or CBC
     - Input sample rate, retained band, wavelet levels and clustering gaps.
     - Reconstructed duration/bandwidth and recovery at fixed amplitude.
   * - A high-frequency burst or ringdown
     - Sample rate and Nyquist limit, ``fHigh``, resampling, time-frequency levels and cross-talk catalog.
     - Whether the injected band survives preparation and contributes to the event.
   * - A long-duration transient
     - Waveform support, segment length and edges, time resolution, clustering gaps and sky/response assumptions.
     - Whether the source is split, truncated or reconstructed as one candidate.

The ``pycwb-config`` repository organizes corresponding templates under
``config/BurstLF``, ``BurstHF``, ``BurstLD`` and ``BBH``. Its templates are
production starting points with data/chunk substitutions; they are not drop-in
replacements for the small tutorial YAML. See :doc:`config_repository`.

Perform a small band-selection experiment
-----------------------------------------

.. code-block:: python

   from pathlib import Path
   import yaml

   source = Path("tutorial-work/all_sky.yaml")
   config = yaml.safe_load(source.read_text())
   config["fLow"] = 220.0
   Path("tutorial-work/band_exclusion.yaml").write_text(yaml.safe_dump(config))

.. code-block:: bash

   pycwb validate tutorial-work/band_exclusion.yaml
   pycwb run tutorial-work/band_exclusion.yaml \
     --work-dir tutorial-work/runs/band_exclusion

The original source is centered at 150 Hz. Inspect the recovered waveform and
trigger statistics when the retained band starts above its center. A finite
waveform has spectral width; do not treat the center frequency as a hard edge
or promise that the catalog will be empty. Compare this run with ``all_sky``
using :doc:`tutorial_comparisons`.

Move to another morphology
--------------------------

Select a source from the burst or CBC generators in
:doc:`tutorial_customized_wf_gen`. First plot the generated waveform and its
spectrum. Then choose analysis settings that contain its support and band.
When changing WDM levels, use a compatible cross-talk catalog; changing only
``fHigh`` does not add missing resolutions or increase the sample rate.

Keep numerical search modes, packet patterns, regulators, chirp features and
Q-veto conventions explicit. Their available combinations depend on the
processor; use :doc:`likelihood_guide` and :doc:`backends` before comparing them.
For very long signals, check the scope of the detector-response and sky-delay
model rather than assuming larger segments alone provide the required model.

**Result to keep:** a source plot, an annotated configuration difference and
a recovery comparison for the same injected population.
