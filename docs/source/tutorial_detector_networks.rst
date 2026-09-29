.. _tutorial_detector_networks:

Change the Detector Network
===========================

**Question:** How do geometry and an additional detector affect reconstruction?
Use the same source as :doc:`tutorial_signals`.

Compare H1/L1 with H1/L1/V1
---------------------------

.. code-block:: bash

   pycwb validate tutorial-work/hlv.yaml
   pycwb run tutorial-work/hlv.yaml --work-dir tutorial-work/runs/hlv

The prepared configuration adds V1 and a third independent noise seed. The
H1/L1 source and noise seeds remain unchanged. For this geometry exercise,
all three streams use the same default analytic noise model; this is not a
forecast of the actual detectors' relative sensitivities.

Compare detector arrival times, ``rho``, ``net_cc``, sky position and saved
waveforms with ``all_sky``. Use :doc:`tutorial_comparisons` to keep both runs
in one report. The change in recovery depends on the added detector's response,
noise and search settings.

For a realistic noise comparison, supply one PSD per detector in the same
order as ``ifo`` under ``injection.segment.noise.psds``. Keep PSD frequency and
spectral-density conventions consistent with :doc:`units_conventions`.
The scheduled-population example ``examples/new_injection_infra_with_LHV``
shows detector-specific PSD files through the population noise interface.

Add an illustrative detector
----------------------------

The preparation script writes ``detectors.json`` and ``custom_network.yaml``:

.. code-block:: yaml

   ifo: [L1, H1, X1]
   refIFO: L1
   detector_definitions_file: detectors.json
   detector_geometry:
     X1: X1:tutorial

The generated YAML uses an absolute JSON path so it also works after run
setup changes the working directory. The JSON defines an illustrative
Earth-fixed interferometer, with a geographic position and two arms. It is
not a surveyed instrument. See :doc:`detector_support` for the full definition,
angle units and geometry validation rules.

.. code-block:: bash

   pycwb validate tutorial-work/custom_network.yaml
   pycwb run tutorial-work/custom_network.yaml \
     --work-dir tutorial-work/runs/custom_network

Inspect the saved detector geometry and reconstructed X1 waveform. For your
own instrument, supply its data and noise model along with its geometry.
The response model assumes fixed ground-based detectors.

**Result to keep:** a comparison of recovery and sky reconstruction for the
three networks, stating which geometry and noise model each run used.
