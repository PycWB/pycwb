.. _tutorial_conditioning:

Inspect Conditioning and Time Vetoes
====================================

**Question:** What changes the strain, and what excludes times from analysis?
The native pipeline conditions data before pixel selection. The hook stages
``conditioning.post_whitening`` and ``selection.time_vetoes`` let a configured
module apply a correction or return excluded intervals.

See a deterministic veto
------------------------

This small exercise creates a 64-second sampled stream with one large excursion.
It does not run a search or represent calibrated detector noise.

.. code-block:: bash

   python examples/tutorials/gating_demo.py

.. literalinclude:: ../../examples/tutorials/gating_demo.py
   :language: python
   :start-at: samples =

The output is an excluded interval ``[1028, 1032]``, accepted intervals
``[1010, 1028]`` and ``[1032, 1054]``, and 40 accepted seconds. The edge
padding has already removed the first and last ten seconds. The strain
comparison prints ``True``: this gate excludes analysis times instead of
zeroing samples.

Enable the gate in a search
---------------------------

The ``gated.yaml`` prepared in :doc:`tutorial_signals` contains:

.. code-block:: yaml

   selection:
     time_vetoes:
       - module: pycwb.modules.conditioning_plugins.cwb_gating
         options:
           energy_threshold: 1000000.0
           integration_seconds: 0.5
           padding_seconds: 1.5

.. code-block:: bash

   pycwb validate tutorial-work/gated.yaml
   pycwb run tutorial-work/gated.yaml --work-dir tutorial-work/runs/gated

Inspect ``conditioning/job_*/trial_*/diagnostics.json`` beneath that run.
It records plugin options, module hashes and excluded GPS intervals. Compare
the catalog and progress with ``all_sky``. The seeded demo may produce no
excluded intervals at this threshold; that is a valid observation. A lower
threshold can remove real signals as well as noise excursions.

Distinguish correction from selection
-------------------------------------

The bundled ``pycwb.modules.conditioning_plugins.o3a_conditioning`` is a
specific 16–48-Hz variability correction. It belongs in
``conditioning.post_whitening`` and can select detectors through
``options.detectors``. It also updates the noise information used downstream.
It is not a general recommendation for every observing period or frequency
band. Use a representative dataset and inspect its noise-variation diagnostics
before applying it to an analysis.

Wavelet whitening and the optional MESA path are described in
:doc:`data_conditioning`; parameter defaults are in :doc:`schema`. When comparing conditioning choices, use
the same input stream and compare raw/conditioned spectra, recovered signals,
background and accepted exposure. A visually flatter spectrum alone does not
demonstrate better search sensitivity.

**Result to keep:** the veto intervals and diagnostics, a before/after recovery
comparison, and an explicit account of any lost exposure.
