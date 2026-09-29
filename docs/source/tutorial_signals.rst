.. _tutorial_signals:

Working with Signals
====================

Complete :doc:`start_here`, then choose an experiment below. The synthetic
lessons share one seeded input so changes in recovery can be traced to the
configuration you changed.

Prepare the synthetic exercises
-------------------------------

From the repository root, in your installed PycWB environment:

.. code-block:: bash

   python examples/tutorials/prepare.py
   pycwb validate tutorial-work/all_sky.yaml
   pycwb run tutorial-work/all_sky.yaml --work-dir tutorial-work/runs/all_sky
   pycwb progress --work-dir tutorial-work/runs/all_sky

The preparation script requires NumPy, PyYAML and healpy. It creates complete
YAML files, a binary sky-region FITS file, and an illustrative detector JSON.
It refuses to overwrite an existing directory; use ``--output another-directory``
for a new experiment. Generated file references are absolute: regenerate them
after moving the exercise. Searches also need the dependencies and cross-talk
catalog described in :doc:`start_here`.

The base is ``examples/demo/user_parameters.yaml``: one 128-second H1/L1
interval with a seeded noise realization and a loud 150-Hz sine-Gaussian.
Tutorial settings favor a small workload. Compare measured outputs; they are
not production search settings or promised sensitivity results.

.. toctree::
   :maxdepth: 1

   tutorial_open_data
   tutorial_event_inspection
   tutorial_data_quality
   tutorial_conditioning
   tutorial_population
   tutorial_sky_masks
   tutorial_detector_networks

The complete input preparation code is available as
:download:`prepare.py <../../examples/tutorials/prepare.py>`.
