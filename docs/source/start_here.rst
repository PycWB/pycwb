.. _start_here:

Your First Search
=================

PycWB searches for coherent transient signals across gravitational-wave
detectors. This tutorial recovers one deliberately loud simulated sine-Gaussian
burst in generated Gaussian noise. It needs no detector data, collaboration
account or ROOT installation. The example YAML lives in the source checkout
under ``examples/demo/`` and runs through the ordinary PycWB CLI.

Install the version described by this documentation using :ref:`installing_pycwb`.
The ``doctor`` and ``validate`` commands are new development features;
older PyPI releases do not contain them. Use the source installation until a
release containing these commands is available.

Check the environment
---------------------

.. code-block:: bash

   pycwb --version
   pycwb doctor

``doctor`` records the interpreter, platform and installed package versions.
It reads distribution metadata and does not check backend imports, devices or
pipeline readiness. Its zero exit status means the report was generated. Use
``python -m pip check`` for declared dependency consistency. See :ref:`troubleshooting`.

Create and run the example
--------------------------

From the source checkout, validate the YAML and run it in a fresh working directory:

.. code-block:: bash

   pycwb validate examples/demo/user_parameters.yaml
   pycwb run examples/demo/user_parameters.yaml --work-dir my_first_search

The pipeline generates and processes one 128-second synthetic segment with H1
and L1. No example-specific Python script is needed. Use a new working directory
for each tutorial run.

The first execution may download the approximately 53 MiB cross-talk catalog
and compile numerical kernels. Allow several minutes and several GiB of free
memory; runtime depends on the CPU and compilation cache. No real strain data
are downloaded. To reuse a compatible local cross-talk catalog, copy the YAML,
set ``filter_dir`` to its directory and ``wdmXTalk`` to its filename, then
validate and run that copy.

Inspect the result
------------------

.. code-block:: bash

   pycwb progress --work-dir my_first_search

.. code-block:: python

   import pandas as pd

   events = pd.read_parquet("my_first_search/catalog/catalog.parquet")
   print(events[["time_H1", "time_L1", "rho", "net_cc"]])

The injected burst is centered at GPS 1126259526 and 150 Hz. Open the waveform
plots under ``my_first_search/trigger/`` to compare the reconstructed detector
signals. Read :ref:`understanding_results` for field meanings and output layout.

The CLI prints its run log to the terminal. ``pycwb progress`` reports completion,
not whether an injection was recovered. Inspect the catalog for a finite ``rho``
above the configured threshold and detector times near the injection. The
automated smoke test checks for a trigger within one second in both detectors;
this is not a false-alarm probability or precision-validation claim. If the run
fails, use :ref:`troubleshooting`; an incomplete run is not zero detections.

Change one setting
------------------

From the source checkout, copy the YAML to change one setting:

.. code-block:: bash

   cp examples/demo/user_parameters.yaml quieter_parameters.yaml

Edit ``injection.parameters.hrss`` in ``quieter_parameters.yaml`` from ``1.0e-21``
to ``5.0e-22``. This halves the injected source amplitude while keeping the noise
seeds, sky position and search settings fixed. Then run:

.. code-block:: bash

   pycwb validate quieter_parameters.yaml
   pycwb run quieter_parameters.yaml --work-dir quieter_search
   pycwb progress --work-dir quieter_search

Compare the reconstructed waveform amplitude and ranking statistic. A sufficiently
weak injection may not be recovered. Estimating efficiency requires many
injections; one recovery cannot establish sensitivity or significance.

Continue learning
-----------------

* :ref:`understanding_results`: understand the catalog and plots.
* :ref:`reproducibility`: preserve the configuration, inputs and environment.
* :ref:`tutorials`: inspect pipeline stages and build larger searches.
  The Colab notebooks demonstrate individual Python stages on GW150914 open data;
  they serve a different purpose from this synthetic CLI example.
* :ref:`support`: ask a question or report a reproducible problem.
