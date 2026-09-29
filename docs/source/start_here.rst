.. _start_here:

Your First Search
=================

PycWB searches for coherent transient signals across gravitational-wave
detectors. This tutorial recovers one deliberately loud simulated sine-Gaussian
burst in generated Gaussian noise. It needs no detector data, collaboration
account or ROOT installation. The example script and YAML live in the source
checkout under ``examples/demo/``.

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

From the source checkout, record the script path so it also works after changing
directories:

.. code-block:: bash

   demo_script="$(pwd)/examples/demo/run_demo.py"
   python "$demo_script" my_first_search --run

The script copies its adjacent example YAML, uses the installed PycWB to run one
128-second synthetic segment with H1 and L1, and checks recovery. It refuses to
overwrite an existing directory. Use another name for a new run.

The first execution may download the approximately 53 MiB cross-talk catalog
and compile numerical kernels. Allow several minutes and several GiB of free
memory; runtime depends on the CPU and compilation cache. No real strain data
are downloaded. To reuse an existing compatible catalog:

.. code-block:: bash

   python "$demo_script" another_search --run --xtalk /path/to/OverlapCatalog16-1024.bin

A successful run ends with JSON containing ``"ok": true`` and
``"recovered_triggers"`` of at least one. It also writes ``demo-result.json``.
The check requires a completed zero-lag job and a finite trigger above the
configured threshold within one second of the injection in both detectors.
This tolerance allows for detector arrival delays and reconstruction timing;
it is not a false-alarm probability or a precision-validation claim.

Inspect the result
------------------

.. code-block:: bash

   pycwb progress --work-dir my_first_search
   python "$demo_script" my_first_search --check

.. code-block:: python

   import pandas as pd

   events = pd.read_parquet("my_first_search/catalog/catalog.parquet")
   print(events[["time_H1", "time_L1", "rho", "net_cc"]])

The injected burst is centered at GPS 1126259526 and 150 Hz. Open the waveform
plots under ``my_first_search/trigger/`` to compare the reconstructed detector
signals. Read :ref:`understanding_results` for field meanings and output layout.

``my_first_search/log/demo.log`` records the run. If the check fails, use
:ref:`troubleshooting`; do not interpret an incomplete run as zero detections.

Change one setting
------------------

Create another example without executing it:

.. code-block:: bash

   python "$demo_script" quieter_search
   cd quieter_search

Edit ``injection.parameters.hrss`` in ``user_parameters.yaml`` from ``1.0e-21``
to ``5.0e-22``. This halves the injected source amplitude while keeping the noise
seeds, sky position and search settings fixed. Then run:

.. code-block:: bash

   pycwb validate user_parameters.yaml
   pycwb run user_parameters.yaml
   python "$demo_script" . --check

Compare the reconstructed waveform amplitude and ranking statistic. A sufficiently
weak injection may fail the recovery check. Estimating efficiency requires many
injections; one recovery cannot establish sensitivity or significance.

Continue learning
-----------------

* :ref:`understanding_results`: understand the catalog and plots.
* :ref:`reproducibility`: preserve the configuration, inputs and environment.
* :ref:`tutorials`: inspect pipeline stages and build larger searches.
* :ref:`support`: ask a question or report a reproducible problem.
