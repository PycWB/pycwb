.. _tutorial_customized_wf_gen:

Injections with Customized Waveform Generation
==============================================

Prerequisite: :doc:`tutorial_population`.

PycWB injection configurations can use waveform parameters directly, parameters
returned by a Python function, or a custom waveform generator. The useful
examples to start from are:

.. list-table::
   :header-rows: 1

   * - Example
     - Use case
   * - ``examples/sine_gaussian_injection``
     - Burst waveform injections using ``burst-waveform``.
   * - ``examples/white_noise_burst_injection``
     - White-noise-burst injections on real data.
   * - ``examples/pyseobnr_injection``
     - Custom waveform module and Python parameter generator.

Run a small custom generator
----------------------------

The shared input preparation writes ``custom_waveform.yaml`` with an absolute
function path to ``examples/tutorials/waveform.py``. This example generates a
linearly polarized sine-Gaussian normalized to a requested source hrss:

.. literalinclude:: ../../examples/tutorials/waveform.py
   :language: python

.. code-block:: bash

   pycwb validate tutorial-work/custom_waveform.yaml
   pycwb run tutorial-work/custom_waveform.yaml \
     --work-dir tutorial-work/runs/custom_waveform

Inspect its injection and reconstruction products with
:doc:`tutorial_event_inspection`. Its polarization differs from the base SGE
example, so equal source hrss need not give equal detector response.

The production generator contract
---------------------------------

``injection.generator`` is a function-path string. For an importable module:

.. code-block:: yaml

   injection:
     generator: my_waveforms.get_td_waveform

The native injection path calls the generator with the source parameters as
keyword arguments, including ``delta_t``. Return a dictionary with
``type: polarizations``, ``hp`` and ``hc`` time series. Their sample interval,
epoch and support must match the declared waveform. PycWB then projects the
polarizations using the configured detector geometry, sky position and GPS time.
The older tuple ``(hp, hc)`` return is deprecated.

For already projected detector strains, return ``type: strain`` and one time
series for every configured detector. This bypasses polarization projection.
``pycwb.modules.injection.inj_generators.get_strain_from_file`` supplies this
route for supported HDF/NumPy/text inputs. Preserve or explicitly set sampling
and epoch metadata: a plain array is not a complete physical time series.

Parameter generation and waveform generation are separate. A population
function returns source parameter records; the waveform function turns one
record into samples. ``parameters_from_python.function`` selects the population
function; see :doc:`injection_infrastructure` for its path convention.

For real-data injection, first fetch the configured GWOSC files:

.. code-block:: bash

   pycwb gwosc-data user_parameters.yaml
   pycwb run user_parameters.yaml


----

You have learned
----------------

- ✅ How to use built-in waveform types: burst-waveform, white-noise-burst
- ✅ How to configure custom waveform generators from Python modules
- ✅ How to fetch GWOSC data for real-data injection runs
- ✅ When to use parameter-from-Python vs. static parameter lists

**Apply this to your analysis:** :doc:`injection_infrastructure` covers the
general population interface; :ref:`cluster_injection_campaigns` covers batch
execution with your own inputs.
