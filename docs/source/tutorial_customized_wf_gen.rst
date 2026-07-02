.. _tutorial_customized_wf_gen:

Injections with Customized Waveform Generation
----------------------------------------------

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

Custom generators are configured with the ``injection.generator`` section:

.. code-block:: yaml

   injection:
     parameters_from_python:
       file: "generate_parameters.py"
       function: "get_injection_parameters"
     generator:
       module: "waveform_model/waveform.py"
       function: "get_td_waveform"

The generator function should return detector-frame waveform data compatible
with the injection reader used by the configured waveform module. When injecting
into real data, fetch the required GWOSC files first:

.. code-block:: bash

   pycwb gwosc-data user_parameters.yaml
   pycwb run user_parameters.yaml
