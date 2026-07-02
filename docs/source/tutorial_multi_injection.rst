.. _tutorial_multi_injection:

Performing Multi-Injection
--------------------------

Multi-injection runs are configured by providing a list under
``injection.parameters``. The scheduler attaches each injection to the job
segment that contains its GPS time.

The repository includes a complete example in ``examples/multiple_injection``:

.. code-block:: bash

   cd examples/multiple_injection
   pycwb run user_parameters_injection.yaml

For larger simulation campaigns, use the newer scheduled-injection options
shown in ``examples/new_injection_infra_with_gaussian_noise`` and related
example folders. Those examples support repeated injections, sky and time
distributions, generated Gaussian noise, and real-data injections.
