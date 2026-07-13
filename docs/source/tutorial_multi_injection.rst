.. _tutorial_multi_injection:

Performing Multi-Injection
--------------------------

⭐ Intermediate  ·  ~25 min  ·  Prerequisites: :doc:`tutorial_injection`

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


----

You have learned
----------------

- ✅ How to configure multiple injections with a parameter list
- ✅ How GPS-time scheduling assigns injections to job segments
- ✅ The difference between simple parameter lists and scheduled-injection options
- ✅ When to use the newer injection infrastructure for large campaigns

**Next:** :doc:`tutorial_customized_wf_gen` — use custom waveform generators
