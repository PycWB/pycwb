.. _troubleshooting:

Troubleshooting
===============

Start by recording the package version and environment inventory, then check
declared dependency consistency:

.. code-block:: bash

   pycwb --version
   pycwb doctor
   python -m pip check

``doctor`` lists installed package metadata; it does not probe imports or
certify that a chosen backend can run.

If Python cannot import PycWB at all, ``doctor`` may not start. Confirm the active
interpreter with ``python -c "import sys; print(sys.executable)"`` and reinstall
into the environment described by :ref:`installing_pycwb`.

Command not found or unrecognized command
-----------------------------------------

Activate the environment in which PycWB was installed. Try
``python -m pycwb --help`` with the current source version. An older release may
lack ``doctor`` or ``validate``; select matching release
documentation or install the development checkout. Avoid mixing a new tutorial
with an older environment.

Configuration rejected
----------------------

.. code-block:: bash

   pycwb validate user_parameters.yaml

This check does not download cross-talk files, open detector frames, or run
waveform generators. Render configuration templates first. The detector key is
``ifo``, not ``ifos``. ``iwindow`` is the full time window in seconds. For sky
masks and distributions, include explicit angle units and the coordinate frame.
See :ref:`schema` and :ref:`coordinate_systems_angles`.

Cross-talk catalog unavailable
------------------------------

A first run downloads a missing catalog from the public PycWB cross-talk data
repository. On an offline machine, copy a compatible catalog from an online
machine and set ``filter_dir`` and ``wdmXTalk`` explicitly in your YAML. Record the checksum and verify
that the catalog covers ``l_low`` through ``l_high``.

If the download was interrupted, preserve the error log and replace the
incomplete file with a verified copy. Repeatedly starting the same search does
not repair a corrupt existing catalog automatically.

Slow first run or memory exhaustion
-----------------------------------

The first run compiles Numba/JAX kernels and may build cross-talk caches. Compare
warm runs separately from cold runs. For the demo, use one worker and its bundled
sky resolution. Adding workers can increase memory use. For production, measure
a representative segment before choosing worker counts and scheduler memory.
See :ref:`dev_performance` and :ref:`tutorial_search` for execution planning.

No recovered events
-------------------

First check completion:

.. code-block:: bash

   pycwb progress --work-dir my_first_search --verbose

Inspect the run log, injection time, detector network and search band. For a
modified example, a faint injection can legitimately be missed. Do not change
production thresholds merely to force a detection. Compare with a fresh,
unmodified demo to separate installation problems from analysis choices.

Missing frames or data-quality files
------------------------------------

Check that frame lists and data-quality intervals cover the requested GPS range
in every detector. Relative paths are interpreted by the configured workflow;
run from the intended working directory or provide explicit paths. Public GWOSC
data and collaboration-restricted frame services have different access paths.

Interrupted production or existing output
-----------------------------------------

For a tutorial, create a fresh directory so results are unambiguous. Production
recovery requires identifying completed job/trial/lag tuples with ``progress``
and preserving configuration and batch membership. ``--jobs``, ``--trial-idx``
and ``--lags`` select work; consult their exact syntax in :ref:`cli_reference`.
Do not treat ``--force-overwrite`` as a general resume command. Archive existing
results and test the intended recovery selection on a copy before merging.

Manifest missing or mismatched
------------------------------

Copy the referenced ``jobs.parquet`` together with the catalog and preserve the
relative layout. A missing manifest is a provenance error, not an empty job
list. Do not remove catalog metadata to suppress it. See :ref:`reproducibility`.

Still stuck
-----------

Use :ref:`support` and include the failing command, version, minimal YAML,
relevant traceback, and whether the CLI example succeeds.
