.. _online_search:

Configure a Streaming Search
============================

Use this guide to configure frame ingestion, overlapping analysis windows,
local outputs and restart behavior. Start from a working offline environment
with frame-reading and frame-writing dependencies. The example below provides
a local replay before connecting an operational data source.

.. _online_search_local_stream:

Prepare a local stream
----------------------

Create ``stream-work`` and copy ``examples/online_shm_run/user_parameters.yaml`` to
``stream-work/online.yaml`` and copy ``online_schema_extension.yaml`` beside
it. Edit these settings in the copied YAML:

* Set ``online_data_source.base_path`` to the absolute path of
  ``stream-work/frames``.
* Set ``pycwb_schema.schema_file`` to the absolute path of the copied schema.
* Keep ``inRate: 16384`` and ``levelR: 3`` from the example. Generate frames
  at the same 16384-Hz rate so buffer timing and sample counts agree.
* Keep ``ifo: [H1, L1]`` and the ``online_channels`` ending in
  ``GDS-CALIB_STRAIN_CLEAN`` to match the generator below.
* Set ``online_alert.gracedb: false``, ``online_alert.webhook_url: ""`` and
  ``online_background.enabled: false`` for this local exercise.

Start producing frames:

.. code-block:: bash

   python examples/online_shm_run/fake_data_generator.py \
     --duration 180 --sample-rate 16384 --ifos H1 L1 \
     --channel GDS-CALIB_STRAIN_CLEAN --shm-base stream-work/frames \
     --realtime

The file-source name is ``shm`` even when this local demonstration uses a
normal directory. Start the reader in another terminal while new frames arrive:

.. code-block:: bash

   pycwb online stream-work/online.yaml \
     --work-dir stream-work/runs/online --n-workers 1

Keep the example background estimator disabled until you separately configure
and assess that part of the workflow.

.. _online_search_restart:

Observe scheduling and restart
------------------------------

Compare ``online_segment_duration`` with ``online_segment_stride`` and
``segEdge``. Inspect queued/completed segments, missing-frame handling,
local trigger products and the configured state file. Stop and restart the
reader with the same work directory; inspect its recovery log and whether
previously handled candidates are repeated. Do not infer recovery only from
the process starting successfully.

Use the generated stream to inspect measured latency. Shorter strides create
more overlapping work; they do not guarantee lower total latency if processing
cannot keep up. Local ranking thresholds alone do not establish a calibrated
online false-alarm rate.

**Keep with the run:** a record of input frame times, completed windows, emitted
candidates and restart behavior. The example README describes deployment
inputs; match it to the current code before using a live data source.
